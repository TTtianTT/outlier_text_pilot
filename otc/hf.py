"""Optional GPU backend. Imports stay lazy so the CPU demo needs no PyTorch."""
from __future__ import annotations
import gc
import hashlib
import importlib.metadata
import os
from pathlib import Path
import numpy as np
from .io import digest, file_hash, unit

PROMPT_VERSION = "paraphrase-v1"
STYLES = ["Use a close synonym where suitable.", "Change the sentence structure slightly.",
          "Use a natural alternative phrasing.", "Keep the wording changes small."]


def paraphrase_messages(text, attempt):
    return [
        {"role": "system", "content": "Rewrite the supplied English sentence. Preserve all facts, numbers, names, negation, and uncertainty. Output exactly one sentence without explanations or quotation marks."},
        {"role": "user", "content": f"{STYLES[attempt % len(STYLES)]}\nSentence: {text}"},
    ]


def model_provenance(model, tokenizer, name, revision):
    local = Path(name)
    weights = {}
    if local.is_dir():
        for p in sorted(local.glob("*")):
            if p.is_file() and p.suffix in {".safetensors", ".bin", ".json", ".model"}:
                weights[p.name] = file_hash(p)
    tok = tokenizer.backend_tokenizer.to_str() if tokenizer.is_fast else repr(tokenizer.get_vocab())
    return {"name": name, "requested_revision": revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "tokenizer_sha256": hashlib.sha256(tok.encode()).hexdigest(),
            "local_file_hashes": weights,
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch")}


class CausalBackend:
    def __init__(self, spec):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.spec = torch, dict(spec)
        self.device = spec["device"]
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        dtype = getattr(torch, spec["dtype"])
        common = {"revision": spec.get("revision", "main"), "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(spec["name"], **common)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec["name"], torch_dtype=dtype, attn_implementation="eager", **common
        ).to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        prefix = self.tokenizer.bos_token_id
        self.prefix_id = prefix if prefix is not None else self.tokenizer.eos_token_id
        if self.prefix_id is None:
            raise ValueError("model requires an explicit BOS or EOS token")
        self.provenance = model_provenance(self.model, self.tokenizer, spec["name"], common["revision"])
        self.signature = {"backend": "hf", "feature_version": "plain-prefix-mean-unit-v1",
                          "spec": self.spec, "prefix_id": self.prefix_id,
                          "provenance": self.provenance}

    def feature(self, text):
        """Single-sentence forward, fixed prefix, no chat prompt, no truncation."""
        torch = self.torch
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids or len(ids) > self.spec["max_tokens"]:
            raise ValueError("empty or overlong carrier; refusing silent truncation")
        if any(t in self.tokenizer.all_special_ids for t in ids):
            raise ValueError("special-token literal in carrier")
        inp = torch.tensor([[self.prefix_id] + ids], device=self.device)
        with torch.inference_mode():
            output = self.model(inp, attention_mask=torch.ones_like(inp),
                                output_hidden_states=True, use_cache=False)
            h = output.hidden_states[self.spec["layer"]][0, 1:].float().mean(dim=0)
            logits = output.logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                      inp[:, 1:].reshape(-1), reduction="sum")
        return unit(h.cpu().numpy()), {"n_tokens": len(ids), "nll_sum": float(loss),
                                      "nll_mean": float(loss) / len(ids)}

    def generate(self, text, attempt, seed, settings):
        torch = self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prompt = self.tokenizer.apply_chat_template(paraphrase_messages(text, attempt),
                                                     tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=True, temperature=settings["temperature"],
                top_p=settings["top_p"], top_k=0,
                max_new_tokens=settings["max_new_tokens"],
                pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
            )
        new = output[0, inputs.input_ids.shape[1]:]
        raw = self.tokenizer.decode(new, skip_special_tokens=True)
        ended = bool(len(new) and int(new[-1]) in set(
            self.model.generation_config.eos_token_id if isinstance(self.model.generation_config.eos_token_id, list)
            else [self.model.generation_config.eos_token_id]))
        return raw, len(new), ended

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class SemanticBackend:
    """Independent MiniLM sentence embedding and bidirectional DeBERTa NLI."""
    def __init__(self, cfg):
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        self.torch, self.cfg = torch, cfg
        self.device = cfg["device"]
        self.stok = AutoTokenizer.from_pretrained(cfg["semantic_model"], revision=cfg["semantic_revision"])
        self.smodel = AutoModel.from_pretrained(cfg["semantic_model"], revision=cfg["semantic_revision"]).to(self.device).eval()
        self.ntok = AutoTokenizer.from_pretrained(cfg["nli_model"], revision=cfg["nli_revision"])
        self.nmodel = AutoModelForSequenceClassification.from_pretrained(
            cfg["nli_model"], revision=cfg["nli_revision"]).to(self.device).eval()
        self.entailment_id = cfg["entailment_id"]
        label = str(self.nmodel.config.id2label.get(self.entailment_id, "")).lower()
        if "contradiction" in label or "neutral" in label:
            raise ValueError("entailment_id conflicts with model label mapping")
        self.cache = {}
        self.provenance = {
            "semantic": model_provenance(self.smodel, self.stok, cfg["semantic_model"], cfg["semantic_revision"]),
            "nli": model_provenance(self.nmodel, self.ntok, cfg["nli_model"], cfg["nli_revision"]),
            "entailment_id": self.entailment_id,
        }

    def embed(self, text):
        if text not in self.cache:
            inputs = self.stok(text, return_tensors="pt", truncation=False).to(self.device)
            if inputs.input_ids.shape[1] > 256:
                raise ValueError("semantic evaluator input over 256 tokens")
            with self.torch.inference_mode():
                out = self.smodel(**inputs).last_hidden_state
                mask = inputs.attention_mask.unsqueeze(-1)
                h = (out * mask).sum(1) / mask.sum(1)
            self.cache[text] = unit(h[0].cpu().numpy())
        return self.cache[text]

    def compare(self, ref, text):
        sim = float(np.clip(self.embed(ref) @ self.embed(text), -1, 1))
        inp = self.ntok([ref, text], [text, ref], padding=True, truncation=False, return_tensors="pt").to(self.device)
        if inp.input_ids.shape[1] > 512:
            raise ValueError("NLI pair over 512 tokens")
        with self.torch.inference_mode():
            prob = self.nmodel(**inp).logits.float().softmax(-1)[:, self.entailment_id].cpu().numpy()
        return {"sem_dist": 1 - sim, "nli_min": float(prob.min()),
                "nli_forward": float(prob[0]), "nli_backward": float(prob[1])}

    def close(self):
        del self.smodel, self.nmodel
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
