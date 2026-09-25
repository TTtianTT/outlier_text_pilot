# Outlier Text Pilot：方案一 + 方案二预实验

先回答一个问题：**在同样的语义约束、候选预算和已匹配的文本统计下，高表示异常分候选，能否比普通候选覆盖更多可读出的比特模式？**

本项目包含真实 Hugging Face 模型流水线，以及不下载模型的合成数据自检。`demo` 的向量、语义分和似然均为人工构造，**不能用作任何 LLM 实验结论**。真实模型路径已实现，但交付环境没有 PyTorch、模型权重或可用 GPU，未在这里执行；详细验证范围见 `docs/VALIDATION.md`。

## 1. 第一轮的默认设定

| 项目 | 设置 |
|---|---|
| 生成与表示模型 | Qwen/Qwen2.5-1.5B-Instruct，冻结，不训练 |
| 载体语言 | 英文；默认语义与 NLI 模型均为英文模型 |
| 初始数据 | 附带 90 条编写的句子：30 dev、30 test、30 OOD |
| 领域 | dev/test：日常叙述、产品描述；OOD：科学解释 |
| 候选预算 | 每个原句固定 32 次生成，重复、截断、无效输出都消耗预算 |
| 主表示 | 原始句子 + 固定 BOS/EOS 前缀，最后隐藏层，排除前缀后均值池化，再做 L2 归一化 |
| 语义过滤 | 独立 MiniLM cosine ≥ 0.85；双向 DeBERTa NLI entailment ≥ 0.80 |
| 异常分 | 表示距离减去只在 dev 上拟合的二阶 ridge 预测值 |
| 主对照 | 每个原句内部，outlier 与非 outlier 按六个混杂变量进行严格配对 |
| 对照预算 | 每组 8 个；配不到 8 对时四个组一起记为不可用，不放宽条件 |
| 读出 | 密钥控制的随机符号投影；阈值只取 dev 原句投影的中位数 |
| 评测 | 1/2/3 bit，4 个间隔，20 个公开实验密钥种子 |
| 预先指定主指标 | 1 bit、margin=0.0001，outlier − matched 的覆盖率差 |
| 区间 | 按原始来源 group 与密钥两个维度重采样 2,000 次 |

**这里的“行为距离”先用隐藏表示距离做代理，不代表已经证明下游输出行为变化。**先做这个可复现版本；若出现可靠信号，再用预先固定的任务探针或输出分布差异验证。

90 条数据用于启动；不是此前建议的正式 300 条规模，也不是现成公开 benchmark。不要把同一句生成多个近义版本当成增加独立原句数。扩到 300 条的规则见第 7 节。

## 2. 安装

在 Linux GPU 服务器上解压，进入本目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[hf,plot]'
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

需要 Python 3.10+。PyTorch 必须与服务器驱动兼容；若已配置可用的 PyTorch 环境，可直接使用该环境。首次真实模型运行需要下载三个公开模型；也可以把配置中的 `name` 改为已下载的本地目录。代码不需要 sudo，不调用外部付费 API。

只验证 CPU 数学与文件流程：

```bash
python -m pip install -e '.[plot]'
python -m unittest discover -s tests -v
python -m otc demo --run runs/demo
python -m otc verify-readout --run runs/demo --limit 100
python -m otc plot --run runs/demo
```

`runs/demo/report.md` 和图中会显式写出 SYNTHETIC。请勿把它们写进论文结果表。若目录已存在，用新的 `--run` 名称。

## 3. 第一天：先跑 12 个原句的真实模型检查

```bash
CUDA_VISIBLE_DEVICES=0 python -m otc all \
  --config configs/smoke.yaml --run runs/smoke

CUDA_VISIBLE_DEVICES=0 python -m otc verify-readout \
  --run runs/smoke --limit 50
```

`smoke.yaml` 从每个 split 稳定选 4 个原句，共 12 个；每句仍然固定 32 次生成。它主要检查下载、生成格式、过滤率、匹配可行性和重新推理的一致性。不要用这 12 句判定研究假设。

先检查：

1. `candidate_shards/`：句子是否自然，重复输出是否过多。生成截断、重复、原句照抄都会记录。
2. `selection_diagnostics.jsonl`：有多少原句配到 8 对，失败是候选太少还是匹配太严。
3. `human_audit.csv`：抽查是否保留数字、实体、否定、程度与事实；高 cosine 不保证同义。
4. `readout_verification.json`：接收方重新推理后，受间隔保护的比特有无错误或拒收。

如果配对率极低，先在 **dev** 上解决候选多样性与阈值问题。新建配置和 run，记录变更。不要看完 test/OOD 的 outlier 优势再调参数。若 dev 合格样本不足 24 个，程序会停止拟合，要求先检查上游。

## 4. 方案一：运行 90 句预实验

```bash
CUDA_VISIBLE_DEVICES=0 python -m otc all \
  --config configs/pilot.yaml --run runs/pilot

CUDA_VISIBLE_DEVICES=0 python -m otc verify-readout \
  --run runs/pilot --limit 100

python -m otc plot --run runs/pilot
```

也可以分步运行，便于在 tmux 或调度器中接续：

```bash
python -m otc generate --config configs/pilot.yaml --run runs/pilot
python -m otc features --config configs/pilot.yaml --run runs/pilot
python -m otc evaluate --run runs/pilot
```

生成与特征阶段按原句写缓存，正常中断后可用原命令续跑。当前正在处理的原句可能需要重算；用于报告的成本是最终完成该原句的一次成本，**不含进程崩溃后重做的浪费**。跨任务的总 GPU 消耗请同时保存 Slurm 日志。不要并发运行两个写同一目录的任务。

所有方法使用同一个生成池、相同过滤器和相同入选候选数。`outlier` 和 `matched` 是主配对比较；`random` 与 `nearest` 是辅助基线，它们不保证具有相同的语义/概率分布。这里没有声称复现 SemStamp 或某篇隐写论文的完整算法。

| 输出 | 怎么看 |
|---|---|
| `report.md` | 主设置、失败计数、解释边界 |
| `summary.csv` | 各 bit/margin 下所有方法的覆盖率 |
| `paired_deltas.csv` | outlier − matched 的配对差及探索性区间 |
| `selection_diagnostics.jsonl` | 配对数量、剔除原因、六个变量的配对后差异 |
| `metrics.jsonl` | 每原句 × 密钥 × 方法 × bit × margin 明细 |
| `costs.jsonl` | 全部生成尝试、生成 token、特征与质量评估耗时 |
| `human_audit.csv` | 不提供方法标签的候选人工检查表 |
| `detector.json` | 字符 n-gram 检测器的经验 AUROC，训练/测试原句与密钥隔离 |
| `profile/` | 接收方需要的公开模型配置和 dev 校准向量 |

优先看 `coverage_all`：无法配对的原句按零覆盖计入。`coverage_matched` 是成功配对子集上的结果，两者一起报告。如果绝大部分原句无法配对，“差值为零”不能被解释成充分验证了 outlier 没有用。

一个有价值的继续信号是：主设置的 test 与 OOD 上都有可重复的正差；人工审查与配对平衡没有暴露新的差异；再换一个模型/独立数据，优势仍然存在。CI 只是探索性工具，多个 bit、margin、模型中的最大值不能直接当作预注册结论。

## 5. 方案二：先测随机比特的真实整条恢复

先测 1 字节，即 8 bit。**每句 1 bit，所以一条 1 字节消息需要 8 个载体句。**这不是高效成品通信系统，正是预实验要暴露的成本。

```bash
CUDA_VISIBLE_DEVICES=0 python -m otc codec-bench \
  --run runs/pilot --out runs/pilot_codec_1byte \
  --trials 20 --bytes 1 --bits 1 --margin 0.0001
```

这个命令会：

- 在每个 trial 内给四个方法相同的随机消息、密钥、nonce 和原句顺序。
- 从每个方法的既定 8 个候选中选择正确码字，不重新生成或扩大池。
- 找不到码字就中止整条消息；不重抽 nonce，不跳过难题，不传递候选编号。
- 对成功构造的帧重新调用冻结模型提取特征，再恢复字节并逐字节核对。
- 输出 `trials.csv` 与 `summary.json`，包含完整恢复率、接受帧的 BER、传输开销与运行时间。

再测 4 字节或 OOD，分别用新目录：

```bash
python -m otc codec-bench --run runs/pilot --out runs/pilot_codec_4byte \
  --trials 20 --bytes 4 --bits 1 --margin 0.0001

python -m otc codec-bench --run runs/pilot --out runs/pilot_codec_ood \
  --trials 20 --bytes 1 --bits 1 --margin 0.0001 --split ood
```

初始池如果连单块覆盖率都不高，长消息恢复率很可能迅速下降，这是有效实验结果。不要只挑一次成功的 demo。发送前中止的帧传输 token 为零，但计算成本与失败尝试仍需报告；净载荷要与完整恢复率一起看。`codec-bench` 不包括前期公共候选池生成/特征成本，结合 `costs.jsonl` 报告冷启动与缓存复用成本。

### 手动发送与独立解码

```bash
python -m otc keygen --out experiment.key
python -m otc random-payload --out payload.bin --bytes 1

python -m otc encode --run runs/pilot --key experiment.key \
  --input payload.bin --out packet.json \
  --method outlier --bits 1 --margin 0.0001

# 只有 encode 退出码为 0 且产生 packet.json 时再执行下面两步。
python -m otc decode --profile runs/pilot/profile --key experiment.key \
  --packet packet.json --out recovered.bin

python -m otc check --original payload.bin --recovered recovered.bin \
  --packet packet.json
```

`encode` 退出码 2 表示候选不可编码；只产生失败诊断，不产生有效帧。每次使用新的输出名，避免混用旧结果。接收方只需要：项目代码及依赖、相同公开模型、`profile/`、共享密钥、`packet.json`。它不需要 `features.jsonl`、`vectors.npz`、原句或明文。

`packet.json` 包含公开帧头和一列载体句，帧头含随机 nonce、块宽、长度和配置标识；没有目标比特、候选编号或密钥。实验尚未实现把这些元数据一起隐藏进自然文本，也没有纠错、重同步或抗改写保证。完整 JSON 的 token/字节开销均计入诊断，`profile/` 与模型是预先共享的设置成本。

### 可选认证加密封装

将 `encode` 命令加上 `--mode aead` 即先用 AES-GCM 加密，再编码密文比特。密钥通过 HKDF 分离用途；采用随机 nonce，认证失败不输出解密文件。它引入 16 字节认证标签，加上原消息后通常需要很多载体句，初期池很可能无法承载。

**安全性来自标准 AEAD；本项目不证明文本编码层是安全密码或不可检测隐写。**默认 raw 模式与 `codec-bench` 仅用于均匀随机比特实验，不能用于发送敏感明文。实验密钥文件和 benchmark 的 private_trials.key 不要放进共享结果包。

## 6. 计算与运行建议

- 按单张 GPU 配置。先用 `smoke` 实测显存、候选通过率和时间，再排长任务；这里没有给出未经测量的 GPU 小时承诺。
- `CUDA_VISIBLE_DEVICES=...` 选物理卡后，配置内保持 `cuda:0`。默认 bfloat16；不支持时新建配置改 float32。不要在同一 run 中混用精度。
- `scripts/slurm_example.sh` 提供 4 小时、单卡的模板；提交时自行传合法的 partition/account，代码没有猜测你集群的权限。
- 使用单句特征 forward、eager attention 和固定精度，减少批大小差异。不同 GPU/驱动仍不保证逐比特一致，接收方必须做一致性验证。
- 当前依赖范围刻意限定 Transformers 4.x；项目不是最新版本兼容性保证。首次跑通后执行 `python -m pip freeze > runs/pilot/environment.txt`。
- `main` 方便首次下载；正式实验前把三个 revision 改成不可变 commit。产物会记录实际模型 commit、tokenizer 哈希、运行库版本；不一致时拒绝续跑/解码。

## 7. 扩大到 300 条与第二模型

`data/carriers_en.jsonl` 的格式：

```json
{"id":"test-daily-001","group_id":"source-001","split":"test","domain":"daily","text":"The bus stop is across the street from the post office."}
```

准备 100 dev、100 test、100 OOD 独立句子；同一来源或同一原句的变体必须共享 `group_id` 且留在同一个 split。OOD 主题在 dev 中不出现。程序检查重复原句及跨 split 的 group，不自动识别近重复，仍要人工/语义去重。不要自动展开模板后宣称有 300 个独立样本。

复制配置，修改 `data` 并使用新 run。改模型、层、阈值、候选数也要新 run；初次探索如果使用过测试结果调参，之后需要新的未见测试集。第二个模型实验可以复制配置替换 `model.name`，其结果属于整条生成+表示流水线的复现；若要隔离“相同候选在第二表示模型上能否迁移”，需另设固定候选池实验，不能把这两个问题混为一谈。

## 8. 代码位置与下一步

| 文件 | 作用 |
|---|---|
| `otc/hf.py` | 模型加载、生成、表示与 NLL、独立语义/NLI |
| `otc/pipeline.py` | 固定预算、断点缓存、数据与特征对应 |
| `otc/selection.py` | dev 残差模型、严格匹配与辅助基线 |
| `otc/readout.py` | 密钥派生的投影与确定性符号读出 |
| `otc/evaluate.py` | 覆盖率、配对区间、有限池偏移、弱检测器 |
| `otc/codec.py` | 帧协议、随机比特/AES-GCM、独立接收方 |
| `otc/benchmark.py` | 消息级成功率与 fresh-forward 恢复 |
| `otc/verification.py` | 数值一致性与字节比较 |

当前版本先检验可用性。更强检测器、真正保持分布的采样、功能行为探针、纠错和抗改写等都未假装实现。若主差值没有优势、或者优势在控制语义/词频后消失，优先停在机制结论，不继续包装加密方法。

模型与 API 依据：

- [Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [MiniLM 模型卡与 mean pooling](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [NLI 模型卡，标签顺序为 contradiction / entailment / neutral](https://huggingface.co/cross-encoder/nli-deberta-v3-small)
- [cryptography AEAD 文档](https://cryptography.io/en/latest/hazmat/primitives/aead/)

具体定义、排除规则与结论边界见 `docs/PROTOCOL.md`。
