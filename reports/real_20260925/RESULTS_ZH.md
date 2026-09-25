# 真实模型实验结果（2026-09-25）

真实 GPU 流程已运行完成，但原始协议下 test/OOD 没有任何成功配对原句，因而本轮无法检验 outlier 相对 matched 的优势。消息实验全部在发送前失败，尚未验证真实接收方的恢复可靠性。

## 运行与复现

- Slurm 最终作业：1576，RTXq / node05，单张 NVIDIA RTX PRO 6000 Blackwell，COMPLETED，退出码 0，耗时 13 分 15 秒。
- 前序作业 1574 因原虚拟环境缺少 activate 脚本退出（1 秒）；1575 跑完 smoke 后因读出状态 no_candidates 返回退出码 3 而结束（2 分 45 秒）。总分配时间约 16 分 1 秒。
- 仅调整环境与任务编排，没有更改实验源码、候选预算、过滤阈值、配对条件或数据。
- 环境复用共享环境的 PyTorch，在项目 .venv 中安装 Transformers 4.x 等依赖；该虚拟环境依赖共享环境路径。完整版本见 [environment.txt](environment.txt)。
- 核心测试：13/13 通过。
- 模型：Qwen2.5-1.5B-Instruct，bfloat16，eager attention；独立 MiniLM 和双向 DeBERTa NLI。
- 实际模型提交：Qwen `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；MiniLM `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`；NLI `fa2804872c3b4bd748f38c0185cc85775361e735`。
- Torch 2.11.0+cu128，Transformers 4.57.6。配置的 revision 为 main，实际解析提交和 tokenizer 哈希已记录在各 provenance 文件中。

## 方案一

12 句 smoke：384 次生成尝试，302 个有效候选，134 个通过全部过滤；12 句均配对失败。

90 句 pilot：2,880 次尝试，650 次重复或照抄原句，2,230 个有效候选；962 个通过全部过滤（43.14%）。生成 37,386 tokens，生成、特征与质量检查记录耗时合计 723.12 秒，不含全部模型加载和评测开销。

| 划分 | 有效候选 | 合格候选 | 成功配对原句 | 候选不足原句 | 配对条件不满足原句 |
|---|---:|---:|---:|---:|---:|
| dev | 781 | 305 | 1/30 | 25 | 4 |
| test | 713 | 289 | 0/30 | 27 | 3 |
| OOD | 736 | 368 | 0/30 | 20 | 10 |

唯一成功配对原句为 dev-product-013。每组要求 8 个候选，至少需要 16 个合格候选，并满足六个变量的逐对阈值；90 句中 72 句达不到候选数量门槛，17 句达不到配对门槛。

过滤失败次数（同一候选可触发多个条件，不能相加解释为淘汰总数）：

| 划分 | cosine | 双向 NLI | 编辑比例 | 非 ASCII |
|---|---:|---:|---:|---:|
| dev | 370 | 238 | 164 | 7 |
| test | 338 | 200 | 99 | 0 |
| OOD | 286 | 79 | 183 | 1 |

主设置 1 bit、margin=0.0001、20 个密钥种子：test/OOD 四种方法的 coverage_all 均为 0；coverage_matched 不可计算。outlier − matched 均为 0，程序输出探索性 95% CI [0, 0]。这来自全部未配对原句按零覆盖计入，不能解释为两种方法等效或 outlier 无效。其余 bit/margin 设置也没有可比较的 test/OOD 配对子集。

真实读出检查状态为 no_candidates，fresh_text_forwards=0、bits_checked=0、BER=null；不能将 bit_errors=0 解释为读出验证通过。

## 方案二

test 上运行 20 个 trial，每条 1 字节、每句 1 bit、margin=0.0001，四种方法使用相同 trial 消息、密钥、nonce 和原句顺序。

| 方法 | 消息尝试 | 成功编码 | 完整恢复 |
|---|---:|---:|---:|
| outlier | 20 | 0 | 0/20 |
| matched | 20 | 0 | 0/20 |
| random | 20 | 0 | 0/20 |
| nearest | 20 | 0 | 0/20 |

80 次方法级尝试均在第 0 块因 candidate_selection_failed 中止。没有发送成功帧，也没有执行成功帧的独立解码；恢复率为 0%，接受帧 BER 和单位传输 token 净恢复率没有定义。此结果说明当前候选池不能支撑该协议，不是解码错误或 AES-GCM 失败。可选 AES-GCM 真实载体流程本轮未运行，相关核心测试已通过。

## 结论与下一步

本轮得到的是候选预算与严格匹配组合的可行性失败，不能回答“语义合格且匹配充分时 outlier 是否有优势”。下一轮应只在 dev 上检查语义评估误拒情况和候选多样性，并预先选择增加候选预算或调整每组数量等协议变更；先把 dev 配对率提高，再固定协议做新的验证。当前 test/OOD 结果不能用于寻找最有利于 outlier 的阈值。人工语义审查尚未完成。

## 结果文件

- [方案一报告](pilot/report.md)、[主结果与过滤汇总](pilot/execution_summary.json)
- [配对诊断](pilot/selection_diagnostics.jsonl)、[配对差及区间](pilot/paired_deltas.csv)
- [读出验证](pilot/readout_verification.json)、[人工审查候选](pilot/human_audit.csv)
- [消息恢复汇总](pilot_codec_1byte/summary.json)、[逐次消息实验](pilot_codec_1byte/trials.csv)
- Slurm 完整日志：`runs/slurm-1576.log`（项目根目录下）。
- 任务脚本：`scripts/slurm_real_experiment.sh`；额外统计：`scripts/summarize_real_run.py`。
