# 2026-09-25 真实模型运行结果

请先阅读 [中文实验报告](RESULTS_ZH.md)。Slurm 作业 1576 已完成，但 test/OOD 均无成功配对原句，因此无法检验 outlier 优势；消息实验全部在发送前中止，读出检查没有可验证候选。

此目录保存 smoke 与 90 句 pilot 的报告、指标、过滤与配对诊断、全部生成尝试、汇总特征和向量、公开读出 profile、模型版本记录，以及 20 次一字节消息实验的四方法结果。运行日志位于 [logs/](logs/)。

未上传实验私钥、随机消息文件、模型权重及可由汇总特征替代的 causal/quality 中间缓存。汇总向量和原始指标均保留，可用于复查分析；此结果目录不作为直接续跑生成任务的缓存目录。

项目根目录下的 `scripts/slurm_real_experiment.sh` 为本次任务脚本，`scripts/summarize_real_run.py` 用于生成 `execution_summary.json`。报告中 `runs/` 路径是原始运行位置，上传后的对应结果位于本目录。
