# 第二轮：dev 缓存诊断

完整结果见 [RESULTS_ZH.md](RESULTS_ZH.md)。作业 1577 完成，17 项测试通过；仅使用 dev，未新增生成预算或消息实验。

- 读出：305 条合格候选重新推理，6,100 个单比特比较，错误 0。
- 每组 2 条候选：13/30 个原句成功共同配对。
- 完整合格池：56.83% 原句×密钥组合同时覆盖 0 和 1；random 两候选为 23.50%。
- 共同子集：outlier − matched 为 +1.15 个百分点，探索性 95% CI [−3.46，+6.35]；暂无稳定额外优势。
- [human_audit_blind.csv](human_audit_blind.csv) 为待填写人工表；[ai_audit_preliminary.csv](ai_audit_preliminary.csv) 仅为 AI 初审，不是人工标签。

原第一轮归档已将没有可比较原句时的方法效应与区间更正为 NA；协议全体原句交付率仍为零。未修改原始 runs 目录中的历史产物。
