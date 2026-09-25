# Outlier text pilot

Real-model pilot; human equivalence audit is still required.

Primary setting: 1 bit(s), margin 0.0001.

| Split | Method | Matched / total parents | Coverage (all) | Coverage (matched) | Full coverage (all) |
|---|---|---:|---:|---:|---:|
| test | outlier | 0 / 4 | 0.0000 | NA | 0.0000 |
| test | matched | 0 / 4 | 0.0000 | NA | 0.0000 |
| test | random | 0 / 4 | 0.0000 | NA | 0.0000 |
| test | nearest | 0 / 4 | 0.0000 | NA | 0.0000 |
| ood | outlier | 0 / 4 | 0.0000 | NA | 0.0000 |
| ood | matched | 0 / 4 | 0.0000 | NA | 0.0000 |
| ood | random | 0 / 4 | 0.0000 | NA | 0.0000 |
| ood | nearest | 0 / 4 | 0.0000 | NA | 0.0000 |

Paired outlier minus matched-control differences (source-group × key bootstrap; exploratory 95% intervals):

- test: 0.0000, [0.0000, 0.0000], 4 groups, 4 keys.
- ood: 0.0000, [0.0000, 0.0000], 4 groups, 4 keys.

## Checks and interpretation

- Unmatched parents remain in coverage_all with zero delivery. coverage_matched is explicitly conditional.
- Main proxy is last-layer representation distance, not a demonstrated change of downstream behavior.
- Quality and NLI are independent models, but neither replaces human review. Audit human_audit.csv.
- NLL and mean word frequency are confound proxies. Matching does not establish causality or exact distribution preservation.
- Median thresholds balance dev-reference marginal bits approximately; joint codeword mass need not be balanced.
- pool_proxy_tv_success is conditional finite-pool TV under exp(-total NLL), not global text TV or a security bound.
- Detector AUROC uses held-out parents and held-out evaluation keys. One weak detector cannot certify security.
- These are coding-layer coverage metrics. Full-message, authentication and wire overhead are in codec diagnostics.
- See selection_diagnostics.jsonl for pair counts, exclusion reasons and confound balance; costs.jsonl includes generation and evaluator work.
- Readout times omit plane construction and full feature extraction; do not present them as end-to-end latency.
- Development residuals are in-sample and not evidence. Do not tune on test/ood.

## Selection accounting

- dev: {'caliper_match_failed': 1, 'too_few_eligible': 3}
- test: {'caliper_match_failed': 1, 'too_few_eligible': 3}
- ood: {'caliper_match_failed': 2, 'too_few_eligible': 2}

## Shared preprocessing cost

Generation attempts: 384.
Generated tokens (including rejected candidates): 4811.
Generation + feature + quality seconds: 95.18.
These costs are shared across methods on a common proposal pool; label cold and amortized costs separately.

## Next decision

Inspect exclusions and manually review semantics first. A positive paired difference is a pilot signal only.
Check the OOD split and a second public model with the same frozen protocol before investing in a larger codec.
