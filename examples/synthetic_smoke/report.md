# Outlier text pilot

**SYNTHETIC SMOKE TEST — NOT LLM EXPERIMENTAL EVIDENCE.**

Primary setting: 1 bit(s), margin 0.0001.

| Split | Method | Matched / total parents | Coverage (all) | Coverage (matched) | Full coverage (all) |
|---|---|---:|---:|---:|---:|
| test | outlier | 12 / 12 | 0.9583 | 0.9583333333333334 | 0.9167 |
| test | matched | 12 / 12 | 0.9722 | 0.9722222222222222 | 0.9444 |
| test | random | 12 / 12 | 0.9722 | 0.9722222222222222 | 0.9444 |
| test | nearest | 12 / 12 | 0.9722 | 0.9722222222222222 | 0.9444 |
| ood | outlier | 12 / 12 | 0.9514 | 0.9513888888888888 | 0.9028 |
| ood | matched | 12 / 12 | 0.9583 | 0.9583333333333334 | 0.9167 |
| ood | random | 12 / 12 | 0.9792 | 0.9791666666666666 | 0.9583 |
| ood | nearest | 12 / 12 | 0.9722 | 0.9722222222222222 | 0.9444 |

Paired outlier minus matched-control differences (source-group × key bootstrap; exploratory 95% intervals):

- test: -0.0139, [-0.0903, 0.0420], 12 groups, 6 keys.
- ood: -0.0069, [-0.0764, 0.0625], 12 groups, 6 keys.

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

- dev: {'ok': 12}
- test: {'ok': 12}
- ood: {'ok': 12}

## Shared preprocessing cost

Generation attempts: 1152.
Generated tokens (including rejected candidates): 0.
Generation + feature + quality seconds: 0.00.
These costs are shared across methods on a common proposal pool; label cold and amortized costs separately.

## Next decision

Inspect exclusions and manually review semantics first. A positive paired difference is a pilot signal only.
Check the OOD split and a second public model with the same frozen protocol before investing in a larger codec.
