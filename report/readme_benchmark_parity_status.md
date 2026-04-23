# README Benchmark And Same-Module Parity Status

Source JSON: `/home/deepx/Desktop/RapidDocCpp/output-benchmark/readme_finegrained_mandatory_20260414_110640/cpp_readme_suite_with_python_baseline.json`

## Summary

- Python README baseline: `ok`; pages=66; wall=263.370s; pages/s=0.300; Formula included=true.
- C++ README input-set counterpart: `ok`; pages=66; wall=43.289s; pages/s=1.525.
- Complete same-module parity: `not_proven`; blocker=`formula_parity`.
- Formula-disabled README-adjacent lane: `not_established`.

## README Baseline

- Actual command: `python demo/demo_offline.py --finegrained`.
- Mandatory README command: `python demo/demo_offline.py --finegrained`.
- Required env: `source deepx_scripts/set_env.sh 1 2 1 3 2 4`.
- Markdown hash: `e7797d86e411224e44d5fc6124a17f9310c274bcc98bc8d7f7d8d175f6cb2a63`.
- Content-list hash: `a1dd1b6021d9a4e6f86576fc18e9140255a4ad24cda346edb640918008c8a6ec`.

## C++ Counterpart

- Label: `README input-set counterpart / 当前 C++ 能力边界对照`.
- Overlap factor: `1.738`.
- Formula timing: `0.076 ms` (placeholder-level; not Formula parity).
- Hash mismatch: `false`.
- Markdown hash: `00aba3516ea34e9d95539f10c941b5e6fbbb0657307ac42e4eb28f37f5a05c69`.
- Content-list hash: `3facfe2d4e9277b6534f257f8e129dde6078a4ae119c87656c6171f8c4d24614`.

## Parity Matrix

| Capability | Status |
| --- | --- |
| Python README baseline | `ok` |
| C++ README input-set counterpart | `ok` |
| Routing / ownership / ordered output / hash stability | `telemetry_supported` |
| Formula parity | `not_proven` |
| Formula-disabled README-adjacent lane | `not_established` |

## Allowed Conclusion

- C++ is faster on the README input set under the current non-complete Formula capability boundary.
- Do not report full same-module parity as proven until Formula is implemented in C++ and the Formula-enabled README suite is rerun.
- Do not describe the remaining blocker as missing Python benchmark data; the blocker is Formula parity.
