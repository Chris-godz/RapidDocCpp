# Phase2 OCR Profile Shadow Analysis Report

- 生成时间：`2026-04-07 15:46:12`
- 提交类型：`analysis-only`（不改推理逻辑）
- 主分析输入(in-process)：`output-benchmark/phase2_profile_after_inprocess.json`
- 主分析输入(HTTP)：`report/phase2_profile_http_single.json`
- 回归验证输入(in-process)：`output-benchmark/phase2_shadow_analysis_verify_inprocess.json`
- 回归验证输入(HTTP)：`report/phase2_shadow_analysis_verify_http.json`

## 1. 固定派生指标定义

- `npu_ms_per_ocr_submit = npu_serial_ms / ocr_submit_count`
- `small_ratio = ocr_submit_small_count / ocr_submit_count`
- `large_ratio = ocr_submit_large_count / ocr_submit_count`
- `category_ratio_text/title/code/list = 对应 count / ocr_submit_count`
- `uncategorized_submit = ocr_submit_count - (text + title + code + list)`（下限 0）

## 2. Workload 画像与判定

| workload | ocr_submit_count | npu_serial_ms | npu_ms_per_ocr_submit | small_ratio | large_ratio | p95/p50 | category_ratio text/title/code/list | uncategorized_submit | auto_verdict | manual_override | override_reason | final_verdict |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|
| ocr_only | 57 | 16173.221 | 283.741 | 0.2807 | 0.6491 | 3.2166 | 0.9825/0.0175/0.0000/0.0000 | 0 | No | No | 非边界 case；保持 auto_verdict。 | No |
| layout_ocr_table | 30 | 6430.901 | 214.363 | 0.7667 | 0.1667 | 40.9079 | 0.6667/0.2667/0.0000/0.0000 | 2 | Yes | No | 非边界 case；保持 auto_verdict。 | Yes |

自动规则：`Yes` 当且仅当 `small_ratio>=0.60 && large_ratio>=0.10 && p95/p50>=8`；否则 `No`。
边界区间：`small_ratio∈[0.55,0.65]` 或 `large_ratio∈[0.08,0.12]` 或 `p95/p50∈[7,9]`，可人工 override 并必须写理由。

## 3. 专项解释

- `ocr_only` 是大框主导型 workload。依据：large_ratio=0.6491，small_ratio=0.2807，p95/p50=3.2166。
- `layout_ocr_table` 是碎片化 + 大框尾部混合型 workload。依据：small_ratio=0.7667，large_ratio=0.1667，p95/p50=40.9079。

## 4. Analysis-only 回归验证

### 4.1 in-process hash 一致性

| workload | main markdown_sha256 | verify markdown_sha256 | markdown same | main content_list_sha256 | verify content_list_sha256 | content_list same |
|---|---|---|---|---|---|---|
| ocr_only | 0934b8bdd4e0943134617ae4bc3770a07edee1297c1dfc793c14ea578f97da42 | 0934b8bdd4e0943134617ae4bc3770a07edee1297c1dfc793c14ea578f97da42 | Yes | f1fb09ef16369ff78dcab106dda6e3d1b97d178535ac37ce52541546b0ab96e8 | f1fb09ef16369ff78dcab106dda6e3d1b97d178535ac37ce52541546b0ab96e8 | Yes |
| layout_ocr_table | a43080bb0ffeff70f0910f274852d2f8c0ef8d80b7be21aea845412d5dae895b | a43080bb0ffeff70f0910f274852d2f8c0ef8d80b7be21aea845412d5dae895b | Yes | 6493c73eb938f5e086a793d64aebb69fa52c6892c3305569f106f1f5004452cf | 6493c73eb938f5e086a793d64aebb69fa52c6892c3305569f106f1f5004452cf | Yes |

### 4.2 HTTP warmup hash 一致性（layout_ocr）

- `main md_sha256`: `0934b8bdd4e0943134617ae4bc3770a07edee1297c1dfc793c14ea578f97da42`
- `verify warmup md_sha256`: `0934b8bdd4e0943134617ae4bc3770a07edee1297c1dfc793c14ea578f97da42`
- `md same`: `Yes`
- `main middle_json_sha256`: `acacceb2f20b612d32b4192f38d7703f1882bd13efa68d537fe36d66d554d67f`
- `verify warmup middle_json_sha256`: `acacceb2f20b612d32b4192f38d7703f1882bd13efa68d537fe36d66d554d67f`
- `middle_json same`: `Yes`

### 4.3 指标展示（仅展示，不作性能门禁）

| scope | workload | total_time_ms(main/verify) | pipeline_call_ms(main/verify) | npu_serial_ms(main/verify) | cpu_only_ms(main/verify) |
|---|---|---:|---:|---:|---:|
| in-process | ocr_only | 16721.665/15946.504 | 16721.673/15946.512 | 16173.221/15445.382 | 45.410/39.007 |
| in-process | layout_ocr_table | 6655.332/6472.491 | 6655.342/6472.499 | 6430.901/6268.077 | 132.601/118.679 |
| HTTP | layout_ocr | 18458.831/17668.121 | 18458.840/17668.121 | 16103.236/15459.234 | 1863.806/1722.158 |

## 5. 下一枪建议（Yes/No）

- `ocr_only`: `No`
- `layout_ocr_table`: `Yes`

结论：下一枪建议：进入 shadow packing/merge 候选试验，范围限定为 `layout_ocr_table`（`ocr_only` 判定为 No）。
