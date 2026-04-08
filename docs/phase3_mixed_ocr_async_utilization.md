# Phase 3: Mixed/Full-Chain OCR Async Attribution

## What Changed

- 新增 mixed/full-chain 归因 telemetry：
  - `layout_npu_service_ms`
  - `layout_npu_slot_wait_ms`
  - `ocr_outer_slot_hold_ms`
  - `ocr_submodule_window_ms`
  - `ocr_slot_wait_ms`
  - `ocr_collect_wait_ms`
  - `ocr_inflight_peak`
  - `ocr_buffered_out_of_order_count`
  - `table_npu_service_ms`
  - `table_npu_slot_wait_ms`
  - `table_ocr_service_ms`
  - `table_ocr_slot_wait_ms`
- 新增实验参数：
  - `ocr_outer_mode=immediate_per_task|shadow_windowed_collect`
  - `ocr_shadow_window=<n>`，默认 `8`

## Boundary

- `DXNN-OCR-cpp` 子模块内部已经具备 image-level async queue/callback 能力：
  - `detQueue_ / recQueue_ / outQueue_`
  - detection callback
  - recognition callback
  - `stageExecutor_`
- RapidDocCpp 外层此前仍是 task-level `submitOcrTask -> waitForOcrResult` 短闭环。
- 本阶段实验只改 text OCR canonical task 的外层 collect 边界：
  - `ImmediatePerTask`：保持原行为，slot 包住 submit+wait。
  - `ShadowWindowedCollect`：slot 只包 submit，collect 在 slot 外按原 task 顺序 drain。
- table NPU、table OCR、DXNN-OCR det/rec 内部架构都不改。

## Interpretation

- `ocr_outer_slot_hold_ms` 代表外层真正把共享 NPU slot 占住多久。
- `ocr_submodule_window_ms` 代表 OCR task 从成功 submit 到结果被 collect 的真实子模块窗口。
- 如果 `shadow_windowed_collect` 下：
  - `ocr_outer_slot_hold_ms` 明显下降；
  - 且 `ocr_submodule_window_ms` 仍保持量级；
  - 同时 mixed/full-chain 的 `wall_time_ms` 或 `pages/s` 改善，
  那就说明外层过去确实把 OCR 子模块的 async window 锁死了。

## Guardrails

- serial/reference path 默认不变，hash 口径不变。
- `ocr_ms` / `npu_service_ms` 继续按 OCR/table 的真实 service window 统计，不退化成只记 submit。
- `ocr_inflight_peak` 在文档级聚合时取 page peak 的最大值，不做跨页相加。
