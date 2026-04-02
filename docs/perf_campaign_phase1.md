# RapidDocCpp Phase 1 性能总攻（核心流水线）

## 1. 背景与范围
- 工作分支：`perf-phase1-core-hotpath`
- 基线分支：`benchmark-baseline-20260330-ingress`
- 正式基线 benchmark：多卡 `27/27 completed`（已可用）
- 当前主瓶颈：`npu_serial_ms`，不是 `npu_lock_wait_ms`
- 核心目标：只打 `src/pipeline/doc_pipeline.cpp` 的 `DocPipeline::processPage()` 热路径及其直接上下游，不扩展 topology 议题

## 2. 本阶段明确不做
- 不继续优化 A/B topology 选择
- 不做大规模 server/LB 架构重写
- 不修改 `3rd-party/DXNN-OCR-cpp` 子模块内部
- 不通过减少输出内容（markdown/content_list/middle_json/images）来“伪提速”

## 3. Baseline 构建阻塞修复（最小提交）
- 提交：`d5c5ad2ecd2b6835b354a84625f4cf310f9ab588`
- commit message：`fix: add missing detail_report header`
- 仅改动文件：`include/output/detail_report.h`
- 约束确认：只补声明，不改逻辑、不改算法、不改 telemetry 行为

## 4. Phase 1 必做 5 项优化（按优先级）

### 4.1 P0：清理 OCR/table 重复与无效调用
- 原因：直接命中 `npu_serial_ms` 主瓶颈，ROI 级别重复调用会线性放大串行 NPU 时间。
- 涉及文件：
  - `src/pipeline/doc_pipeline.cpp`
  - `include/pipeline/doc_pipeline.h`
  - 测试：`test/test_phase1_integration.cpp`、`test/test_phase1_contracts.cpp`
- 函数级落点：`DocPipeline::processPage()`，在 layout bucket 后、OCR/table 工作项进入 NPU 前去重。
- 风险：误去重导致合法内容丢失。
- 回归点：`ContentElement` 顺序稳定；`md_content`、`content_list`、`middle_json` 不漂移。
- 验收指标：重复框不再重复 submit；`npu_serial_ms` 不回升。

### 4.2 P1：PDF render 与 page CPU 阶段 overlap（低风险）
- 原因：原链路先全量 render 再 page 处理，render 与 CPU pipeline 无重叠。
- 涉及文件：
  - `src/pdf/pdf_renderer.cpp`
  - `include/pdf/pdf_renderer.h`
  - `src/pipeline/doc_pipeline.cpp`
  - `include/pipeline/doc_pipeline.h`
- 函数级落点：
  - `PdfRenderer` 新增 streaming 入口（additive）
  - `DocPipeline::processPdfFromMemoryInternal()` 引入 bounded producer-consumer
- 风险：队列无界导致内存膨胀，页序错乱。
- 回归点：`result.pages` 必须按 `pageIndex` 输出；`npuSerialMutex()` 保护不变。
- 验收指标：`pipeline_call_ms` 非回退；`pdf_render_ms` 或总耗时下降；内存受上限约束。

### 4.3 P2：削减 clone/copy/中间对象物化
- 原因：深拷贝抬高 `cpu_only_ms` 与内存抖动。
- 涉及文件：
  - `src/pdf/pdf_renderer.cpp`
  - `src/pipeline/doc_pipeline.cpp`
  - `src/server/server.cpp`
- 函数级落点：
  - `PdfRenderer::renderFromMemory()` 去无争议 `clone`
  - `DocPipeline::processImageDocumentInternal()`、`DocPipeline::processImage()` 去整页冗余拷贝
  - `processDocumentBytes()` 避免 image decode 的二次 buffer 复制
- 风险：`cv::Mat` 生命周期错误引发脏图/随机崩溃。
- 回归点：HTTP 集成稳定；输出 hash 稳定。
- 验收指标：`cpu_only_ms`、`prepare_ms`、`pipeline_call_ms` 至少一项可见下降。

### 4.4 P3：补齐 stage telemetry 完整性
- 原因：没有统一口径就无法稳定归因。
- 涉及文件：
  - `include/common/types.h`
  - `include/common/perf_utils.h`
  - `src/common/perf_utils.cpp`
  - `src/server/server.cpp`
  - `tools/rapid_doc_bench.cpp`
  - `report/run_multinpu_bench.py`
- 函数级落点：
  - `makeStatsJson()`、`buildStatusJson()` 对齐字段
  - `stageStatsToJson()`、`meanStageStats()` 对齐字段
- 风险：统计口径重复计时或 key 漂移。
- 回归点：旧 key 不删；新增 key 全为 additive。
- 验收指标：HTTP 与 benchmark 的同名字段同义且可比。

### 4.5 P4：扩充真实 perf baseline
- 原因：仅总时长无法解释收益来源。
- 涉及文件：
  - `tools/rapid_doc_bench.cpp`
  - `report/run_multinpu_bench.py`
  - `output-benchmark/*`、`report/*` 产物规范
- 风险：case 配置漂移导致“数字不可比”。
- 回归点：workload/page range/output 配置固定，不改测试口径。
- 验收指标：before/after 同 workload、同并发、同配置，可直接按阶段归因。

## 5. 给 Cursor 的逐提交执行顺序
1. `perf: align pipeline telemetry across http and benchmark`
2. `perf: dedupe exact duplicate ocr and table work items`
3. `perf: remove redundant page image and decode copies`
4. `perf: stream pdf pages through bounded render queue`
5. `perf: publish phase1 core hotpath regression summary`

## 6. 给 Copilot 的 Review Checklist
1. 是否触碰 `3rd-party/DXNN-OCR-cpp`（若触碰直接退回）。
2. 是否改动 server/LB/topology 语义（本阶段禁止）。
3. `DocPipeline::processPage()` 去重是否仅覆盖低风险重复（严格避免大范围吞框）。
4. telemetry 是否 additive（旧 key 不删不改义）。
5. 去 copy 后 `cv::Mat` 生命周期是否安全。
6. streaming 队列是否 bounded，`result.pages` 是否按 `pageIndex` 回填。
7. benchmark 脚本是否同步，before/after 是否同口径。
8. hash 是否稳定（warmup md/middle_json）。

## 7. 正式 before/after 结果

### 7.1 数据路径
- before in-process：`/home/deepx/Desktop/RapidDocCpp_phase1_baseline/output-benchmark/phase1_final_inprocess_before.json`
- before HTTP：`/home/deepx/Desktop/RapidDocCpp_phase1_baseline/report/phase1_final_http_before.json`
- after in-process：`/home/deepx/Desktop/RapidDocCpp/output-benchmark/phase1_final_inprocess_after.json`
- after HTTP：`/home/deepx/Desktop/RapidDocCpp/report/phase1_final_http_after.json`

### 7.2 简表（workload: layout_ocr / layout_ocr_table）
说明：`total_time_ms` 取 in-process `document_total.mean_ms`；`pipeline_call_ms`、`npu_serial_ms`、`cpu_only_ms` 取 HTTP `c2` 档位（同口径对比）。

| workload | total_time_ms before | total_time_ms after | pipeline_call_ms before | pipeline_call_ms after | npu_serial_ms before | npu_serial_ms after | cpu_only_ms before | cpu_only_ms after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| layout_ocr | 19134.720 | 15850.255 | 20904.492 | 17565.999 | 15320.102 | 15421.340 | 1672.850 | 1673.604 |
| layout_ocr_table | 6461.912 | 6465.634 | 6575.926 | 6537.074 | 6266.754 | 6256.429 | 202.382 | 198.538 |

### 7.3 输出 hash / 一致性检查
- HTTP warmup（`layout_ocr_c2`）：
  - `warmup_md_sha256` before==after：`0934b8bdd4e0943134617ae4bc3770a07edee1297c1dfc793c14ea578f97da42`
  - `warmup_middle_json_sha256` before==after：`acacceb2f20b612d32b4192f38d7703f1882bd13efa68d537fe36d66d554d67f`
- HTTP warmup（`layout_ocr_table_c2`）：
  - `warmup_md_sha256` before==after：`a43080bb0ffeff70f0910f274852d2f8c0ef8d80b7be21aea845412d5dae895b`
  - `warmup_middle_json_sha256` before==after：`e6a5a99637066f2c0ab851048fc59724316e9873b92e3821e1f35ee019246805`
- after in-process 输出 hash：
  - `ocr_only`：`markdown_sha256=0934...da42`，`content_list_sha256=f1fb...96e8`
  - `layout_ocr_table_full_chain`：`markdown_sha256=a430...895b`，`content_list_sha256=6493...52cf`

## 8. 剩余 blocker
- 无代码级 blocker。
- 基线分支历史 `rapid_doc_bench` 不输出 `pipeline_call` 与 in-process 内容 hash；本轮通过 HTTP 同口径字段与 warmup hash 完成可比性闭环。

## 9. 第一枪（结论）
第一枪仍然是 `src/pipeline/doc_pipeline.cpp` 的 `DocPipeline::processPage()`：先清理重复/无效 OCR 与 table 调用。

原因：
- 它最贴近 `npu_serial_ms` 主瓶颈。
- 改动面小、回滚成本低。
- 在输出 hash 稳定前提下最容易形成“可证明收益”的首个增量。
