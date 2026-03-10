# RapidDoc C++ (DEEPX NPU Edition)

基于 DEEPX NPU 的文档分析 C++ 实现

## 特性

- **PDF 渲染**：Poppler 转图像
- **版面分析**：`pp_doclayout_l_part1.dxnn` + `pp_doclayout_l_part2.onnx`
- **OCR**：DXNN-OCR-cpp 子模块（检测 + 识别）
- **有线表格**：UNET 结构识别 + 单元格 OCR → HTML 表格
- **阅读顺序**：XY-Cut++ 排序
- **输出**：Markdown + JSON 内容列表
- **服务接口**：CLI、HTTP Server、Gradio Demo

## NPU 支持

| 功能 | 支持 | 说明 |
|------|------|------|
| Layout | ✅ | DXNN 主模型 + ONNX NMS 后处理 |
| OCR | ✅ | 检测/识别均在 NPU |
| 有线表格 | ✅ | UNET 结构 + 单元格 OCR |
| 无线表格 | ❌ | 不支持，跳过 |
| 公式 | ⚠️ | 不做公式识别，按图片区域导出 |
| 表格分类 | ❌ | 不支持 |

## 安装与运行

### 1. 环境

- 编译器（支持 C++17）、CMake、OpenCV、ONNX Runtime
- **首次拉取建议先初始化子模块**：
  ```bash
  git submodule update --init --recursive
  ```
- 默认会使用项目内子模块源码构建 OpenCV；如需改用系统 OpenCV，可在构建前设置：
  ```bash
  BUILD_OPENCV_FROM_SOURCE=OFF ./build.sh
  ```
- **DEEPX 运行时**：安装 `dx_rt` 后执行 `source /path/to/dx_rt/set_env.sh`（NPU 推理必需）
- 可选：项目根目录 `set_env.sh` 用于配置 `CUSTOM_INTER_OP_THREADS_COUNT`、`DXRT_TASK_MAX_LOAD`、`NFH_*` 等环境变量
- **ONNX Runtime 说明**：版面分析后处理依赖 ONNX Runtime。若未找到 ONNX Runtime，CMake 可继续配置，但 layout 后处理会被跳过，最终版面框结果会为空

### 2. 下载模型

从 `sdk.deepx.ai` 拉取 Layout、Table、OCR 模型并解压到对应目录：

```bash
./setup.sh
```

可选 `./setup.sh --force` 强制重新下载。完成后关键模型位于：

- `engine/model_files/layout/pp_doclayout_l_part1.dxnn`
- `engine/model_files/layout/pp_doclayout_l_part2.onnx`
- `engine/model_files/table/unet.dxnn`
- `3rd-party/DXNN-OCR-cpp/engine/model_files/server/det_v5_640.dxnn`
- `3rd-party/DXNN-OCR-cpp/engine/model_files/server/rec_v5_ratio_*.dxnn`

其中 OCR 检测模型会按当前代码要求重命名到 `det_v5_640.dxnn`。

### 3. 构建

```bash
./build.sh              # Release，产物在 build_Release/bin/
./build.sh debug        # Debug，产物在 build_Debug/bin/
./build.sh clean        # 清空后重新 configure
./build.sh test         # 编译测试并自动运行 rapiddoc_tests / rapiddoc_cross_tests
```

如需使用系统 OpenCV：

```bash
BUILD_OPENCV_FROM_SOURCE=OFF ./build.sh
```

### 4. 使用

**CLI**（可执行文件在 `build_Release/bin/rapid_doc_cli`，可将该目录加入 PATH）：

```bash
# 帮助
./build_Release/bin/rapid_doc_cli -h

# 必填：-i 输入 PDF，-o 输出目录
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out

# 可选：DPI、最大页数、关闭表格/OCR、仅 JSON、详细日志
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out -d 300 -m 10
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out --no-table
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out --no-ocr
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out --json-only
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out -v
```

输出文件（在 `-o` 目录下）：

- `{文件名}.md`
- `{文件名}_content.json`
- `images/page{n}_fig{k}.png`
- `images/page{n}_eq{k}.png`

**HTTP 服务**（`build_Release/bin/rapid_doc_server`）：

```bash
# 启动
./build_Release/bin/rapid_doc_server -H 0.0.0.0 -p 8080 -w 4

# 帮助
./build_Release/bin/rapid_doc_server -h

# 健康检查 / 状态
curl http://localhost:8080/health
curl http://localhost:8080/status

# 批量解析接口（当前更完整）
curl -X POST http://localhost:8080/file_parse \
  -F "files=@doc.pdf" \
  -F "return_md=true" \
  -F "return_content_list=true"
```

当前服务接口包括：

- `POST /process`
- `POST /process/base64`
- `POST /file_parse`
- `POST /v1/images:annotate`
- `GET /health`
- `GET /status`

其中 `file_parse` 支持 PDF 和图片输入；`v1/images:annotate` 支持 `TEXT_DETECTION` / `DOCUMENT_TEXT_DETECTION` 风格请求；远程 `http://` / `https://` 图片 URL 在当前实现里会被拒绝。

**Gradio UI 可视化 Demo**（`demo/gradio_app.py`）：

```bash
# 安装依赖
pip install -r demo/requirements-gradio.txt

# 先启动 C++ Server
./build_Release/bin/rapid_doc_server -H 0.0.0.0 -p 8080 -w 4

# 再启动 Gradio UI
RAPIDDOC_CPP_SERVER_URL=http://127.0.0.1:8080 python3 demo/gradio_app.py

# 或显式指定参数
python3 demo/gradio_app.py --server-url http://127.0.0.1:8080 --host 0.0.0.0 --port 7860
```

默认访问地址为 `http://127.0.0.1:7860`。该 Demo 本身不直接解析文档，而是把上传的 PDF / 图片请求转发到 C++ HTTP 服务，再展示：

- Markdown 预览
- Markdown 原文
- 提取图片
- Layout 可视化文件
- 简单性能统计

## 目录结构

```
RapidDocCpp/
├── include/          # 头文件（common, pdf, layout, table, reading_order, output, pipeline, server）
├── src/              # 实现
├── app/              # CLI 入口
├── demo/             # Gradio Demo（请求转发到 C++ Server）
├── engine/           # 模型目录（由 setup.sh 填充）
├── test/             # 单元测试、交叉验证、E2E 测试
├── tools/            # 脚本：compare_e2e.py、render_table_html.py、compare_table_structure.py 等
├── build.sh          # 构建（支持 release|debug、clean、test）
├── setup.sh          # 模型下载
└── set_env.sh        # 可选环境变量
```

## 开发与测试

- **单元测试 + 交叉验证**：`./build.sh test`。会打开 `BUILD_TESTS`、编译并自动执行 `rapiddoc_tests`、`rapiddoc_cross_tests`
- **E2E 测试**：`rapiddoc_e2e_tests` 会被构建，但不会被 `./build.sh test` 自动执行，可手动运行：
  ```bash
  ./build_Release/test/rapiddoc_e2e_tests --gtest_color=yes
  ```
- **CTest**：可在构建目录执行：
  ```bash
  cd build_Release
  ctest --output-on-failure
  ```
- **E2E 跑批**：对指定目录下全部 PDF 跑完整流水线，结果写入 `test/fixtures/e2e/cpp/`。需先执行一次 `./build.sh test` 以编译 `run_cpp_e2e`。项目根目录由 CMake 在编译时写入，无需手传；默认 PDF 目录为 `test_files`
  ```bash
  ./build_Release/test/run_cpp_e2e
  ./build_Release/test/run_cpp_e2e --pdf-dir /path/to/pdfs --output /path/to/out
  ```
- **注意**：`rapiddoc_cross_tests` 中包含 live inference 测试。这些测试依赖 DEEPX 硬件、模型文件和完整 DXRT 执行环境；缺少输入时可能 `SKIP`，但环境不完整时也可能直接失败

## 致谢

- [RapidDoc](https://github.com/RapidAI/RapidDoc) - Python 实现
- [DXNN-OCR-cpp](https://github.com/DEEPX-AI/DXNN-OCR-cpp) - OCR 子模块
- [DEEPX](https://deepx.ai) - NPU 与运行时
