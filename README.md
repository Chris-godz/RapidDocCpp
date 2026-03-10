# RapidDoc C++ (DEEPX NPU Edition)

基于 DEEPX NPU 的文档分析 C++ 实现
## 特性

- **PDF 渲染**：Poppler 转图像
- **版面分析**：20+ 类别（标题、段落、图、表、公式等）
- **OCR**：DXNN-OCR-cpp 子模块（检测 + 识别）
- **有线表格**：UNET 结构识别 → HTML 表格
- **阅读顺序**：XY-Cut++ 排序
- **输出**：Markdown + JSON 内容列表

## NPU 支持

| 功能 | 支持 | 说明 |
|------|------|------|
| Layout | ✅ | DXNN 主模型 + ONNX NMS 后处理 |
| OCR | ✅ | 检测/识别均在 NPU |
| 有线表格 | ✅ | UNET 结构 + 单元格 OCR |
| 无线表格 | ❌ | 不支持，跳过 |
| 公式 | ❌ | 不支持，跳过 |
| 表格分类 | ❌ | 不支持 |

## 安装与运行

### 1. 环境

- 编译器（支持 C++17）、CMake、OpenCV、ONNX Runtime、Poppler（或使用项目内子模块）
- **DEEPX 运行时**：安装 dx_rt 后 `source /path/to/dx_rt/set_env.sh`（NPU 推理必需）
- 可选：项目根目录 `set_env.sh` 用于配置 DXRT/NFH 等高级环境变量

### 2. 下载模型

从 sdk.deepx.ai 拉取 Layout、Table、OCR 模型并解压到对应目录：

```bash
./setup.sh
```

可选 `./setup.sh --force` 强制重新下载。完成后模型在 `engine/model_files/` 与 `3rd-party/DXNN-OCR-cpp/engine/model_files/server/`。

### 3. 构建

```bash
./build.sh              # Release，产物在 build_Release/bin/
./build.sh debug        # Debug，产物在 build_Debug/bin/
./build.sh clean        # 清空后重新 configure
```

### 4. 使用

**CLI**（可执行文件在 `build_Release/bin/rapid_doc_cli`，可将该目录加入 PATH）：

```bash
# 必填：-i 输入 PDF，-o 输出目录
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out

# 可选：DPI、最大页数、关闭表格/OCR、仅 JSON、详细日志
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out -d 300 -m 10
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out --no-table
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out --json-only
./build_Release/bin/rapid_doc_cli -i document.pdf -o ./out -v
```

输出文件（在 `-o` 目录下）：`{文件名}.md`、`{文件名}_content.json`。

**HTTP 服务**（`build_Release/bin/rapid_doc_server`）：

```bash
./build_Release/bin/rapid_doc_server -H 0.0.0.0 -p 8080 -w 4
# 上传 PDF：curl -X POST http://localhost:8080/process -F "file=@doc.pdf"
# 健康检查：curl http://localhost:8080/health
```

## 目录结构

```
RapidDocCpp/
├── include/          # 头文件（common, pdf, layout, table, reading_order, output, pipeline, server）
├── src/              # 实现
├── app/              # CLI 入口
├── engine/           # 模型目录（由 setup.sh 填充）
├── tools/            # 脚本：compare_e2e.py、render_table_html.py、compare_table_structure.py 等
├── build.sh          # 构建（支持 release|debug、clean、test）
├── setup.sh          # 模型下载
└── set_env.sh        # 可选环境变量
```

## 开发与测试

- **单元测试 + 交叉验证**：`./build.sh test`。会打开 BUILD_TESTS、编译并自动执行 `rapiddoc_tests`、`rapiddoc_cross_tests`。
- **E2E 跑批**：对指定目录下全部 PDF 跑完整流水线，结果写入 `test/fixtures/e2e/cpp/`，便于与 Python 版对比。需先执行一次 `./build.sh test` 以编译 `run_cpp_e2e`。项目根目录由 CMake 在编译时写入，无需手传；默认 PDF 目录为 `test_files`。
  ```bash
  ./build_Release/test/run_cpp_e2e
  ./build_Release/test/run_cpp_e2e --pdf-dir /path/to/pdfs --output /path/to/out
  ```

## 致谢

- [RapidDoc](https://github.com/RapidAI/RapidDoc) - Python 实现
- [DXNN-OCR-cpp](https://github.com/DEEPX-AI/DXNN-OCR-cpp) - OCR 子模块
- [DEEPX](https://deepx.ai) - NPU 与运行时
