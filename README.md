# RapidDoc C++ (DEEPX NPU Edition)

基于 DEEPX NPU 的文档分析 C++ 实现，源自 [RapidDoc](https://github.com/RapidAI/RapidDoc) Python 项目。

## 特性

- **PDF 渲染**：使用 Poppler 将 PDF 转换为图像
- **版面分析**：检测文档中的标题、段落、图片、表格、公式等 20+ 类别
- **文本识别**：基于 DXNN-OCR-cpp 子模块的 OCR 功能
- **表格识别**：支持有线表格结构识别（UNET 模型）
- **阅读顺序**：XY-Cut++ 算法自动排序页面元素
- **输出格式**：Markdown 和 JSON 结构化输出

## NPU 支持情况

| 功能 | DEEPX NPU | 备注 |
|------|-----------|------|
| Layout 检测 | ✅ | 主模型 NPU + NMS 后处理 ONNX RT |
| OCR 检测/识别 | ✅ | DXNN-OCR-cpp 子模块 |
| 有线表格 | ✅ | UNET 模型 |
| 无线表格 | ❌ | SLANet 不支持 |
| 公式识别 | ❌ | LaTeX-OCR 不支持 |
| 表格分类 | ❌ | 暂不支持 |

## 目录结构

```
RapidDocCpp/
├── include/              # 头文件
│   ├── common/           # 通用类型和配置
│   ├── pdf/              # PDF 渲染
│   ├── layout/           # 版面分析
│   ├── table/            # 表格识别
│   ├── reading_order/    # 阅读顺序
│   ├── output/           # 输出格式化
│   ├── pipeline/         # 主流水线
│   └── server/           # HTTP 服务
├── src/                  # 源文件
├── app/                  # CLI 入口
├── cmake/                # CMake 模块
├── 3rd-party/            # 第三方依赖
│   └── DXNN-OCR-cpp/     # OCR 子模块
├── engine/               # 模型文件
│   └── model_files/      # .dxnn 模型
├── build.sh              # 构建脚本
├── setup.sh              # 依赖安装脚本
└── set_env.sh            # 环境变量
```

## 快速开始

### 1. 环境准备

```bash
# 安装系统依赖
./setup.sh

# 配置 DEEPX 运行时（需要先安装 dx_rt）
source /path/to/dx_rt/set_env.sh

# 配置项目环境
source set_env.sh
```

### 2. 构建

```bash
# Release 构建
./build.sh

# 或 Debug 构建
./build.sh debug
```

### 3. 模型文件

将以下 `.dxnn` 模型放到 `engine/model_files/` 目录：

| 模型 | 文件 | 用途 |
|------|------|------|
| Layout | `layout_det.dxnn` | 版面检测主模型 |
| Layout NMS | `layout_nms.onnx` | NMS 后处理（ONNX） |
| Table UNET | `table_unet.dxnn` | 有线表格检测 |
| OCR | (DXNN-OCR-cpp 模型) | det/rec 模型 |

### 4. 使用

#### CLI 模式

支持参数（与程序实现一致）：

- `-i, --input <path>`：输入 PDF（必填）
- `-o, --output <dir>`：输出目录（默认 `./output`）
- `-d, --dpi <num>`：渲染 DPI（默认 `200`）
- `-m, --max-pages <num>`：最多处理页数（`0` 表示全部）
- `--no-table`：关闭表格识别
- `--no-ocr`：关闭 OCR
- `--json-only`：仅输出 JSON
- `-v, --verbose`：详细日志
- `-h, --help`：帮助信息

```bash
# 基本用法
rapid_doc_cli -i document.pdf -o ./result

# 等价长参数
rapid_doc_cli --input document.pdf --output ./result

# 指定 DPI 和最大页数
rapid_doc_cli -i document.pdf -o ./result -d 300 -m 10

# 仅输出 JSON（不生成 Markdown）
rapid_doc_cli -i document.pdf -o ./result --json-only

# 详细日志
rapid_doc_cli -i document.pdf -o ./result -v
```

#### HTTP 服务

支持参数（与程序实现一致）：

- `-H, --host <addr>`：监听地址（默认 `0.0.0.0`）
- `-p, --port <num>`：端口（默认 `8080`）
- `-w, --workers <num>`：工作线程数（默认 `4`）
- `-h, --help`：帮助信息

```bash
# 启动服务
rapid_doc_server -p 8080

# 自定义 host 和 workers
rapid_doc_server -H 0.0.0.0 -p 8080 -w 4

# 调用 API
curl -X POST http://localhost:8080/process \
    -F "file=@document.pdf"

# Base64 方式
curl -X POST http://localhost:8080/process/base64 \
    -H "Content-Type: application/json" \
    -d '{"data": "<base64_pdf>", "filename": "doc.pdf"}'

# 健康检查
curl http://localhost:8080/health

# 状态查询
curl http://localhost:8080/status
```

#### C++ API

```cpp
#include "pipeline/doc_pipeline.h"

rapid_doc::PipelineConfig config = rapid_doc::PipelineConfig::Default("/path/to/project");
rapid_doc::DocPipeline pipeline(config);

if (!pipeline.initialize()) {
    // handle error
}

auto result = pipeline.processPdf("document.pdf");
std::cout << result.markdown << std::endl;
```

## 致谢

- [RapidDoc](https://github.com/RapidAI/RapidDoc) - 原始 Python 实现
- [DXNN-OCR-cpp](https://github.com/DEEPX-AI/DXNN-OCR-cpp) - OCR 子模块
- [DEEPX](https://deepx.ai) - NPU 硬件和运行时
