# RapidDocCpp 项目开发规范（Copilot）

## 目标

Copilot 与 Skills（`.cursor/rules/*.mdc`）使用同一套规范，保持强一致。

## 规则对齐映射

- 核心总则：`.cursor/rules/00-core.mdc`
- 脚本职责：`.cursor/rules/10-scripts.mdc`
- CMake 规范：`.cursor/rules/20-cmake.mdc`
- 参数解析：`.cursor/rules/30-argparse.mdc`
- DEEPX NPU：`.cursor/rules/40-deepx-npu.mdc`

## 核心原则：功能解耦，单一职责

每个文件 / 脚本 / 模块只做一件事，职责边界清晰，不混合。

## 脚本职责划分

| 脚本 | 职责 | 禁止 |
|------|------|------|
| `build.sh` | 仅负责编译（clean/debug/release/test） | 不装依赖、不配环境、不下模型 |
| `setup.sh` | 仅负责下载模型、初始化子模块 | 不 apt-get、不配环境变量、不编译 |
| `set_env.sh` | 仅负责配置运行时环境变量 | 不编译、不下载、不安装 |

## 编译系统规范

- CMakeLists.txt 中用 `option()` 设好默认值，build.sh 不重复指定已有默认值的选项
- 每个模块独立一个 `CMakeLists.txt`，根 CMake 只做 `add_subdirectory`
- build.sh 参数风格参考 ocr_demo：`./build.sh [release|debug] [clean] [test]`

## 参数解析规范

- C++ 的 CLI/Server 参数解析统一使用 `getopt_long`（`<getopt.h>`）
- 不再手写参数循环逻辑，避免重复造轮子
- shell 脚本参数解析保持 `case` 方式

## 代码组织

- 头文件放 `include/<module>/`，实现放 `src/<module>/`
- 每个模块（common, pdf, layout, table, reading_order, output, pipeline, server）各自独立
- 第三方依赖统一放 `3rd-party/`，通过 git submodule 管理
- 模型文件放 `engine/model_files/`，不纳入版本管理

## DEEPX NPU 约束

- 使用 dxrt::InferenceEngine API 进行模型推理
- Layout 使用双引擎：DX NPU（主模型）+ ONNX Runtime（NMS 后处理）
- 不支持功能（公式、无线表格、表格分类）必须显式 `skip`，不要留空实现
- 环境变量通过 `set_env.sh` 管理，不硬编码

## 风格偏好

- 脚本保持精简，不加过多装饰性输出
- 不要在脚本中混入不属于其职责的功能
- 新增功能时先确认应该放在哪个脚本/模块中
