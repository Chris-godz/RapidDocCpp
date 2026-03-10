#!/usr/bin/env python3
"""
RapidDocCpp Gradio UI.

The UI layout follows the Python RapidDoc Gradio demo, but all parsing
requests are forwarded to the C++ Crow server.
"""

from __future__ import annotations

import argparse
import contextlib
import mimetypes
import os
import re
import socket
from pathlib import Path
from typing import Iterable, List, Tuple

os.environ.setdefault("GRADIO_DEFAULT_LANGUAGE", "en")
os.environ.setdefault("LANG", "en_US.UTF-8")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")

import gradio as gr
import requests
from starlette.middleware.base import BaseHTTPMiddleware


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo" / "output-gradio"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SERVER_URL = os.environ.get("RAPIDDOC_CPP_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")


def detect_lan_ip() -> str:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
        try:
            # No payload is sent; this lets the kernel pick the outbound interface.
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        except OSError:
            pass
    return "127.0.0.1"


def convert_md_images_for_gradio(md_content: str, image_dir: Path) -> str:
    if not md_content:
        return ""

    def replace_image(match: re.Match[str]) -> str:
        relative_path = match.group(1).strip()
        full_path = image_dir.parent / relative_path
        if not full_path.exists():
            full_path = image_dir / Path(relative_path).name
        if not full_path.exists():
            return match.group(0)

        mime_type = mimetypes.guess_type(full_path.name)[0] or "image/png"
        data = full_path.read_bytes()
        import base64

        encoded = base64.b64encode(data).decode("ascii")
        return f'<img src="data:{mime_type};base64,{encoded}" style="max-width: 100%; height: auto;" />'

    return re.sub(r"!\[\]\(([^)]+)\)", replace_image, md_content)


def format_performance_markdown(perf_rows: List[List[str]], display_mode: str = "all") -> str:
    if not perf_rows:
        return "**No performance data available yet.**\n\nRun a parsing task to see performance metrics."

    headers = {
        "all": ("File", "Pages", "Time", "Speed"),
        "time": ("File", "Pages", "Time", "Speed"),
        "items": ("File", "Pages", "Time", "Speed"),
        "throughput": ("File", "Pages", "Time", "Speed"),
    }
    h0, h1, h2, h3 = headers.get(display_mode, headers["all"])

    lines = [
        f"| {h0} | {h1} | {h2} | {h3} |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in perf_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    return "\n".join(lines)


def update_engine_settings(preset: str) -> Tuple[str, str, str, str]:
    if preset == "onnxruntime":
        return "onnxruntime", "onnxruntime", "onnxruntime", "onnxruntime"
    return "dxengine", "dxengine", "onnxruntime", "dxengine"


def _iter_valid_files(file_paths: Iterable[str]) -> List[Path]:
    supported_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    valid_files: List[Path] = []
    for file_path in file_paths or []:
        path = Path(file_path)
        if path.is_file() and path.suffix.lower() in supported_suffixes:
            valid_files.append(path)
    return valid_files


def parse_document(
    file_paths: List[str],
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    layout_engine: str,
    ocr_engine: str,
    formula_engine: str,
    table_engine: str,
    use_async_pipeline: bool,  # kept for UI compatibility
    progress=gr.Progress(),
) -> Tuple[str, str, str | None, List[str], List[List[str]], str]:
    del use_async_pipeline

    try:
        valid_files = _iter_valid_files(file_paths)
        if not valid_files:
            return "", "❌ Please upload at least one PDF or image file.", None, [], [], ""

        progress(0.05, desc="Preparing request...")

        with contextlib.ExitStack() as stack:
            files = []
            for path in valid_files:
                mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                files.append(("files", (path.name, stack.enter_context(path.open("rb")), mime_type)))

            data = {
                "output_dir": str(DEFAULT_OUTPUT_DIR),
                "clear_output_file": "false",
                "backend": "pipeline",
                "parse_method": parse_method,
                "formula_enable": str(formula_enable).lower(),
                "table_enable": str(table_enable).lower(),
                "deepx": "true",
                "layout_engine": layout_engine,
                "ocr_engine": ocr_engine,
                "formula_engine": formula_engine,
                "table_engine": table_engine,
                "return_md": "true",
                "return_middle_json": "false",
                "return_model_output": "false",
                "return_content_list": "true",
                "return_images": "false",
                "start_page_id": "0",
                "end_page_id": "99999",
            }

            progress(0.2, desc="Calling C++ API server...")
            response = requests.post(
                f"{SERVER_URL}/file_parse",
                files=files,
                data=data,
                timeout=(10, 600),
            )

        response.raise_for_status()
        payload = response.json()

        progress(0.8, desc="Collecting outputs...")

        markdown_sections: List[str] = []
        preview_sections: List[str] = []
        extracted_images: List[str] = []
        performance_rows: List[List[str]] = []
        info_lines = [
            "✅ Batch Parsing Complete!",
            "",
            f"📡 API Server: {SERVER_URL}",
            f"📁 Output Directory: {DEFAULT_OUTPUT_DIR}",
            f"📄 Total Files: {payload.get('total_files', 0)}",
            f"✅ Successful Files: {payload.get('successful_files', 0)}",
            "",
        ]

        layout_file: str | None = None
        total_pages = 0
        total_time_ms = 0.0

        for item in payload.get("results", []):
            filename = item.get("filename", "unknown")
            if "error" in item:
                info_lines.append(f"❌ {filename}: {item['error']}")
                continue

            output_dir = Path(item.get("output_dir", DEFAULT_OUTPUT_DIR))
            image_dir = output_dir / "images"
            md_content = item.get("md_content", "")
            md_preview = convert_md_images_for_gradio(md_content, image_dir)

            markdown_sections.append(f"## 📄 {filename}\n\n{md_content}\n\n---\n")
            preview_sections.append(f"## 📄 {filename}\n\n{md_preview}\n\n---\n")

            if image_dir.exists():
                for image_path in sorted(image_dir.glob("*")):
                    if image_path.is_file():
                        extracted_images.append(str(image_path))

            if not layout_file:
                layout_files = item.get("layout_files", [])
                if layout_files:
                    layout_file = layout_files[0]

            stats = item.get("stats", {})
            pages = int(stats.get("pages", 0))
            time_ms = float(stats.get("time_ms", 0.0))
            total_pages += pages
            total_time_ms += time_ms
            speed = f"{(pages / (time_ms / 1000.0)):.2f} it/s" if time_ms > 0 and pages > 0 else "-"
            performance_rows.append([
                filename,
                str(pages),
                f"{time_ms / 1000.0:.2f}s",
                speed,
            ])

            info_lines.extend([
                f"📄 {filename}",
                f"  Pages: {pages}",
                f"  Time: {time_ms / 1000.0:.2f}s",
            ])

            for warning in item.get("warnings", []):
                info_lines.append(f"  Warning: {warning}")
            for warning in item.get("request_warnings", []):
                info_lines.append(f"  Request Warning: {warning}")
            info_lines.append("")

        if total_pages > 0:
            total_speed = f"{(total_pages / (total_time_ms / 1000.0)):.2f} it/s" if total_time_ms > 0 else "-"
            performance_rows.append([
                "Total",
                str(total_pages),
                f"{total_time_ms / 1000.0:.2f}s",
                total_speed,
            ])
            info_lines.extend([
                f"📊 Total Pages: {total_pages}",
                f"⏱️ Total Time: {total_time_ms / 1000.0:.2f}s",
                f"⚡ Average Speed: {total_speed}",
            ])

        progress(1.0, desc="Complete!")

        return (
            "\n".join(markdown_sections).strip(),
            "\n".join(info_lines).strip(),
            layout_file,
            extracted_images,
            performance_rows,
            "\n".join(preview_sections).strip(),
        )
    except requests.RequestException as exc:
        return "", f"❌ Failed to call C++ API server `{SERVER_URL}`:\n{exc}", None, [], [], ""
    except Exception as exc:  # pragma: no cover - UI fallback
        return "", f"❌ Error occurred:\n{exc}", None, [], [], ""


async def _set_language_cookie(request, call_next):
    response = await call_next(request)
    if response is not None:
        response.set_cookie("language", "en", path="/", max_age=30 * 24 * 3600)
    return response


def create_ui():
    custom_css = """
    #md-preview,
    #md-preview > div {
        overflow: auto;
    }

    #right-pane {
        max-width: 900px;
        min-width: 720px;
    }

    #right-pane .tabitem {
        max-height: 720px;
        overflow: auto;
    }

    #md-preview img {
        max-width: 100%;
        height: auto;
    }
    """

    with gr.Blocks(
        title="RapidDocCpp - C++ Backend",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(
                    f"""
                    # RapidDocCpp - Document Parsing (C++ Backend)

                    **Gradio UI transplanted from Python RapidDoc and bound to the C++ API server**

                    Current API server: `{SERVER_URL}`
                    """
                )

            with gr.Column(scale=2):
                gr.Markdown("### Performance Summary")
                perf_display = gr.Markdown(
                    value="**No performance data available yet.**\n\nRun a parsing task to see performance metrics."
                )
                perf_mode = gr.Radio(
                    choices=[
                        ("All", "all"),
                        ("Time", "time"),
                        ("Items", "items"),
                        ("Throughput", "throughput"),
                    ],
                    value="all",
                    label="",
                    container=False,
                )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="File Upload (PDF or Image) - Multiple files supported",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
                    file_count="multiple",
                )

                with gr.Group():
                    gr.Markdown("### Engine Selection")
                    engine_preset = gr.Radio(
                        choices=[
                            ("DeepX NPU (Recommended)", "deepx-npu"),
                            ("ONNX Runtime (Compatibility Placeholder)", "onnxruntime"),
                        ],
                        value="deepx-npu",
                        label="Engine Preset",
                        info="The C++ backend currently executes with DeepX engines; non-DeepX choices are kept only for UI compatibility.",
                    )

                layout_engine = gr.State(value="dxengine")
                ocr_engine = gr.State(value="dxengine")
                formula_engine = gr.State(value="onnxruntime")
                table_engine = gr.State(value="dxengine")

                with gr.Group():
                    gr.Markdown("### Basic Settings")
                    parse_method = gr.Radio(
                        choices=["auto", "ocr", "txt"],
                        value="auto",
                        label="Parsing Method",
                        info="The C++ backend falls back to the standard pipeline for txt mode.",
                    )

                formula_enable = gr.State(value=True)
                table_enable = gr.State(value=True)
                use_async_pipeline = gr.State(value=True)

                parse_btn = gr.Button("Start Parsing", variant="primary", size="lg")

                with gr.Accordion("Parsing Information", open=True):
                    info_output = gr.Textbox(
                        label="info box",
                        lines=13,
                        max_lines=40,
                        show_copy_button=True,
                    )

            with gr.Column(scale=2, elem_id="right-pane"):
                with gr.Tabs():
                    with gr.Tab("Markdown Preview"):
                        md_preview = gr.Markdown(value="", elem_id="md-preview")

                    with gr.Tab("Markdown Source"):
                        md_output = gr.Textbox(
                            label="Markdown Text (for copy)",
                            lines=100,
                            show_copy_button=True,
                        )

                    with gr.Tab("Extracted Images"):
                        image_gallery = gr.Gallery(
                            label="Images Extracted from Document",
                            columns=3,
                            height="auto",
                            object_fit="contain",
                        )

                    with gr.Tab("Layout Visualization"):
                        layout_output = gr.File(label="Layout Visualization File")

        perf_rows_state = gr.State(value=[])

        def parse_and_display(
            file_paths,
            parse_method,
            formula_enable,
            table_enable,
            layout_engine,
            ocr_engine,
            formula_engine,
            table_engine,
            use_async_pipeline,
            perf_mode,
            progress=gr.Progress(),
        ):
            raw_md, info_text, layout_file, images, perf_rows, preview_md = parse_document(
                file_paths,
                parse_method,
                formula_enable,
                table_enable,
                layout_engine,
                ocr_engine,
                formula_engine,
                table_engine,
                use_async_pipeline,
                progress,
            )
            perf_md = format_performance_markdown(perf_rows, perf_mode)
            return raw_md, preview_md, info_text, layout_file, images, perf_md, perf_rows

        def update_perf_display(perf_rows, perf_mode):
            return format_performance_markdown(perf_rows, perf_mode)

        engine_preset.change(
            fn=update_engine_settings,
            inputs=[engine_preset],
            outputs=[layout_engine, ocr_engine, formula_engine, table_engine],
        )

        parse_btn.click(
            fn=parse_and_display,
            inputs=[
                file_input,
                parse_method,
                formula_enable,
                table_enable,
                layout_engine,
                ocr_engine,
                formula_engine,
                table_engine,
                use_async_pipeline,
                perf_mode,
            ],
            outputs=[
                md_output,
                md_preview,
                info_output,
                layout_output,
                image_gallery,
                perf_display,
                perf_rows_state,
            ],
        )

        perf_mode.change(
            fn=update_perf_display,
            inputs=[perf_rows_state, perf_mode],
            outputs=[perf_display],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RapidDocCpp Gradio UI")
    parser.add_argument("--server-url", default=SERVER_URL, help="C++ API server base URL")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio bind host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio bind port")
    args = parser.parse_args()

    SERVER_URL = args.server_url.rstrip("/")
    bind_host = "0.0.0.0" if args.host in {"0.0.0.0", "*", ""} else args.host
    lan_ip = detect_lan_ip()

    print(f"RapidDocCpp Gradio UI binding on: http://{bind_host}:{args.port}")
    if bind_host == "0.0.0.0":
        print(f"RapidDocCpp Gradio UI LAN URL: http://{lan_ip}:{args.port}")
    print(f"RapidDocCpp C++ API server: {SERVER_URL}")

    demo = create_ui()
    demo.app.add_middleware(BaseHTTPMiddleware, dispatch=_set_language_cookie)
    demo.launch(
        server_name=bind_host,
        server_port=args.port,
        share=False,
        show_error=True,
        allowed_paths=[str(DEFAULT_OUTPUT_DIR), str(DEFAULT_OUTPUT_DIR.resolve())],
    )
