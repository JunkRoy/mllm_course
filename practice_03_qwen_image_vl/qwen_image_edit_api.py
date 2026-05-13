"""通过阿里云百炼 DashScope API 调用 Qwen-Image-Edit 的示例。"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests
import yaml

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError as exc:  # pragma: no cover - exercised by users without deps.
    raise SystemExit(
        "缺少依赖：dashscope。请先运行 "
        "`pip install -r practice_03_qwen_image_vl/requirements.txt`."
    ) from exc


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | None, base_dir: Path) -> Path | None:
    """把配置中的相对路径解析为相对 config.yaml 的绝对路径。"""

    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def image_to_data_url(path: Path) -> str:
    """把本地图片编码成 DashScope 可接收的 data URL。"""

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def normalize_image_input(value: str, config_dir: Path) -> str:
    """统一处理图片输入：URL 直接使用，本地路径转成 data URL。"""

    if value.startswith(("http://", "https://", "data:")):
        return value
    path = resolve_path(value, config_dir)
    if path is None or not path.exists():
        raise FileNotFoundError(f"API input image does not exist: {path}")
    return image_to_data_url(path)


def build_messages(api_cfg: dict[str, Any], config_dir: Path) -> list[dict[str, Any]]:
    """按 DashScope 多模态对话格式组装图片和提示词。"""

    images = api_cfg.get("input_images") or []
    if not images:
        raise ValueError("qwen_image.api.input_images must contain at least one image.")

    content: list[dict[str, str]] = []
    # 文档要求图片和文本都放在 content 列表中，图片可以有一张或多张。
    for image in images:
        content.append({"image": normalize_image_input(image, config_dir)})
    content.append({"text": api_cfg["prompt"]})
    return [{"role": "user", "content": content}]


def call_qwen_image_edit(api_cfg: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """读取 API Key 和参数，并调用 DashScope 图像编辑接口。"""

    api_key = os.getenv(api_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
    if not api_key:
        env_name = api_cfg.get("api_key_env", "DASHSCOPE_API_KEY")
        raise EnvironmentError(f"调用 DashScope API 前请先设置环境变量：{env_name}")

    # parameters 中值为 null 的字段不传给 API，避免覆盖服务端默认值。
    parameters = {
        key: value
        for key, value in (api_cfg.get("parameters") or {}).items()
        if value is not None
    }
    if api_cfg.get("base_http_api_url"):
        dashscope.base_http_api_url = api_cfg["base_http_api_url"]

    # MultiModalConversation.call 会返回包含图片 URL 的响应对象。
    return MultiModalConversation.call(
        api_key=api_key,
        model=api_cfg.get("model", "qwen-image-2.0-pro"),
        messages=build_messages(api_cfg, config_dir),
        stream=False,
        **parameters,
    )


def response_to_dict(response: Any) -> dict[str, Any]:
    """把 DashScope 响应对象统一转换成普通 dict，便于保存和解析。"""

    # DashScopeResponse 继承自 dict，并重载了 __getattr__。
    # 先用 Mapping 判断，避免 hasattr(response, "to_dict") 触发 KeyError。
    if isinstance(response, Mapping):
        return dict(response)

    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return json.loads(json.dumps(response, default=lambda obj: getattr(obj, "__dict__", str(obj))))


def looks_like_image_url(value: str) -> bool:
    """判断字符串是否像图片 URL 或 data URL。"""

    if value.startswith("data:image/"):
        return True
    if not value.startswith(("http://", "https://")):
        return False
    path = value.split("?", 1)[0].lower()
    return path.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))


def find_image_urls(value: Any) -> list[str]:
    """递归扫描响应结构，提取可能的图片 URL。"""

    if isinstance(value, str):
        return [value] if looks_like_image_url(value) else []
    if isinstance(value, Mapping):
        urls: list[str] = []
        for item in value.values():
            urls.extend(find_image_urls(item))
        return urls
    if isinstance(value, list):
        urls = []
        for item in value:
            urls.extend(find_image_urls(item))
        return urls
    return []


def collect_image_urls(payload: dict[str, Any]) -> list[str]:
    """从 DashScope 响应中提取生成图片 URL。"""

    output = payload.get("output") or {}
    choices = output.get("choices") or []
    urls: list[str] = []
    for choice in choices:
        message = choice.get("message") or {}
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("image"):
                urls.append(item["image"])

    if not urls:
        urls = find_image_urls(payload)

    return list(dict.fromkeys(urls))


def download_images(urls: list[str], output_dir: Path, timeout: int = 60) -> list[str]:
    """下载 API 返回的图片 URL，并保存到输出目录。"""

    paths: list[str] = []
    for idx, url in enumerate(urls):
        if url.startswith("data:image/"):
            header, encoded = url.split(",", 1)
            suffix = "." + header.split(";", 1)[0].split("/", 1)[1]
            path = output_dir / f"api_edited_{idx:03d}{suffix}"
            path.write_bytes(base64.b64decode(encoded))
            paths.append(str(path))
            continue

        # URL 可能带查询参数，先去掉参数再推断文件后缀。
        suffix = Path(url.split("?", 1)[0]).suffix or ".png"
        path = output_dir / f"api_edited_{idx:03d}{suffix}"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        path.write_bytes(response.content)
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通过 DashScope API 调用 Qwen-Image-Edit。")
    parser.add_argument("--config", default="config.yaml", help="config.yaml 配置文件路径")
    parser.add_argument(
        "--save-response",
        action="store_true",
        help="把完整 API 响应保存为 JSON，便于排查错误。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = load_config(config_path)
    api_cfg = cfg["qwen_image"]["api"]

    # 输出目录相对配置文件解析，脚本从任意工作目录运行都能找到正确位置。
    output_dir = resolve_path(api_cfg.get("output_dir", "outputs"), config_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    response = call_qwen_image_edit(api_cfg, config_dir)
    payload = response_to_dict(response)
    # DashScope 失败时通常会在响应里返回 status_code、code 和 message。
    if payload.get("status_code") not in (None, 200):
        raise RuntimeError(
            f"DashScope API failed: status_code={payload.get('status_code')}, "
            f"code={payload.get('code')}, message={payload.get('message')}"
        )

    image_urls = collect_image_urls(payload)
    saved_paths = download_images(image_urls, output_dir)

    response_path: str | None = None
    if args.save_response:
        # 保存完整响应有助于确认服务端返回结构是否发生变化。
        path = output_dir / "api_response.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        response_path = str(path)

    summary = {
        "backend": "dashscope_api",
        "model": api_cfg.get("model", "qwen-image-2.0-pro"),
        "image_count": len(saved_paths),
        "image_urls": image_urls,
        "saved_paths": saved_paths,
        "response_path": response_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
