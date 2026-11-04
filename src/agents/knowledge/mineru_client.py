from __future__ import annotations

import asyncio
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from src.core.config import get_settings

settings = get_settings()


class MinerUError(RuntimeError):
    """MinerU 调用失败。"""


@dataclass
class MinerUResult:
    """MinerU 解析结果，保留 Markdown 与带页码的 content_list。"""

    markdown: str
    content_list: list[dict[str, Any]]
    page_count: int
    batch_id: str | None = None


def _headers() -> dict[str, str]:
    token = settings.MINERU_API_TOKEN.strip()
    if not token:
        raise MinerUError(
            "未配置 MINERU_API_TOKEN，请在 .env 中填写 mineru.net 的 API Token"
        )
    return {"Authorization": f"Bearer {token}"}


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=30.0,
        read=float(settings.MINERU_REQUEST_TIMEOUT),
        write=float(settings.MINERU_REQUEST_TIMEOUT),
        pool=30.0,
    )


async def parse_document(
    file_path: str | None = None,
    file_name: str | None = None,
    *,
    file_bytes: bytes | None = None,
    is_ocr: bool | None = None,
) -> MinerUResult:
    """
    调用 mineru.net 精准解析 API v4，返回 Markdown + content_list。

    支持两种输入：
    - parse_document("a.pdf")
    - parse_document(file_bytes=data, file_name="a.pdf")

    is_ocr:
    - True：扫描件强制 OCR
    - False：只走文本层
    - None：使用 MinerU 默认策略
    """
    if file_bytes is None:
        if not file_path:
            raise ValueError("file_path 和 file_bytes 至少提供一个")
        file_bytes = Path(file_path).read_bytes()
    if not file_bytes:
        raise ValueError("文件内容为空，无法解析")

    if not file_name:
        if not file_path:
            raise ValueError("使用 file_bytes 时必须提供 file_name")
        file_name = Path(file_path).name

    headers = _headers()
    base = settings.MINERU_API_BASE.rstrip("/")

    async with httpx.AsyncClient(timeout=_timeout(), follow_redirects=True) as client:
        # 1. 申请签名上传地址（精准解析 API 支持本地文件上传）
        file_meta: dict[str, Any] = {"name": file_name}
        if is_ocr is not None:
            file_meta["is_ocr"] = is_ocr

        payload: dict[str, Any] = {
            "files": [file_meta],
            "model_version": settings.MINERU_MODEL_VERSION,
            "language": settings.MINERU_LANGUAGE,
            "enable_table": True,
            "enable_formula": True,
        }

        data = await _request_json(
            client,
            "POST",
            f"{base}/api/v4/file-urls/batch",
            headers,
            json=payload,
        )
        batch_id = data.get("batch_id")
        upload_urls = data.get("file_urls") or []
        if not batch_id or not upload_urls:
            raise MinerUError(f"MinerU 未返回上传地址: {data}")

        # 2. PUT 到 OSS。不要带 Authorization，也不要手动设置 Content-Type
        resp = await client.put(upload_urls[0], content=file_bytes)
        resp.raise_for_status()

        # 3. 轮询批量结果
        zip_url = await _wait_batch(client, base, headers, batch_id, file_name)

        # 4. 下载并解析 ZIP
        resp = await client.get(zip_url)
        resp.raise_for_status()
        result = _parse_zip(resp.content)
        result.batch_id = batch_id
        return result


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict[str, str],
    **kwargs: Any,
) -> dict[str, Any]:
    resp = await client.request(method, url, headers=headers, **kwargs)
    try:
        body = resp.json()
    except Exception:
        resp.raise_for_status()
        raise MinerUError(
            f"MinerU 返回非 JSON: HTTP {resp.status_code} {resp.text[:300]}"
        )

    if resp.status_code >= 400 or body.get("code") != 0:
        raise MinerUError(
            "MinerU 调用失败: "
            f"code={body.get('code')} msg={body.get('msg')} HTTP {resp.status_code}"
        )

    data = body.get("data")
    if not isinstance(data, dict):
        raise MinerUError(f"MinerU 返回格式异常: {body}")
    return data


async def _wait_batch(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    batch_id: str,
    file_name: str,
) -> str:
    deadline = time.monotonic() + settings.MINERU_POLL_TIMEOUT
    last_state = ""

    while True:
        try:
            data = await _request_json(
                client,
                "GET",
                f"{base}/api/v4/extract-results/batch/{batch_id}",
                headers,
            )
        except httpx.HTTPError as e:
            # 轮询期间网络抖动不应直接判定解析失败，保留 batch_id 继续查
            logger.warning(f"MinerU 轮询网络错误，稍后重试: {e}")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"MinerU 轮询超时（{settings.MINERU_POLL_TIMEOUT}s），"
                    f"batch_id={batch_id}，可凭 batch_id 继续查询，不要重复提交"
                )
            await asyncio.sleep(settings.MINERU_POLL_INTERVAL)
            continue

        items = data.get("extract_result") or []
        item = next(
            (i for i in items if i.get("file_name") == file_name),
            items[0] if items else None,
        )

        if item:
            state = item.get("state", "")
            if state != last_state:
                logger.info(
                    "MinerU batch={} file={} state={} progress={}",
                    batch_id,
                    file_name,
                    state,
                    item.get("extract_progress"),
                )
                last_state = state

            if state == "done":
                zip_url = item.get("full_zip_url")
                if not zip_url:
                    raise MinerUError(f"MinerU 任务完成但没有 full_zip_url: {item}")
                return zip_url

            if state == "failed":
                raise MinerUError(f"MinerU 解析失败: {item.get('err_msg') or item}")

        if time.monotonic() > deadline:
            raise TimeoutError(
                f"MinerU 解析超时（{settings.MINERU_POLL_TIMEOUT}s），"
                f"batch_id={batch_id}，可凭 batch_id 继续查询，不要重复提交"
            )

        await asyncio.sleep(settings.MINERU_POLL_INTERVAL)


def _parse_zip(blob: bytes) -> MinerUResult:
    """从 MinerU 结果 ZIP 中读取 Markdown 与 content_list。"""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        md_name = next((n for n in names if n.endswith("full.md")), None)
        cl_name = next((n for n in names if n.endswith("content_list.json")), None)

        markdown = zf.read(md_name).decode("utf-8") if md_name else ""
        if cl_name:
            content_list = json.loads(zf.read(cl_name).decode("utf-8"))
        elif markdown:
            content_list = [{"type": "text", "text": markdown, "page_idx": 0}]
        else:
            raise MinerUError(
                f"MinerU ZIP 中没有 Markdown 或 content_list: {names}"
            )

    if not isinstance(content_list, list):
        raise MinerUError(f"content_list 格式异常: {type(content_list)}")

    # 兼容 content_list v2（按页分组）格式
    if content_list and isinstance(content_list[0], list):
        flattened: list[dict[str, Any]] = []
        for page_idx, page_blocks in enumerate(content_list):
            for block in page_blocks:
                if isinstance(block, dict):
                    block.setdefault("page_idx", page_idx)
                    flattened.append(block)
        content_list = flattened

    page_count = max(
        (int(b.get("page_idx", 0)) for b in content_list if isinstance(b, dict)),
        default=0,
    ) + 1
    return MinerUResult(
        markdown=markdown,
        content_list=content_list,
        page_count=page_count,
    )


async def check_mineru_health() -> dict[str, Any]:
    """检查 MinerU 配置是否可用（不消耗解析额度）。"""
    if not settings.MINERU_API_TOKEN.strip():
        return {"status": "unavailable", "error": "MINERU_API_TOKEN 未配置"}
    return {
        "status": "configured",
        "base": settings.MINERU_API_BASE,
        "model_version": settings.MINERU_MODEL_VERSION,
    }