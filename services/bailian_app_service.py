"""百炼（阿里云 AI）应用调用服务模块。

封装阿里云 DashScope 百炼应用的 REST API 调用，提供：
- 同步调用：ask() 方法
- 异步调用：ask_async() 方法

调用百炼应用的 /completion 接口，传入 prompt 获取 AI 回复。
包含完善的错误处理、超时控制、日志记录和降级策略。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

# 调用失败时的默认降级回复文本
FALLBACK_TEXT = "不好意思，导游服务响应超时，请再问一次。"
# 默认 API 基础地址
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
# 默认请求超时（秒）
DEFAULT_TIMEOUT = 30

logger = logging.getLogger(__name__)


class BailianAppService:
    """百炼应用调用服务。

    封装阿里云 DashScope 百炼应用的 HTTP API 调用逻辑，
    负责 prompt 提交和 AI 回复提取。

    Attributes:
        api_key: DashScope API Key
        app_id: 百炼应用 ID
        base_url: API 基础地址
        timeout: HTTP 请求超时时间
    """

    def __init__(
        self,
        api_key: str | None = None,
        app_id: str | None = None,
        base_url: str | None = None,
        timeout: int | float | None = None,
    ):
        """初始化百炼服务。

        Args:
            api_key: DashScope API Key，默认从环境变量 BAILIAN_API_KEY 读取
            app_id: 百炼应用 ID，默认从环境变量 BAILIAN_APP_ID 读取
            base_url: API 基础地址，默认从环境变量 BAILIAN_APP_BASE_URL 读取
            timeout: 请求超时，默认从环境变量 BAILIAN_TIMEOUT 读取（30秒）
        """
        self.api_key = (api_key if api_key is not None else os.getenv("BAILIAN_API_KEY", "")).strip()
        self.app_id = (app_id if app_id is not None else os.getenv("BAILIAN_APP_ID", "")).strip()
        self.base_url = (
            base_url if base_url is not None else os.getenv("BAILIAN_APP_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.timeout = _read_timeout(timeout)

    def ask(self, prompt: str) -> str:
        """同步调用百炼应用（内部使用 asyncio.run）。

        Args:
            prompt: 用户提示词

        Returns:
            str: AI 回复文本
        """
        return asyncio.run(self.ask_async(prompt))

    async def ask_async(self, prompt: str) -> str:
        """异步调用百炼应用。

        完整的调用流程：
        1. 参数校验（API Key、App ID）
        2. 构建请求体并发送 POST 请求
        3. 解析 JSON 响应
        4. 提取输出文本
        5. 任何步骤失败都返回降级文本

        Args:
            prompt: 用户提示词

        Returns:
            str: AI 回复文本，失败时返回 FALLBACK_TEXT
        """
        total_start = time.perf_counter()
        # 构建请求日志
        request_log = {
            "prompt_len": len(prompt),
            "prompt_preview": _preview_text(prompt),
            "payload_keys": ["input", "parameters"],
            "timeout": self.timeout,
            "base_url": self.base_url,
            "app_id_masked": _mask_app_id(self.app_id),
            "has_HTTP_PROXY": bool(os.getenv("HTTP_PROXY")),
            "has_HTTPS_PROXY": bool(os.getenv("HTTPS_PROXY")),
        }
        logger.info("[BAILIAN] request %s", json.dumps(request_log, ensure_ascii=False))
        print(f"[BAILIAN] request {json.dumps(request_log, ensure_ascii=False)}", flush=True)

        # 校验 API Key
        if not self.api_key:
            logger.error("[BAILIAN] BAILIAN_API_KEY 未配置")
            failed_after = time.perf_counter() - total_start
            _log_bailian_result(failed_after, "", "missing_api_key", failed_after)
            print(f"[BAILIAN-TIME] failed_after={failed_after:.3f}s error=缺失 API Key", flush=True)
            return FALLBACK_TEXT

        # 校验 App ID
        if not self.app_id:
            logger.error("[BAILIAN] BAILIAN_APP_ID 未配置")
            failed_after = time.perf_counter() - total_start
            _log_bailian_result(failed_after, "", "missing_app_id", failed_after)
            print(f"[BAILIAN-TIME] failed_after={failed_after:.3f}s error=缺失 App ID", flush=True)
            return FALLBACK_TEXT

        # 构建请求
        build_start = time.perf_counter()
        url = f"{self.base_url}/api/v1/apps/{self.app_id}/completion"
        payload = {
            "input": {
                "prompt": prompt,
            },
            "parameters": {},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        print(
            f"[BAILIAN-TIME] build_payload={time.perf_counter() - build_start:.3f}s "
            "input_keys=['prompt']",
            flush=True,
        )
        print(f"[BAILIAN-TIME] http_start url={url}", flush=True)

        # 发送 HTTP 请求
        http_start = time.perf_counter()
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except Exception as exc:
            logger.exception("[BAILIAN] 请求失败")
            failed_after = time.perf_counter() - total_start
            _log_bailian_result(failed_after, "", type(exc).__name__, failed_after)
            print(f"[BAILIAN-TIME] failed_after={failed_after:.3f}s error={exc}", flush=True)
            return FALLBACK_TEXT

        response_text = response.text or ""
        print(
            f"[BAILIAN-TIME] http_total={time.perf_counter() - http_start:.3f}s "
            f"status={response.status_code} response_chars={len(response_text)}",
            flush=True,
        )

        # 检查 HTTP 状态码
        if response.status_code != 200:
            logger.error(
                "[BAILIAN] HTTP %s response_preview=%s",
                response.status_code,
                _preview_text(response_text),
            )
            print(
                f"[BAILIAN-TIME] failed_after={time.perf_counter() - total_start:.3f}s "
                f"error=HTTP {response.status_code}",
                flush=True,
            )
            failed_after = time.perf_counter() - total_start
            _log_bailian_result(failed_after, "", f"HTTP_{response.status_code}", failed_after)
            return FALLBACK_TEXT

        # 解析 JSON 响应
        json_start = time.perf_counter()
        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            logger.exception("[BAILIAN] 无效的 JSON response_preview=%s", _preview_text(response_text))
            print(
                f"[BAILIAN-TIME] json_parse={time.perf_counter() - json_start:.3f}s error={exc}",
                flush=True,
            )
            failed_after = time.perf_counter() - total_start
            print(f"[BAILIAN-TIME] failed_after={failed_after:.3f}s error=无效 JSON", flush=True)
            _log_bailian_result(failed_after, "", "invalid_json", failed_after)
            return FALLBACK_TEXT
        print(f"[BAILIAN-TIME] json_parse={time.perf_counter() - json_start:.3f}s", flush=True)

        # 提取 AI 回复文本
        extract_start = time.perf_counter()
        answer = _extract_text(data)
        if not answer:
            logger.error("[BAILIAN] 响应缺少 output.text full_json=%s", json.dumps(data, ensure_ascii=False))
            print(
                f"[BAILIAN-TIME] extract_answer={time.perf_counter() - extract_start:.3f}s answer_chars=0",
                flush=True,
            )
            failed_after = time.perf_counter() - total_start
            print(f"[BAILIAN-TIME] failed_after={failed_after:.3f}s error=缺少 output.text", flush=True)
            _log_bailian_result(failed_after, "", "missing_output_text", failed_after)
            return FALLBACK_TEXT

        answer = answer.strip()
        print(
            f"[BAILIAN-TIME] extract_answer={time.perf_counter() - extract_start:.3f}s "
            f"answer_chars={len(answer)}",
            flush=True,
        )
        elapsed = time.perf_counter() - total_start
        print(f"[BAILIAN-TIME] total={elapsed:.3f}s", flush=True)
        _log_bailian_result(elapsed, answer, "", None)
        return answer


def _extract_text(data: dict[str, Any]) -> str:
    """从百炼 API 响应中提取 output.text 字段。

    Args:
        data: API 响应字典

    Returns:
        str: 提取的文本，失败返回空字符串
    """
    output = data.get("output")
    if isinstance(output, dict):
        text = output.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _read_timeout(timeout: int | float | None) -> int | float:
    """读取超时配置，支持环境变量覆盖。

    Args:
        timeout: 显式传入的超时值

    Returns:
        int | float: 超时值
    """
    if timeout is not None:
        return timeout
    raw_value = os.getenv("BAILIAN_TIMEOUT", str(DEFAULT_TIMEOUT)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        logger.error("[BAILIAN] 无效的 BAILIAN_TIMEOUT=%r，使用默认值 %s", raw_value, DEFAULT_TIMEOUT)
        return DEFAULT_TIMEOUT
    if value <= 0:
        logger.error("[BAILIAN] 无效的 BAILIAN_TIMEOUT=%r，使用默认值 %s", raw_value, DEFAULT_TIMEOUT)
        return DEFAULT_TIMEOUT
    return value


def _preview_text(text: str, limit: int = 500) -> str:
    """截取文本预览（用于日志），特殊字符转义。

    Args:
        text: 原始文本
        limit: 最大字符数

    Returns:
        str: 转义并截断后的文本
    """
    normalized = (text or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _mask_app_id(app_id: str) -> str:
    """脱敏处理 App ID，仅保留首尾部分用于日志。

    Args:
        app_id: 原始 App ID

    Returns:
        str: 脱敏后的 App ID
    """
    if not app_id:
        return ""
    if len(app_id) <= 8:
        return f"{app_id[:1]}***{app_id[-1:]}"
    return f"{app_id[:4]}***{app_id[-4:]}"


def _log_bailian_result(
    elapsed: float,
    answer: str,
    error_type: str,
    failed_after: float | None,
) -> None:
    """记录百炼调用结果日志（结构化 JSON）。

    Args:
        elapsed: 总耗时（秒）
        answer: AI 回复文本
        error_type: 错误类型（空字符串表示成功）
        failed_after: 失败时间点（成功时为 None）
    """
    payload = {
        "elapsed_ms": int(elapsed * 1000),
        "answer_preview": _preview_text(answer),
        "error_type": error_type,
        "failed_after": None if failed_after is None else int(failed_after * 1000),
    }
    logger.info("[BAILIAN] response %s", json.dumps(payload, ensure_ascii=False))
    print(f"[BAILIAN] response {json.dumps(payload, ensure_ascii=False)}", flush=True)
