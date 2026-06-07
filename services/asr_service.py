"""语音识别（ASR）服务模块。

支持两种模式：
1. Mock 模式：返回固定文本，用于开发测试
2. DashScope 模式：调用阿里云 DashScope 实时语音识别 API（paraformer-realtime-v2）

通过环境变量 ASR_PROVIDER 切换模式，默认使用 mock。
"""

from __future__ import annotations

import os
from http import HTTPStatus
from pathlib import Path
from typing import Any

# Mock 模式下的固定返回文本，用于开发测试
MOCK_TEXT = "大雁塔有什么故事？"


def _log(message: str) -> None:
    """输出 ASR 日志，带 [ASR] 前缀并立即刷新。"""
    print(f"[ASR] {message}", flush=True)


def _safe_dump_result(result: Any) -> Any:
    """安全地将 ASR 结果对象转为可打印的形式。

    尝试转为 dict，失败时返回类型名。
    """
    if isinstance(result, dict):
        return dict(result)
    try:
        result_dict = getattr(result, "__dict__", None)
        if isinstance(result_dict, dict):
            return dict(result_dict)
    except Exception:
        pass
    return type(result).__name__


def _result_to_dict(result: Any) -> dict[str, Any]:
    """将 ASR 结果对象转为字典。

    提取常见的属性字段：status_code、code、message、output、usage、request_id。
    """
    if isinstance(result, dict):
        return result

    data: dict[str, Any] = {}
    for name in ("status_code", "code", "message", "output", "usage", "request_id"):
        try:
            if hasattr(result, name):
                data[name] = getattr(result, name)
        except Exception:
            pass
    return data


def _extract_sentence(result: Any) -> str:
    """从 DashScope ASR 结果中提取识别文本。

    按优先级尝试多种返回格式：
    1. 调用 get_sentence() 方法
    2. 从 output.text 读取
    3. 从 output.sentence.text 读取（字典格式）
    4. 从 output.sentence 列表拼接（列表格式）
    5. 从 output.sentences 列表拼接

    Raises:
        RuntimeError: 无法从结果中提取任何文本
    """
    # 尝试调用 get_sentence() 方法
    get_sentence = getattr(result, "get_sentence", None)
    if callable(get_sentence):
        try:
            sentence = get_sentence()
            if isinstance(sentence, str) and sentence.strip():
                return sentence.strip()
        except Exception:
            pass

    result_dict = _result_to_dict(result)
    # 尝试从 text 字段直接读取
    if isinstance(result_dict.get("text"), str) and result_dict["text"].strip():
        return result_dict["text"].strip()

    output = result_dict.get("output")
    if isinstance(output, dict):
        # 尝试 sentence 为字典格式：{"text": "..."}
        sentence = output.get("sentence")
        if isinstance(sentence, dict):
            sentence_text = sentence.get("text")
            if isinstance(sentence_text, str) and sentence_text.strip():
                return sentence_text.strip()
        # 尝试 sentence 为列表格式：[{"text": "..."}, ...]
        if isinstance(sentence, list):
            parts = [
                item.get("text", "").strip()
                for item in sentence
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item.get("text", "").strip()
            ]
            if parts:
                return "".join(parts)
        if isinstance(sentence, str) and sentence.strip():
            return sentence.strip()
        # 尝试 output.text
        output_text = output.get("text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        # 尝试 output.sentences 列表
        sentences = output.get("sentences")
        if isinstance(sentences, list):
            parts = [
                item.get("text", "")
                for item in sentences
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            text = "".join(parts).strip()
            if text:
                return text

    raise RuntimeError(f"DashScope ASR 响应中没有可识别的文本: {_safe_dump_result(result)}")


def _transcribe_with_dashscope(wav_path: Path) -> str:
    """使用阿里云 DashScope 进行语音识别。

    Args:
        wav_path: WAV 音频文件路径

    Returns:
        str: 识别出的文本

    Raises:
        RuntimeError: API Key 未配置、文件不存在或识别失败
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置")
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    import dashscope
    from dashscope.audio.asr import Recognition

    dashscope.api_key = api_key
    dashscope.base_websocket_api_url = os.getenv(
        "DASHSCOPE_WEBSOCKET_URL",
        "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
    )

    # 创建语音识别实例
    recognition = Recognition(
        model=os.getenv("ASR_MODEL", "paraformer-realtime-v2"),
        format="wav",
        sample_rate=16000,
        language_hints=["zh", "en"],  # 支持中英文混合识别
        callback=None,
    )
    result = recognition.call(str(wav_path))
    status_code = getattr(result, "status_code", None)
    if status_code != HTTPStatus.OK:
        message = getattr(result, "message", "")
        raise RuntimeError(f"DashScope ASR 失败 status={status_code} message={message}")

    text = _extract_sentence(result)
    _log(f"识别文本字符数={len(text)}")
    return text


def transcribe_wav(wav_path: str | Path) -> str:
    """语音识别主入口，将 WAV 音频转为文本。

    根据 ASR_PROVIDER 环境变量选择识别引擎：
    - "mock": 返回固定测试文本
    - "dashscope": 调用阿里云 DashScope 实时语音识别

    Args:
        wav_path: WAV 音频文件路径

    Returns:
        str: 识别出的文本

    Raises:
        ValueError: 不支持的 ASR_PROVIDER 值
    """
    provider = os.getenv("ASR_PROVIDER", "mock").strip().lower() or "mock"

    if provider == "mock":
        _log("使用 mock ASR，返回固定文本")
        return MOCK_TEXT
    if provider == "dashscope":
        return _transcribe_with_dashscope(Path(wav_path))

    raise ValueError(f"不支持的 ASR_PROVIDER: {provider}")
