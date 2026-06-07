"""文字转语音（TTS）服务模块。

支持两种模式：
1. Mock 模式：生成简单的正弦波音频，用于开发测试
2. DashScope 模式：调用阿里云 DashScope 多模态 TTS API（qwen3-tts-flash）

输出统一为 16kHz 单声道 16-bit PCM WAV 格式。
支持通过 ffmpeg 将非标准格式转换为 16k WAV。

通过环境变量 TTS_PROVIDER 切换模式，默认使用 mock。
"""

from __future__ import annotations

import base64
import binascii
import math
import os
import sys
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from urllib.parse import urlparse

import requests

# 标准音频参数：16kHz 采样率，单声道，16-bit
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
# 服务不可用时的默认错误提示文本
ERROR_TEXT = "抱歉，当前导游服务暂时不可用。"


def _log(message: str) -> None:
    """输出 TTS 日志，带 [TTS] 前缀并立即刷新。"""
    print(f"[TTS] {message}", flush=True)


def _pcm16_wav(pcm: bytes) -> bytes:
    """将原始 PCM 数据封装为标准 WAV 格式。

    Args:
        pcm: 原始 16-bit PCM 音频数据

    Returns:
        bytes: 标准 WAV 格式的音频数据
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def _silence_wav(duration_seconds: float = 1.0) -> bytes:
    """生成指定时长的静音 WAV 音频。

    Args:
        duration_seconds: 静音时长（秒），默认 1.0 秒

    Returns:
        bytes: 静音 WAV 数据
    """
    samples = max(1, int(SAMPLE_RATE * duration_seconds))
    return _pcm16_wav(b"\x00\x00" * samples)


def _mock_tts_wav(text: str) -> bytes:
    """Mock 模式 TTS：生成 440Hz 正弦波音频。

    音频持续时间与文本长度成正比（约 90ms/字），
    带有淡入淡出包络，模拟真实 TTS 效果。
    注意：生成的是模拟音，ASR 无法从中识别出语义内容。

    Args:
        text: 要"朗读"的文本

    Returns:
        bytes: 正弦波 WAV 音频数据
    """
    duration_seconds = min(max(1.0, len(text) * 0.09), 8.0)
    sample_count = int(SAMPLE_RATE * duration_seconds)
    amplitude = 4500
    pcm = bytearray()
    for i in range(sample_count):
        # 淡入淡出包络
        envelope = min(i / 800, (sample_count - i) / 800, 1.0)
        value = int(amplitude * max(envelope, 0.0) * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        pcm.extend(struct.pack("<h", value))
    return _pcm16_wav(bytes(pcm))


def _validate_wav_16k(wav_bytes: bytes) -> bytes:
    """验证 WAV 音频是否符合 16kHz 单声道 16-bit 标准。

    Args:
        wav_bytes: WAV 音频数据

    Returns:
        bytes: 原样返回（验证通过时）

    Raises:
        RuntimeError: 声道数、采样率、位深或压缩格式不匹配
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = Path(tmp.name)
        tmp.write(wav_bytes)
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != CHANNELS:
                raise RuntimeError(f"声道数不匹配: {wf.getnchannels()} (期望 {CHANNELS})")
            if wf.getframerate() != SAMPLE_RATE:
                raise RuntimeError(f"采样率不匹配: {wf.getframerate()} (期望 {SAMPLE_RATE})")
            if wf.getsampwidth() != SAMPLE_WIDTH:
                raise RuntimeError(f"位深不匹配: {wf.getsampwidth()} (期望 {SAMPLE_WIDTH})")
            if wf.getcomptype() != "NONE":
                raise RuntimeError(f"压缩格式不匹配: {wf.getcomptype()}")
        return wav_bytes
    finally:
        path.unlink(missing_ok=True)


def _convert_with_ffmpeg(input_path: Path, output_path: Path) -> None:
    """使用 ffmpeg 将音频文件转为 16kHz 单声道 WAV。

    Args:
        input_path: 输入音频文件路径
        output_path: 输出 WAV 文件路径

    Raises:
        RuntimeError: ffmpeg 未安装或转换失败
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("需要安装 ffmpeg 来转换 DashScope TTS 音频为 16k 单声道 WAV")

    command = [
        ffmpeg,
        "-y",              # 覆盖已存在的输出文件
        "-i", str(input_path),
        "-ac", str(CHANNELS),     # 单声道
        "-ar", str(SAMPLE_RATE),  # 16kHz 采样率
        "-sample_fmt", "s16",     # 16-bit 采样
        str(output_path),
    ]
    startupinfo = None
    # Windows 下隐藏控制台窗口
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        startupinfo=startupinfo,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 转换失败: {result.stderr.strip()}")


def ensure_wav_16k_mono(audio_bytes: bytes, input_suffix: str = ".wav") -> bytes:
    """确保音频数据为 16kHz 单声道 WAV。

    先尝试验证，如果不符合标准则使用 ffmpeg 转换。

    Args:
        audio_bytes: 原始音频数据
        input_suffix: 输入文件后缀名（用于 ffmpeg 识别格式）

    Returns:
        bytes: 16kHz 单声道 WAV 音频数据
    """
    if not audio_bytes:
        raise RuntimeError("空音频数据无法转换")

    # 先尝试直接验证
    try:
        return _validate_wav_16k(audio_bytes)
    except Exception as exc:
        _log(f"TTS 音频需要转换: {exc}")

    # 验证失败则通过 ffmpeg 转换
    with tempfile.TemporaryDirectory() as tmp_dir:
        suffix = input_suffix if input_suffix.startswith(".") else f".{input_suffix}"
        input_path = Path(tmp_dir) / f"tts_input{suffix}"
        output_path = Path(tmp_dir) / "tts_16k.wav"
        input_path.write_bytes(audio_bytes)
        _convert_with_ffmpeg(input_path, output_path)
        return _validate_wav_16k(output_path.read_bytes())


def _response_to_dict(response) -> dict:
    """将 DashScope API 响应对象转为字典。

    支持 dict、有 to_dict() 方法的对象，以及普通属性对象。
    """
    if isinstance(response, dict):
        return response
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    data: dict[str, object] = {}
    for name in ("status_code", "code", "message", "output", "usage", "request_id"):
        if hasattr(response, name):
            data[name] = getattr(response, name)
    return data


def _find_audio_url(value) -> str | None:
    """从 DashScope TTS 响应中递归查找音频下载 URL。

    Args:
        value: API 响应数据（dict 或 list）

    Returns:
        str | None: 找到的音频 URL，未找到返回 None
    """
    if isinstance(value, dict):
        # 尝试 audio.url 格式
        audio = value.get("audio")
        if isinstance(audio, dict):
            url = audio.get("url")
            if isinstance(url, str) and url:
                return url
        # 尝试直接 url 字段
        url = value.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return url
        # 递归搜索子字段
        for child in value.values():
            found = _find_audio_url(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_audio_url(child)
            if found:
                return found
    return None


def _find_audio_bytes(value) -> tuple[bytes, str] | None:
    """从 DashScope TTS 响应中递归查找内嵌的音频数据。

    支持格式：
    - 原始 bytes/bytearray
    - data: URL（base64 编码）
    - base64 字符串
    - 字典中的 data/audio/content/bytes/base64/audio_data 字段

    Returns:
        tuple[bytes, str] | None: (音频数据, 文件后缀) 或 None
    """
    if isinstance(value, (bytes, bytearray)) and value:
        return bytes(value), ".wav"
    if isinstance(value, str) and value:
        # 处理 data: URL 格式
        if value.startswith("data:"):
            header, _, payload = value.partition(",")
            if ";base64" in header and payload:
                suffix = ".mp3" if "mpeg" in header or "mp3" in header else ".wav"
                return base64.b64decode(payload), suffix
        # 尝试 base64 解码
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            decoded = b""
        if decoded:
            return decoded, ".wav"
    if isinstance(value, dict):
        for key in ("data", "audio", "content", "bytes", "base64", "audio_data"):
            if key in value:
                found = _find_audio_bytes(value[key])
                if found:
                    return found
        for child in value.values():
            found = _find_audio_bytes(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_audio_bytes(child)
            if found:
                return found
    return None


def _suffix_from_url(url: str, content_type: str = "") -> str:
    """从 URL 路径或 Content-Type 推断音频文件格式后缀。

    Args:
        url: 音频 URL
        content_type: HTTP Content-Type 头

    Returns:
        str: 文件后缀（如 .wav, .mp3）
    """
    path_suffix = Path(urlparse(url).path).suffix.lower()
    if path_suffix in {".wav", ".mp3", ".m4a", ".aac", ".pcm", ".ogg", ".flac"}:
        return path_suffix
    content_type = content_type.lower()
    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"
    if "wav" in content_type or "wave" in content_type:
        return ".wav"
    return ".audio"


def _download_audio(url: str) -> tuple[bytes, str]:
    """下载 DashScope TTS 生成的音频文件。

    Args:
        url: 音频下载 URL

    Returns:
        tuple[bytes, str]: (音频数据, 文件后缀)

    Raises:
        RuntimeError: 下载失败或返回空内容
    """
    response = requests.get(url, timeout=120)
    if not 200 <= response.status_code < 300:
        raise RuntimeError(f"DashScope TTS 音频下载失败 HTTP {response.status_code}: {response.text[:500]}")
    audio = response.content
    if not audio:
        raise RuntimeError("DashScope TTS 音频 URL 返回空内容")
    return audio, _suffix_from_url(url, response.headers.get("content-type", ""))


def _synthesize_with_dashscope(text: str) -> bytes:
    """使用阿里云 DashScope 多模态 TTS 进行语音合成。

    调用流程：
    1. 调用 TTS API 生成音频
    2. 下载音频 URL 或提取内嵌音频数据
    3. 通过 ffmpeg 转换为 16kHz 单声道 WAV

    Args:
        text: 要合成的文本

    Returns:
        bytes: 16kHz 单声道 WAV 音频数据

    Raises:
        RuntimeError: API Key 未配置、调用失败或无音频输出
    """
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置")

    import dashscope

    dashscope.api_key = api_key

    # 从环境变量读取 TTS 模型和音色配置
    model = os.getenv("TTS_MODEL", os.getenv("DASHSCOPE_TTS_MODEL", "qwen3-tts-flash")).strip()
    voice = os.getenv("TTS_VOICE", os.getenv("DASHSCOPE_TTS_VOICE", "Cherry")).strip()
    _log("provider=dashscope")
    _log(f"model={model}")
    _log(f"voice={voice}")

    # 调用 DashScope 多模态 TTS
    response = dashscope.MultiModalConversation.call(
        model=model,
        text=text,
        voice=voice,
    )
    response_data = _response_to_dict(response)
    status_code = response_data.get("status_code", getattr(response, "status_code", None))
    if status_code not in (None, 200):
        message = response_data.get("message", getattr(response, "message", ""))
        code = response_data.get("code", getattr(response, "code", ""))
        raise RuntimeError(f"DashScope TTS 失败 status={status_code} code={code} message={message}")

    # 优先尝试下载音频 URL
    audio_url = _find_audio_url(response_data)
    if audio_url:
        audio_bytes, suffix = _download_audio(audio_url)
    else:
        # 回退到内嵌音频数据
        found = _find_audio_bytes(response_data)
        if not found:
            raise RuntimeError(f"DashScope TTS 响应没有音频 URL 或内嵌数据: {response_data}")
        audio_bytes, suffix = found

    _log(f"接收音频字节数={len(audio_bytes)}")
    # 转换为标准 16kHz WAV
    wav_bytes = ensure_wav_16k_mono(audio_bytes, suffix)
    _log(f"最终 WAV 字节数={len(wav_bytes)}")
    return wav_bytes


def synthesize_wav_16k(text: str) -> bytes:
    """语音合成主入口，将文本转为 16kHz WAV 音频。

    根据 TTS_PROVIDER 环境变量选择合成引擎：
    - "mock": 生成正弦波模拟音
    - "dashscope": 调用阿里云 DashScope TTS

    Args:
        text: 要合成的文本

    Returns:
        bytes: 16kHz 单声道 WAV 音频数据

    Raises:
        ValueError: 不支持的 TTS_PROVIDER 值
    """
    provider = os.getenv("TTS_PROVIDER", "mock").strip().lower() or "mock"
    safe_text = text.strip() or ERROR_TEXT

    if provider == "mock":
        _log("使用 mock TTS，生成的模拟音无法被 ASR 识别语义")
        return _mock_tts_wav(safe_text)
    if provider == "dashscope":
        return _synthesize_with_dashscope(safe_text)

    raise ValueError(f"不支持的 TTS_PROVIDER: {provider}")


def synthesize_fallback_wav_16k(text: str = ERROR_TEXT) -> bytes:
    """降级 TTS 合成：当主 TTS 失败时的备用方案。

    使用 mock TTS 生成音频，如果也失败则返回 1 秒静音。

    Args:
        text: 要合成的文本，默认为错误提示

    Returns:
        bytes: WAV 音频数据
    """
    try:
        return _mock_tts_wav(text)
    except Exception as exc:
        _log(f"降级 mock TTS 失败: {exc}，返回 1 秒静音")
        return _silence_wav(1.0)
