#!/usr/bin/env python3
"""WTK1 对讲机后端服务 —— FastAPI HTTP + UDP 服务器。

功能概述：
- UDP WTK1 协议包日志记录和同设备音频回传（带服务端设备名）。
- FastAPI 分块 WAV 回传（用于 AI 语音测试）。
- FastAPI JPEG 上传接收（用于相机测试）。

AI 语音问答流程（/ai/* 接口）：
  1. /ai/start   — 创建会话
  2. /ai/upload  — 分块上传 WAV 音频
  3. /ai/finish  — 结束上传，触发 ASR → LLM → TTS 链路
  4. /ai/result_info  — 查询处理状态和结果信息
  5. /ai/result_chunk — 分块下载 TTS 合成的回复音频
  6. /ai/cancel  — 取消会话

相机拍照讲解流程（/camera/* 接口）：
  1. /camera/upload        — 上传 JPEG 图片，自动触发视觉分析和导游讲解
  2. /camera/analyze_latest — 对最新上传图片进行分析并返回导游讲解

UDP 协议：
  - 魔术字: WTK1
  - 包头 34 字节，设备名 16 字节
  - 支持 register/channel/ptt_start/audio/ptt_stop/heartbeat 包类型
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import struct
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

import core.config  # noqa: F401 - 加载项目 .env 环境变量
from core.paths import (
    DEFAULT_CAMERA_TEST_IMAGE,
    TMP_AUDIO_RECEIVED_WAV_DIR,
    TMP_AUDIO_REPLY_WAV_DIR,
    TMP_CAMERA_RECEIVED_DIR,
    ensure_runtime_dirs,
    env_path,
)
from services.bailian_app_service import BailianAppService
from server.camera_guide_debug import run_camera_guide_test
from services.photo_guide_service import PhotoGuideService, RETAKE_MODE, choose_mode, response_payload
from services.asr_service import transcribe_wav
from services.voice_qa_service import FIXED_ANSWER, VoiceQaService
from services.tts_service import ERROR_TEXT, synthesize_wav_16k
from services.vision_service import VisionObservation, VisionService

# =============================================================================
# WTK1 UDP 协议常量
# =============================================================================
MAGIC = b"WTK1"          # 协议魔术字
HEADER_LEN = 34          # 包头长度（字节）
DEVICE_LEN = 16          # 设备名字段长度（字节）
SERVER_DEVICE = b"server-echo"  # 服务端回传时使用的设备名

# =============================================================================
# 用户可配置的默认值
# =============================================================================
# 设备固件应将 APP_BUSINESS_SERVER_HOST 指向本机局域网 IP。
# APP_BUSINESS_UDP_PORT 应与 DEFAULT_UDP_PORT 一致。
# APP_BUSINESS_HTTP_BASE_URL 通常应为: http://<PC_LAN_IP>:<DEFAULT_HTTP_PORT>
DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_UDP_PORT = 9000
DEFAULT_HTTP_PORT = 8000
# 默认保存目录
DEFAULT_WAV_SAVE_DIR = TMP_AUDIO_RECEIVED_WAV_DIR
DEFAULT_JPG_SAVE_DIR = TMP_CAMERA_RECEIVED_DIR
# 默认分块大小（字节）
DEFAULT_CHUNK_SIZE = 32768
# AI 回复重复次数和额外数据块开关
DEFAULT_AI_REPLY_REPEAT = 1
DEFAULT_AI_REPLY_EXTRA_CHUNK = False

logger = logging.getLogger(__name__)

# WTK1 包类型映射
PKT_TYPES = {
    1: "register",
    2: "channel",
    3: "ptt_start",
    4: "audio",
    5: "ptt_stop",
    6: "heartbeat",
}


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class Packet:
    """WTK1 协议包。"""
    packet_type: int      # 包类型
    channel: int          # 频道号
    seq: int              # 序列号
    timestamp_ms: int     # 时间戳（毫秒）
    device: str           # 设备名
    payload: bytes        # 负载数据


@dataclass
class WavInfo:
    """WAV 音频文件信息。"""
    audio_format: int     # 音频格式（1=PCM）
    channels: int         # 声道数
    sample_rate: int      # 采样率（Hz）
    bits_per_sample: int  # 位深
    data_offset: int      # 音频数据起始偏移
    data_size: int        # 音频数据大小（字节）


@dataclass
class JpegInfo:
    """JPEG 图片文件信息。"""
    width: int | None = None       # 图片宽度（像素）
    height: int | None = None      # 图片高度（像素）
    progressive: bool = False      # 是否渐进式 JPEG


@dataclass
class AiSession:
    """AI 语音问答会话状态。

    跟踪一次 ASR → LLM → TTS 全链路中各个环节的状态。
    """
    session_id: str                           # 会话 ID
    chunks: bytearray | None = None           # 接收缓冲
    total: int = 0                            # 总字节数
    received: int = 0                         # 已接收字节数
    reply: bytes | None = None                # TTS 合成的回复音频
    save_path: Path | None = None             # 上传 WAV 保存路径
    device: str = "walkie-01"                 # 设备标识
    language: str = "zh"                      # 语言代码
    question_text: str = ""                   # 用户问题文本
    answer_text: str = ""                     # AI 回答文本
    asr_text: str = ""                        # ASR 识别文本
    image_context: str = ""                   # 图片上下文信息
    upload_wav_path: Path | None = None       # 上传 WAV 路径
    reply_path: Path | None = None            # 回复音频路径
    status: str = "started"                   # 会话状态
    audio_ready: bool = False                 # 音频是否就绪
    reply_wav_ready: bool = False             # 回复 WAV 是否就绪
    reply_wav_size: int = 0                   # 回复 WAV 大小
    reply_duration: float = 0.0               # 回复音频时长（秒）
    tts_status: str = "idle"                  # TTS 状态
    tts_error: str | None = None              # TTS 错误信息
    tts_task: asyncio.Task | None = None      # 后台 TTS 异步任务
    canceled: bool = False                    # 是否已取消


# =============================================================================
# 工具函数
# =============================================================================

def log(message: str) -> None:
    """带时间戳的日志输出。"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def auto_tts_background_enabled() -> bool:
    """检查是否启用了后台自动 TTS 合成。

    通过环境变量 AUTO_TTS_BACKGROUND 控制，默认启用。
    """
    value = os.getenv("AUTO_TTS_BACKGROUND", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


# =============================================================================
# 二进制数据解析函数
# =============================================================================

def read_u16(data: bytes, offset: int) -> int:
    """从字节数组中读取小端 16 位无符号整数。"""
    return data[offset] | (data[offset + 1] << 8)


def read_u32(data: bytes, offset: int) -> int:
    """从字节数组中读取小端 32 位无符号整数。"""
    return (
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24)
    )


def parse_packet(data: bytes) -> Packet | None:
    """解析 WTK1 协议包。

    Returns:
        Packet | None: 解析成功返回 Packet 对象，否则返回 None
    """
    if len(data) < HEADER_LEN or data[:4] != MAGIC:
        return None

    header_len = data[5]
    payload_len = read_u16(data, 32)
    if header_len != HEADER_LEN or len(data) < header_len + payload_len:
        return None

    # 提取设备名（以 \x00 结尾的字符串）
    device_raw = data[16:32].split(b"\x00", 1)[0]
    return Packet(
        packet_type=data[4],
        channel=read_u16(data, 6),
        seq=read_u32(data, 8),
        timestamp_ms=read_u32(data, 12),
        device=device_raw.decode("utf-8", errors="replace"),
        payload=data[header_len : header_len + payload_len],
    )


def make_server_echo(data: bytes) -> bytes:
    """构造服务端回传包：将原始包的设备名替换为服务端标识。"""
    out = bytearray(data)
    out[16:32] = b"\x00" * DEVICE_LEN
    out[16 : 16 + len(SERVER_DEVICE)] = SERVER_DEVICE
    return bytes(out)


# =============================================================================
# WAV 音频处理
# =============================================================================

def parse_wav(body: bytes) -> WavInfo | None:
    """解析 WAV 文件的 RIFF 头，提取音频参数。

    支持标准 PCM WAV 格式，能跳过 fmt 和数据块之间的非标准块。

    Returns:
        WavInfo | None: 解析成功返回 WavInfo，否则返回 None
    """
    if len(body) < 44 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        return None

    pos = 12
    audio_format = channels = sample_rate = bits_per_sample = None
    data_offset = data_size = None

    # 遍历 RIFF 块查找 fmt 和 data 块
    while pos + 8 <= len(body):
        chunk_id = body[pos : pos + 4]
        chunk_size = read_u32(body, pos + 4)
        chunk_data = pos + 8
        chunk_end = chunk_data + chunk_size
        if chunk_end > len(body):
            return None

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                return None
            audio_format, channels, sample_rate, _byte_rate, _block_align, bits_per_sample = struct.unpack_from(
                "<HHIIHH", body, chunk_data
            )
        elif chunk_id == b"data":
            data_offset = chunk_data
            data_size = chunk_size
            break

        pos = chunk_end + (chunk_size & 1)  # 对齐到偶数边界

    if (
        audio_format is None
        or channels is None
        or sample_rate is None
        or bits_per_sample is None
        or data_offset is None
        or data_size is None
    ):
        return None

    return WavInfo(
        audio_format=audio_format,
        channels=channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        data_offset=data_offset,
        data_size=data_size,
    )


def pcm16_stats(pcm: bytes) -> str:
    """计算 16-bit PCM 音频数据的统计信息。

    包括采样数、最小/最大值、均值、RMS、峰值、削波数和过零率。
    用于日志记录和音频质量诊断。
    """
    sample_count = len(pcm) // 2
    if sample_count == 0:
        return "samples=0"

    samples = struct.unpack_from(f"<{sample_count}h", pcm[: sample_count * 2])
    min_v = min(samples)
    max_v = max(samples)
    mean = sum(samples) / sample_count
    rms = math.sqrt(sum(s * s for s in samples) / sample_count)
    peak = max(abs(min_v), abs(max_v))
    # 统计削波样本数（接近 16-bit 极限值）
    clipped = sum(1 for s in samples if s <= -32760 or s >= 32760)
    # 计算过零率（反映频率特性）
    zero_cross = sum(
        1
        for prev, cur in zip(samples, samples[1:])
        if (prev < 0 <= cur) or (prev > 0 >= cur)
    )
    zcr = zero_cross / max(sample_count - 1, 1)

    return (
        f"samples={sample_count} min={min_v} max={max_v} "
        f"mean={mean:.1f} rms={rms:.1f} peak={peak} "
        f"clipped={clipped} zcr={zcr:.3f}"
    )


def save_wav(body: bytes, save_dir: Path) -> Path:
    """保存 WAV 文件到指定目录，文件名带时间戳。

    Returns:
        Path: 保存的文件路径
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"ai_upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
    path.write_bytes(body)
    return path


def validate_and_log_wav(body: bytes, save_dir: Path, prefix: str) -> tuple[bool, Path | None]:
    """验证 WAV 格式并记录详细日志。

    Returns:
        tuple[bool, Path | None]: (是否有效, 保存路径)
    """
    wav = parse_wav(body)
    if wav is None:
        log(f"{prefix} 无效 WAV len={len(body)}")
        return False, None

    pcm = body[wav.data_offset : wav.data_offset + wav.data_size]
    duration = 0.0
    if wav.sample_rate > 0 and wav.channels > 0 and wav.bits_per_sample > 0:
        bytes_per_sample = wav.channels * wav.bits_per_sample // 8
        if bytes_per_sample > 0:
            duration = wav.data_size / bytes_per_sample / wav.sample_rate

    save_path = save_wav(body, save_dir)
    stats = pcm16_stats(pcm) if wav.audio_format == 1 and wav.bits_per_sample == 16 else "pcm_stats=unsupported"
    log(
        f"{prefix} WAV fmt={wav.audio_format} ch={wav.channels} rate={wav.sample_rate} "
        f"bits={wav.bits_per_sample} data={wav.data_size} duration={duration:.2f}s "
        f"{stats} saved={save_path}"
    )
    return True, save_path


def build_pcm_wav(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
    add_extra_chunk: bool,
) -> bytes:
    """将原始 PCM 数据封装为标准 WAV 文件。

    Args:
        pcm: 原始 PCM 音频数据
        sample_rate: 采样率（Hz）
        channels: 声道数
        bits_per_sample: 位深
        add_extra_chunk: 是否在 fmt 和 data 之间插入 JUNK 块
                         用于测试设备是否假定固定的 44 字节 WAV 头

    Returns:
        bytes: 完整的 WAV 文件数据
    """
    if bits_per_sample % 8 != 0:
        raise ValueError("bits_per_sample 必须是 8 的倍数")

    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    # 构建 fmt 块
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,              # PCM 格式
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    chunks = [fmt_chunk]

    if add_extra_chunk:
        # 插入 JUNK 块，强制设备从非标准偏移查找数据块
        junk_payload = b"stream-test-extra"
        chunks.append(struct.pack("<4sI", b"JUNK", len(junk_payload)) + junk_payload)
        if len(junk_payload) & 1:
            chunks.append(b"\x00")  # RIFF 块对齐

    # 构建 data 块
    chunks.append(struct.pack("<4sI", b"data", len(pcm)) + pcm)
    body = b"".join(chunks)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def make_ai_reply_wav(upload_wav: bytes, repeat: int, add_extra_chunk: bool) -> bytes | None:
    """根据上传的 WAV 生成 AI 测试回复 WAV。

    提取上传 WAV 的 PCM 数据，可选重复多次后重新封装。
    用于设备端的声学回环测试。

    Args:
        upload_wav: 上传的 WAV 数据
        repeat: PCM 数据重复次数
        add_extra_chunk: 是否添加 JUNK 块

    Returns:
        bytes | None: 回复 WAV 数据，格式不兼容时返回 None
    """
    wav = parse_wav(upload_wav)
    if wav is None:
        return None
    if wav.audio_format != 1:  # 仅支持 PCM
        return None
    pcm = upload_wav[wav.data_offset : wav.data_offset + wav.data_size]
    if repeat > 1:
        pcm = pcm * repeat
    return build_pcm_wav(
        pcm,
        sample_rate=wav.sample_rate,
        channels=wav.channels,
        bits_per_sample=wav.bits_per_sample,
        add_extra_chunk=add_extra_chunk,
    )


# =============================================================================
# JPEG 图片处理
# =============================================================================

def parse_jpeg(body: bytes) -> JpegInfo | None:
    """解析 JPEG 文件，提取尺寸和编码类型。

    遍历 JPEG 标记段，从 SOF0/SOF1/SOF2 标记中读取宽高。

    Returns:
        JpegInfo | None: 解析成功返回 JpegInfo，否则返回 None
    """
    # 检查 JPEG SOI 和 EOI 标记
    if len(body) < 4 or body[:2] != b"\xFF\xD8" or body[-2:] != b"\xFF\xD9":
        return None

    pos = 2
    while pos + 4 <= len(body):
        if body[pos] != 0xFF:
            pos += 1
            continue
        # 跳过填充字节
        while pos < len(body) and body[pos] == 0xFF:
            pos += 1
        if pos >= len(body):
            break

        marker = body[pos]
        pos += 1
        # 跳过独立标记和 RST 标记
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > len(body):
            return None

        segment_len = read_u16_be(body, pos)
        if segment_len < 2 or pos + segment_len > len(body):
            return None

        # SOF0（基线）、SOF1（扩展）、SOF2（渐进）
        if marker in (0xC0, 0xC1, 0xC2):
            if segment_len < 7:
                return None
            height = read_u16_be(body, pos + 3)
            width = read_u16_be(body, pos + 5)
            return JpegInfo(width=width, height=height, progressive=(marker == 0xC2))

        pos += segment_len

    return JpegInfo()


def read_u16_be(data: bytes, offset: int) -> int:
    """从字节数组中读取大端 16 位无符号整数。"""
    return (data[offset] << 8) | data[offset + 1]


def save_jpeg(body: bytes, save_dir: Path) -> Path:
    """保存 JPEG 文件到指定目录，文件名带时间戳。

    Returns:
        Path: 保存的文件路径
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"camera_upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path.write_bytes(body)
    return path


def save_camera_raw(body: bytes, save_dir: Path) -> Path:
    """保存无效的 JPEG 原始数据（用于调试）。

    Returns:
        Path: 保存的文件路径
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"camera_upload_invalid_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.bin"
    path.write_bytes(body)
    return path


def validate_and_log_jpeg(body: bytes, save_dir: Path, prefix: str) -> tuple[bool, Path | None, JpegInfo | None]:
    """验证 JPEG 格式并记录详细日志。

    Returns:
        tuple: (是否有效, 保存路径, JPEG 信息)
    """
    jpeg = parse_jpeg(body)
    if jpeg is None:
        save_path = save_camera_raw(body, save_dir)
        log(
            f"{prefix} 无效 JPEG len={len(body)} "
            f"soi={body[:2].hex()} eoi={body[-2:].hex() if len(body) >= 2 else ''} "
            f"saved_raw={save_path}"
        )
        return False, save_path, None

    save_path = save_jpeg(body, save_dir)
    size_text = f"{jpeg.width}x{jpeg.height}" if jpeg.width and jpeg.height else "unknown"
    log(
        f"{prefix} JPEG len={len(body)} size={size_text} "
        f"progressive={int(jpeg.progressive)} saved={save_path}"
    )
    return True, save_path, jpeg


# =============================================================================
# UDP 服务器
# =============================================================================

def run_udp(host: str, port: int) -> None:
    """运行 UDP WTK1 协议服务器。

    功能：
    - 接收并记录所有 WTK1 协议包
    - 转发音频包给同一频道的其他设备
    - 单设备场景下回传服务端回显包
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        log(f"UDP 绑定失败 {host}:{port}: {exc}")
        return
    log(f"UDP WTK1 监听 {host}:{port}")

    # 设备注册表: device -> (ip, port, channel)
    devices: dict[str, tuple[str, int, int]] = {}

    while True:
        data, addr = sock.recvfrom(2048)
        packet = parse_packet(data)
        if packet is None:
            log(f"UDP 原始数据 from {addr[0]}:{addr[1]} len={len(data)} data={data!r}")
            continue

        type_name = PKT_TYPES.get(packet.packet_type, f"type_{packet.packet_type}")
        devices[packet.device] = (addr[0], addr[1], packet.channel)
        log(
            f"UDP {type_name} from {packet.device}@{addr[0]}:{addr[1]} "
            f"ch={packet.channel} seq={packet.seq} payload={len(packet.payload)}"
        )

        # 音频包转发逻辑
        if packet.packet_type == 4 and packet.payload:
            # 查找同一频道的其他设备
            targets = [
                (dev, dev_addr)
                for dev, (ip, port, channel) in devices.items()
                if dev != packet.device and channel == packet.channel
                for dev_addr in [(ip, port)]
            ]
            if targets:
                # 多设备：转发给同一频道的其他设备
                for dev, dev_addr in targets:
                    sock.sendto(data, dev_addr)
                    log(f"UDP 音频转发至 {dev}@{dev_addr[0]}:{dev_addr[1]}")
            else:
                # 单设备测试：回传服务端回显包
                # 需要将 device 字段改为非本地设备名，否则客户端会丢弃
                sock.sendto(make_server_echo(data), addr)


# =============================================================================
# FastAPI HTTP 应用
# =============================================================================

def create_http_app(
    wav_save_dir: Path,
    jpg_save_dir: Path,
    ai_reply_repeat: int,
    ai_reply_extra_chunk: bool,
) -> FastAPI:
    """创建 FastAPI 应用实例。

    配置所有 AI 语音问答和相机拍照讲解相关的路由，
    初始化各类服务实例（ASR/TTS/Vision/Bailian 等）。

    Args:
        wav_save_dir: WAV 音频保存目录
        jpg_save_dir: JPEG 图片保存目录
        ai_reply_repeat: AI 测试回复中 PCM 重复次数
        ai_reply_extra_chunk: 是否在回复 WAV 中插入 JUNK 块

    Returns:
        FastAPI: 配置好的 FastAPI 应用实例
    """
    ensure_runtime_dirs()
    app = FastAPI(title="WTK1 Backend")

    # =========================================================================
    # 应用状态初始化
    # =========================================================================
    app.state.save_dir = wav_save_dir
    app.state.jpg_save_dir = jpg_save_dir
    # AI 会话存储（线程安全）
    app.state.ai_sessions = {}
    app.state.ai_sessions_lock = threading.RLock()
    # 最近相机图片缓存
    app.state.latest_images = {}
    app.state.latest_image_analysis = {}
    # AI 回复参数
    app.state.ai_reply_repeat = max(ai_reply_repeat, 1)
    app.state.ai_reply_extra_chunk = ai_reply_extra_chunk
    # 回复 WAV 保存目录
    app.state.reply_save_dir = env_path("REPLY_WAV_SAVE_DIR", TMP_AUDIO_REPLY_WAV_DIR)
    app.state.latest_reply_dir = env_path("LATEST_TMP_DIR", app.state.reply_save_dir)

    # 初始化各服务实例
    bailian_app_service = BailianAppService()
    app.state.bailian_app_service = bailian_app_service
    app.state.voice_qa_service = VoiceQaService(bailian_app_service)
    app.state.vision_service = VisionService()
    app.state.photo_guide_service = PhotoGuideService(bailian_app_service)

    # =========================================================================
    # 会话管理辅助函数
    # =========================================================================

    def get_session(session_id: str) -> AiSession:
        """获取会话，不存在时抛出 404。"""
        with app.state.ai_sessions_lock:
            session = app.state.ai_sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail={"ok": False, "error": "unknown session"})
            return session

    def is_session_canceled(ai_session: AiSession) -> bool:
        """检查会话是否已取消。"""
        return ai_session.canceled or ai_session.status == "canceled"

    def mark_session_canceled(ai_session: AiSession) -> None:
        """标记会话为已取消状态，清空音频相关标志。"""
        ai_session.canceled = True
        ai_session.status = "canceled"
        ai_session.audio_ready = False
        ai_session.reply_wav_ready = False
        ai_session.reply_wav_size = 0
        ai_session.reply_duration = 0.0
        ai_session.tts_status = "canceled"
        ai_session.tts_error = None

    def canceled_result_info(session_id: str, ai_session: AiSession) -> dict[str, object]:
        """返回已取消会话的 result_info 响应。"""
        return {
            "ok": True,
            "session": session_id,
            "ready": False,
            "total": 0,
            "format": "wav",
            "text": ai_session.answer_text,
            "status": "canceled",
            "asr_text": ai_session.asr_text,
            "answer_text": ai_session.answer_text,
            "audio_ready": False,
            "reply_wav_ready": False,
            "reply_wav_size": 0,
            "reply_duration": 0,
            "tts_status": "canceled",
            "tts_error": None,
        }

    def canceled_response(session_id: str) -> dict[str, object]:
        """返回取消操作的响应。"""
        return {
            "ok": True,
            "session": session_id,
            "status": "canceled",
            "message": "session canceled",
        }

    def reply_duration_seconds(reply: bytes) -> float:
        """计算 WAV 回复音频的时长（秒）。"""
        wav = parse_wav(reply)
        if wav is None:
            return 0.0
        bytes_per_sample = wav.channels * wav.bits_per_sample // 8
        if bytes_per_sample <= 0 or wav.sample_rate <= 0:
            return 0.0
        return wav.data_size / bytes_per_sample / wav.sample_rate

    # =========================================================================
    # 后台 TTS 合成
    # =========================================================================

    async def generate_tts_background(session_id: str, answer_text: str) -> None:
        """在后台异步合成 TTS 音频。

        在 /ai/finish 返回后异步运行，不阻塞 HTTP 响应。
        合成完成后将结果写入会话状态，客户端通过轮询 /ai/result_info 获取。

        Args:
            session_id: 会话 ID
            answer_text: 要合成的文本
        """
        tts_start = time.perf_counter()
        # 获取会话并检查状态
        with app.state.ai_sessions_lock:
            ai_session = app.state.ai_sessions.get(session_id)
            if ai_session is None:
                log(f"[TTS-BG] 会话不存在 session={session_id}")
                return
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"TTS 跳过 因会话已取消 session={session_id}")
                return
            if ai_session.audio_ready or ai_session.reply_wav_ready:
                log(f"[TTS-BG] 跳过 session={session_id} reason=音频已就绪")
                return
            if ai_session.tts_status == "running":
                log(f"[TTS-BG] 跳过 session={session_id} reason=已在运行中")
                return

            ai_session.tts_status = "running"
            ai_session.tts_error = None
        log(f"[TTS-BG] 开始 session={session_id}")

        try:
            # 调用 TTS 合成
            reply = await asyncio.to_thread(synthesize_wav_16k, answer_text)
            with app.state.ai_sessions_lock:
                if is_session_canceled(ai_session):
                    mark_session_canceled(ai_session)
                    log(f"TTS 结果因会话已取消而被忽略 session={session_id}")
                    return
            if parse_wav(reply) is None:
                raise RuntimeError("TTS 生成的回复 WAV 无效")

            # 保存回复音频文件
            write_start = time.perf_counter()
            app.state.reply_save_dir.mkdir(parents=True, exist_ok=True)
            reply_path = app.state.reply_save_dir / f"reply_{session_id}.wav"
            reply_path.write_bytes(reply)
            app.state.latest_reply_dir.mkdir(parents=True, exist_ok=True)
            (app.state.latest_reply_dir / "latest_reply.wav").write_bytes(reply)
            log(f"[AI-TIME] write_reply={time.perf_counter() - write_start:.3f}s reply_wav={reply_path}")

            # 更新会话状态
            with app.state.ai_sessions_lock:
                if is_session_canceled(ai_session):
                    mark_session_canceled(ai_session)
                    log(f"TTS 结果因会话已取消而被忽略 session={session_id}")
                    return
                ai_session.reply = reply
                ai_session.reply_path = reply_path
                ai_session.reply_wav_size = reply_path.stat().st_size
                ai_session.reply_duration = reply_duration_seconds(reply)
                ai_session.audio_ready = True
                ai_session.reply_wav_ready = True
                ai_session.tts_status = "done"
                ai_session.status = "audio_ready"
            cost = time.perf_counter() - tts_start
            log(f"[TTS-BG] 完成 session={session_id} wav={reply_path} cost={cost:.3f}s")
            log(f"[AI-TIME] tts_background={cost:.3f}s")
        except Exception as exc:
            logger.exception("[TTS-BG] 失败 session=%s", session_id)
            with app.state.ai_sessions_lock:
                if is_session_canceled(ai_session):
                    mark_session_canceled(ai_session)
                    log(f"TTS 结果因会话已取消而被忽略 session={session_id}")
                    return
                ai_session.status = "audio_failed"
                ai_session.audio_ready = False
                ai_session.reply_wav_ready = False
                ai_session.tts_status = "failed"
                ai_session.tts_error = str(exc)[:300]
            cost = time.perf_counter() - tts_start
            log(f"[TTS-BG] 失败 session={session_id} error={ai_session.tts_error}")
            log(f"[AI-TIME] tts_background={cost:.3f}s error={ai_session.tts_error}")

    def maybe_start_tts_background(ai_session: AiSession) -> None:
        """条件触发后台 TTS 合成。

        仅在以下条件全部满足时启动 TTS：
        - 会话未取消
        - 有回答文本
        - 启用了自动 TTS
        - 无正在运行的 TTS 任务
        - 音频尚未就绪
        """
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"TTS 跳过 因会话已取消 session={ai_session.session_id}")
                return
        if not ai_session.answer_text.strip():
            ai_session.tts_status = "disabled"
            return
        if not auto_tts_background_enabled():
            ai_session.tts_status = "disabled"
            return
        if ai_session.tts_task is not None and not ai_session.tts_task.done():
            return
        if ai_session.audio_ready or ai_session.reply_wav_ready:
            return
        ai_session.tts_status = "pending"
        ai_session.tts_task = asyncio.create_task(
            generate_tts_background(ai_session.session_id, ai_session.answer_text)
        )

    # =========================================================================
    # ASR + LLM 文本处理（支持中途取消）
    # =========================================================================

    async def process_text_with_cancel(
        ai_session: AiSession,
        wav_path: Path,
        *,
        spot_id: str,
        image_context: str,
        mode: str,
    ) -> tuple[str, str]:
        """处理语音识别和 LLM 问答，支持在每个步骤前检查取消状态。

        处理流程：
        1. 检查模式：fixed 直接返回固定回答
        2. ASR 语音识别 → 检查取消
        3. 判断是否为"最新图片"类问题 → 是则走图片导游路径
        4. LLM 问答 → 检查取消

        Args:
            ai_session: AI 会话对象
            wav_path: WAV 文件路径
            spot_id: 景点 ID
            image_context: 图片上下文
            mode: 处理模式

        Returns:
            tuple[str, str]: (ASR 文本, 回答文本)
        """
        # 固定回答模式
        if mode == "fixed":
            with app.state.ai_sessions_lock:
                if is_session_canceled(ai_session):
                    log(f"LLM 跳过 因会话已取消 session={ai_session.session_id}")
                    return "", ""
            return "", FIXED_ANSWER

        if mode != "asr_bailian_app":
            raise ValueError(f"不支持的 TOUR_MODE: {mode}")

        # 检查取消状态
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                log(f"ASR 跳过 因会话已取消 session={ai_session.session_id}")
                return "", ""

        # 第 1 步：ASR 语音识别
        asr_start = time.perf_counter()
        try:
            asr_text = await asyncio.to_thread(transcribe_wav, wav_path)
        except Exception as exc:
            print(f"[AI-TIME] asr={time.perf_counter() - asr_start:.3f}s error={exc}", flush=True)
            raise RuntimeError(f"ASR 失败: {exc}") from exc
        print(f"[AI] asr_text: {asr_text}", flush=True)
        print(f"[AI-TIME] asr={time.perf_counter() - asr_start:.3f}s text_chars={len(asr_text)}", flush=True)

        # 检查取消状态
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                log(f"ASR 结果因会话已取消而被忽略 session={ai_session.session_id}")
                return "", ""

        # 判断是否为"最新图片"相关提问
        if is_latest_image_question(asr_text):
            with app.state.ai_sessions_lock:
                if is_session_canceled(ai_session):
                    log(f"图片回答跳过 因会话已取消 session={ai_session.session_id}")
                    return "", ""
            answer_text = await answer_latest_image_question(ai_session.device)
            with app.state.ai_sessions_lock:
                if is_session_canceled(ai_session):
                    log(f"图片回答结果因会话已取消而被忽略 session={ai_session.session_id}")
                    return asr_text, ""
            return asr_text, answer_text

        # 检查取消状态
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                log(f"LLM 跳过 因会话已取消 session={ai_session.session_id}")
                return "", ""

        # 第 2 步：LLM 问答
        try:
            answer_text = await asyncio.to_thread(
                app.state.voice_qa_service._ask_llm,
                asr_text,
                device=ai_session.device,
                spot_id=spot_id,
                image_context=image_context,
            )
        except Exception as exc:
            raise RuntimeError(f"百炼应用调用失败: {exc}") from exc
        print(f"[AI] answer_text chars: {len(answer_text)}", flush=True)

        # 检查取消状态
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                log(f"LLM 结果因会话已取消而被忽略 session={ai_session.session_id}")
                return asr_text, ""

        return asr_text, answer_text

    # =========================================================================
    # HTTP 请求日志
    # =========================================================================

    async def log_request(request: Request, body: bytes, op: str) -> None:
        """记录 HTTP 请求日志。"""
        content_type = request.headers.get("content-type", "")
        log(f"HTTP POST {request.url.path}?{request.url.query} len={len(body)} content_type={content_type!r} route_op={op!r}")

    # =========================================================================
    # 调试接口：相机导游端到端测试
    # =========================================================================

    @app.get("/debug/camera_guide/test")
    async def debug_camera_guide_test() -> JSONResponse:
        """端到端相机导游调试接口。

        使用默认测试图片运行完整的视觉分析 → 知识库检索链路。
        """
        result = await run_camera_guide_test(
            vision_service=app.state.vision_service,
            bailian_app_service=app.state.bailian_app_service,
            test_image_path=DEFAULT_CAMERA_TEST_IMAGE,
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 500)

    # =========================================================================
    # 相机视觉分析
    # =========================================================================

    async def analyze_camera_observation(safe_device: str, image_id: str, image_path: Path) -> VisionObservation:
        """分析相机上传的图片并缓存结果。

        Args:
            safe_device: 设备标识
            image_id: 图片 ID
            image_path: 图片路径

        Returns:
            VisionObservation: 视觉观察结果
        """
        vision_start = time.perf_counter()
        try:
            observation = await asyncio.to_thread(app.state.vision_service.analyze_image, image_path)
            status = "retake" if choose_mode(observation) == RETAKE_MODE else "ready"
            error = ""
        except Exception as exc:
            logger.exception("[CAMERA] 视觉识别失败 device=%s image_id=%s", safe_device, image_id)
            observation = VisionObservation(reason=f"视觉识别异常：{exc}")
            status = "failed"
            error = str(exc)[:300]

        # 缓存分析结果
        app.state.latest_image_analysis[safe_device] = {
            "image_id": image_id,
            "path": image_path,
            "time": datetime.now(),
            "status": status,
            "observation": observation,
            "error": error,
        }
        log(
            f"[CAMERA] 视觉识别 image_id={image_id} status={status} "
            f"best_candidate_id={observation.best_candidate_id} "
            f"candidate_confidence={observation.candidate_confidence:.2f} "
            f"category={observation.category} safe_answer_level={observation.safe_answer_level} "
            f"retake={int(observation.need_retake)} selected_mode={choose_mode(observation)} "
            f"cost={time.perf_counter() - vision_start:.3f}s"
        )
        return observation

    def is_latest_image_question(text: str) -> bool:
        """判断用户问题是否在询问最新拍摄的图片。

        通过关键词匹配判断，如"照片"、"图片"、"刚拍"、"这是什么"等。

        Args:
            text: 用户问题文本

        Returns:
            bool: 是否在询问图片
        """
        normalized = (text or "").strip()
        if not normalized:
            return False
        keywords = (
            "照片", "图片", "拍的", "刚拍",
            "这个是什么", "这是什么",
            "这个展品", "这件展品",
            "这个文物", "这件文物",
            "讲讲这个", "看看这个", "识别一下",
        )
        return any(keyword in normalized for keyword in keywords)

    async def answer_latest_image_question(safe_device: str) -> str:
        """回答关于最新图片的问题。

        优先使用当前设备的分析缓存，fallback 到 walkie-01 的缓存。

        Args:
            safe_device: 设备标识

        Returns:
            str: 导游讲解文本
        """
        cached = app.state.latest_image_analysis.get(safe_device)
        # fallback：尝试使用默认设备的缓存
        if cached is None and safe_device != "walkie-01":
            cached = app.state.latest_image_analysis.get("walkie-01")
        if not isinstance(cached, dict):
            return "我还没有收到可以讲解的照片。你可以先拍一张展品，尽量让展品居中，再来问我。"

        observation = cached.get("observation")
        image_id = str(cached.get("image_id") or "")
        if not isinstance(observation, VisionObservation):
            return "这张照片信息不太够。请把展品放在画面中间，靠近一点，避开展柜反光后重拍。"

        guide = await asyncio.to_thread(
            app.state.photo_guide_service.build_answer,
            observation,
            device=safe_device,
            image_id=image_id,
        )
        log(
            f"[CAMERA] 语音使用缓存图片 device={safe_device} image_id={image_id} "
            f"mode={guide.mode} grounded={int(guide.grounded)} answer_chars={len(guide.answer_text)}"
        )
        return guide.answer_text

    # =========================================================================
    # AI 语音问答接口
    # =========================================================================

    @app.post("/ai/start")
    async def ai_start(request: Request) -> dict[str, object]:
        """创建 AI 语音问答会话。

        可选 JSON body 参数：
        - device: 设备标识（默认 walkie-01）
        - language: 语言代码（默认 zh）
        """
        body = await request.body()
        await log_request(request, body, "start")
        body_json: dict[str, object] = {}
        if body:
            try:
                parsed = json.loads(body.decode("utf-8"))
                if isinstance(parsed, dict):
                    body_json = parsed
                else:
                    log("AI start JSON body 不是对象，使用默认值")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                log(f"AI start JSON 解析失败: {exc}，使用默认值")
        device = str(body_json.get("device") or "walkie-01")
        language = str(body_json.get("language") or "zh")
        session_id = uuid.uuid4().hex[:12]
        with app.state.ai_sessions_lock:
            app.state.ai_sessions[session_id] = AiSession(
                session_id=session_id,
                device=device,
                language=language,
            )
        log(f"AI start session={session_id} device={device} language={language}")
        return {"session": session_id, "chunk_size": DEFAULT_CHUNK_SIZE}

    @app.post("/ai/cancel")
    async def ai_cancel(request: Request, session: str = Query(...)) -> dict[str, object]:
        """取消 AI 会话。

        取消后：
        - 后续上传将被拒绝（返回 409）
        - finish 将被忽略
        - result_chunk 将被拒绝
        - 已完成的文本内容会被保留在 result_info 中
        """
        body = await request.body()
        await log_request(request, body, "cancel")
        log(f"取消请求 session={session}")
        with app.state.ai_sessions_lock:
            ai_session = app.state.ai_sessions.get(session)
            if ai_session is None:
                log(f"取消未知会话 session={session}")
                return {
                    "ok": False,
                    "session": session,
                    "status": "not_found",
                    "error": "session not found",
                }
            mark_session_canceled(ai_session)
        log(f"取消已接受 session={session}")
        return canceled_response(session)

    @app.post("/ai/upload")
    async def ai_upload(
        request: Request,
        session: str = Query(...),
        index: int = Query(0),
        offset: int = Query(0),
        total: int = Query(0),
    ) -> dict[str, bool]:
        """分块上传 WAV 音频数据。

        参数：
        - session: 会话 ID
        - index: 块序号
        - offset: 数据在完整文件中的偏移
        - total: 完整文件的预期大小
        """
        body = await request.body()
        await log_request(request, body, "upload")
        ai_session = get_session(session)

        # 检查是否已取消
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"上传被拒绝 因会话已取消 session={session}")
                raise HTTPException(
                    status_code=409,
                    detail={"ok": False, "status": "canceled", "error": "session canceled"},
                )
        # 校验上传参数
        if total <= 0 or offset < 0 or offset + len(body) > total:
            raise HTTPException(status_code=400, detail={"ok": False, "error": "upload range invalid"})

        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"上传被拒绝 因会话已取消 session={session}")
                raise HTTPException(
                    status_code=409,
                    detail={"ok": False, "status": "canceled", "error": "session canceled"},
                )
            # 初始化接收缓冲区
            if ai_session.chunks is None:
                ai_session.status = "uploading"
                ai_session.total = total
                ai_session.chunks = bytearray(total)
            # 校验 total 未变化
            if total != ai_session.total or ai_session.chunks is None:
                raise HTTPException(status_code=409, detail={"ok": False, "error": "total changed"})
            # 写入数据
            ai_session.chunks[offset : offset + len(body)] = body
            ai_session.received += len(body)
        log(
            f"AI upload session={session} index={index} offset={offset} "
            f"len={len(body)} received={ai_session.received}/{ai_session.total}"
        )
        return {"ok": True}

    @app.post("/ai/finish")
    async def ai_finish(request: Request, session: str = Query(...)) -> dict[str, object]:
        """结束 WAV 上传并触发 ASR → LLM → TTS 全链路处理。

        处理完成后立即返回，TTS 合成在后台异步进行。
        客户端应通过 /ai/result_info 轮询获取 TTS 结果。
        """
        total_start = time.perf_counter()
        body = await request.body()
        await log_request(request, body, "finish")
        ai_session = get_session(session)

        # 检查取消状态
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"finish 被忽略 因会话已取消 session={session}")
                return {"ok": True, "status": "canceled"}
            if ai_session.chunks is None or ai_session.total <= 0:
                raise HTTPException(status_code=400, detail={"ok": False, "error": "no upload"})
            if ai_session.received < ai_session.total:
                raise HTTPException(status_code=409, detail={"ok": False, "error": "upload incomplete"})
            ai_session.status = "processing"
            full_wav = bytes(ai_session.chunks)

        # 检查取消状态
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"finish 被忽略 因会话已取消 session={session}")
                return {"ok": True, "status": "canceled"}

        # 验证并保存 WAV
        save_start = time.perf_counter()
        try:
            ok, save_path = validate_and_log_wav(full_wav, app.state.save_dir, f"AI finish session={session}")
        except Exception as exc:
            log(f"[AI-TIME] save_upload={time.perf_counter() - save_start:.3f}s error={exc}")
            log(f"[AI-TIME] total={time.perf_counter() - total_start:.3f}s error={exc}")
            raise
        log(f"[AI-TIME] save_upload={time.perf_counter() - save_start:.3f}s")
        if not ok:
            log(f"[AI-TIME] total={time.perf_counter() - total_start:.3f}s error=无效 wav")
            raise HTTPException(status_code=400, detail={"ok": False, "error": "invalid wav"})

        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"finish 被忽略 因会话已取消 session={session}")
                return {"ok": True, "status": "canceled"}
            ai_session.upload_wav_path = save_path
        log(f"[AI] 已上传 WAV: {save_path}")

        # 读取运行参数
        spot_id = os.getenv("TOUR_DEFAULT_SPOT_ID", "dayanta")
        mode = os.getenv("TOUR_MODE", "asr_bailian_app")
        log(f"[AI] mode={mode} llm_provider=bailian_app")
        image_context = ai_session.image_context

        # 执行 ASR + LLM 处理（支持中途取消）
        try:
            asr_text, answer_text = await process_text_with_cancel(
                ai_session,
                save_path,
                spot_id=spot_id,
                image_context=image_context,
                mode=mode,
            )
        except Exception as exc:
            if str(exc).startswith("ASR 失败"):
                log(f"AI ASR 失败 session={session}: {exc}")
                log(f"[AI-TIME] finish_text_total={time.perf_counter() - total_start:.3f}s error={exc}")
                raise HTTPException(status_code=500, detail={"ok": False, "error": "asr failed"})
            log(f"AI 编排失败 session={session}: {exc}")
            answer_text = ERROR_TEXT
            asr_text = ai_session.asr_text

        # 更新会话状态
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"LLM 结果因会话已取消而被忽略 session={session}")
                return {"ok": True, "status": "canceled"}
            ai_session.asr_text = asr_text
            ai_session.save_path = save_path
            ai_session.answer_text = answer_text
            ai_session.status = "text_ready"
            ai_session.audio_ready = False
            ai_session.reply_wav_ready = False
            ai_session.reply = None
            ai_session.reply_path = None
            ai_session.reply_wav_size = 0
            ai_session.reply_duration = 0.0
            ai_session.tts_error = None
            ai_session.tts_status = "pending" if answer_text.strip() and auto_tts_background_enabled() else "disabled"
        log(f"[AI] text_ready session={session} answer_chars={len(answer_text)}")
        log(f"[AI-TIME] finish_text_total={time.perf_counter() - total_start:.3f}s")

        # 启动后台 TTS
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"TTS 跳过 因会话已取消 session={session}")
                return {"ok": True, "status": "canceled"}
        maybe_start_tts_background(ai_session)
        return {"ok": True, "status": "processing"}

    @app.post("/ai/result_info")
    async def ai_result_info(request: Request, session: str = Query(...)) -> dict[str, object]:
        """查询 AI 会话处理状态和结果信息。

        客户端应轮询此接口检查 audio_ready/reply_wav_ready 状态。
        """
        body = await request.body()
        await log_request(request, body, "result_info")
        ai_session = get_session(session)
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                return canceled_result_info(session, ai_session)
        reply_len = 0
        with app.state.ai_sessions_lock:
            if ai_session.reply_wav_ready:
                reply_len = (
                    ai_session.reply_path.stat().st_size
                    if ai_session.reply_path and ai_session.reply_path.exists()
                    else len(ai_session.reply or b"")
                )
                ai_session.reply_wav_size = reply_len
            return {
                "ok": True,
                "session": session,
                "ready": ai_session.reply_wav_ready,
                "total": reply_len,
                "format": "wav",
                "text": ai_session.answer_text,
                "status": ai_session.status,
                "asr_text": ai_session.asr_text,
                "answer_text": ai_session.answer_text,
                "audio_ready": ai_session.audio_ready,
                "reply_wav_ready": ai_session.reply_wav_ready,
                "reply_wav_size": ai_session.reply_wav_size,
                "reply_duration": ai_session.reply_duration,
                "tts_status": ai_session.tts_status,
                "tts_error": ai_session.tts_error,
            }

    @app.post("/ai/result_chunk")
    async def ai_result_chunk(
        request: Request,
        session: str = Query(...),
        offset: int = Query(0),
        len_: int = Query(DEFAULT_CHUNK_SIZE, alias="len"),
    ) -> Response:
        """分块下载 TTS 合成的回复 WAV 音频。

        参数：
        - session: 会话 ID
        - offset: 数据偏移（字节）
        - len: 块大小（字节），默认 32768
        """
        body = await request.body()
        await log_request(request, body, "result_chunk")
        ai_session = get_session(session)
        with app.state.ai_sessions_lock:
            if is_session_canceled(ai_session):
                mark_session_canceled(ai_session)
                log(f"result_chunk 被拒绝 因会话已取消 session={session}")
                return JSONResponse(
                    {"ok": False, "status": "canceled", "error": "session canceled"},
                    status_code=409,
                )
            if ai_session.reply is None:
                return Response(b"not ready", status_code=409, media_type="text/plain")
            reply_path = ai_session.reply_path
            reply_bytes = ai_session.reply
        reply = reply_path.read_bytes() if reply_path and reply_path.exists() else reply_bytes
        if offset < 0 or len_ <= 0 or offset >= len(reply):
            return Response(b"range invalid", status_code=416, media_type="text/plain")
        chunk = reply[offset : offset + len_]
        log(f"AI result_chunk session={session} offset={offset} len={len(chunk)}")
        return Response(chunk, media_type="application/octet-stream")

    # =========================================================================
    # 相机接口
    # =========================================================================

    @app.post("/camera/upload")
    async def camera_upload(
        request: Request,
        content_type: str = Header("", alias="content-type"),
        device: str = Query("walkie-01"),
    ) -> JSONResponse:
        """相机 JPEG 图片上传接口。

        上传后自动触发：
        1. 图片验证和保存
        2. 视觉分析（VisionService）
        3. 导游讲解生成（PhotoGuideService）
        4. 返回分析结果和讲解文本

        参数：
        - device: 设备标识（默认 walkie-01）
        """
        body = await request.body()
        await log_request(request, body, "camera_upload")
        if "image/jpeg" not in content_type.lower() and "image/jpg" not in content_type.lower():
            log(f"Camera 上传 content-type 警告: {content_type!r}")

        # 验证并保存 JPEG
        ok, save_path, jpeg = validate_and_log_jpeg(body, app.state.jpg_save_dir, "Camera upload")
        if not ok or save_path is None:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "invalid jpeg",
                    "len": len(body),
                    "file": save_path.as_posix() if save_path else "",
                },
                status_code=400,
            )

        width = jpeg.width if jpeg and jpeg.width is not None else 0
        height = jpeg.height if jpeg and jpeg.height is not None else 0
        safe_device = device or "walkie-01"
        image_id = save_path.stem

        # 缓存最新图片
        app.state.latest_images[safe_device] = {
            "image_id": image_id,
            "path": save_path,
            "time": datetime.now(),
            "width": width,
            "height": height,
        }
        log(f"Camera 最新图片已更新 device={safe_device} image_id={image_id} file={save_path}")

        # 视觉分析和导游讲解
        observation = await analyze_camera_observation(safe_device, image_id, save_path)
        mode = choose_mode(observation)
        analysis_ok = mode != RETAKE_MODE
        data = observation.to_dict()
        response_data = {
            "ok": True,
            "len": len(body),
            "width": width,
            "height": height,
            "file": save_path.as_posix(),
            "device": safe_device,
            "image_id": image_id,
            "analysis_ok": analysis_ok,
            "mode": mode,
            "best_candidate_id": data["best_candidate_id"],
            "best_candidate_name": data["best_candidate_name"],
            "candidate_confidence": data["candidate_confidence"],
            "category": data["category"],
            "top_candidates": data["top_candidates"],
            "visible_features": data["visible_features"],
            "visual_evidence": data["visual_evidence"],
            "risk": data["risk"],
            "safe_answer_level": data["safe_answer_level"],
            "need_retake": data["need_retake"] or mode == RETAKE_MODE,
            "grounded": False,
            "answer_text": ""
            if analysis_ok
            else "这张照片信息不太够。请把展品放在画面中间，靠近一点，避开展柜反光后重拍。",
            # 兼容旧版客户端/调试工具
            "scene_type": data["scene_type"],
            "object_category": data["object_category"],
            "visual_features": data["visual_features"],
            "readable_text": data["readable_text"],
            "possible_subject": data["possible_subject"],
            "category_confidence": data["category_confidence"],
            "specific_name_confidence": data["specific_name_confidence"],
        }
        return JSONResponse(response_data)

    @app.post("/camera/analyze_latest")
    async def camera_analyze_latest(
        request: Request,
        device: str = Query("walkie-01"),
    ) -> JSONResponse:
        """分析最新上传的相机图片并返回导游讲解。

        如果有缓存的视觉分析结果则直接复用，否则重新分析。
        适用于语音场景下对最新图片进行讲解。

        参数：
        - device: 设备标识（默认 walkie-01）
        """
        body = await request.body()
        await log_request(request, body, "camera_analyze_latest")
        safe_device = device or "walkie-01"

        # 获取最新图片
        latest = app.state.latest_images.get(safe_device)
        if latest is None and safe_device != "walkie-01":
            latest = app.state.latest_images.get("walkie-01")
        if latest is None:
            return JSONResponse(
                {"ok": False, "device": safe_device, "error": "no camera image uploaded"},
                status_code=404,
            )

        image_path = latest.get("path")
        if not isinstance(image_path, Path) or not image_path.exists():
            return JSONResponse(
                {"ok": False, "device": safe_device, "error": "latest image missing"},
                status_code=404,
            )

        image_id = str(latest.get("image_id") or image_path.stem)

        # 优先使用缓存的分析结果
        cached = app.state.latest_image_analysis.get(safe_device)
        if (
            isinstance(cached, dict)
            and cached.get("image_id") == image_id
            and isinstance(cached.get("observation"), VisionObservation)
        ):
            observation = cached["observation"]
            log(f"[CAMERA] 使用缓存视觉结果 image_id={image_id} status={cached.get('status')}")
        else:
            observation = await analyze_camera_observation(safe_device, image_id, image_path)

        # 生成导游讲解
        guide_start = time.perf_counter()
        guide = await asyncio.to_thread(
            app.state.photo_guide_service.build_answer,
            observation,
            device=safe_device,
            image_id=image_id,
        )
        log(
            f"[CAMERA] 导游讲解 image_id={image_id} mode={guide.mode} grounded={int(guide.grounded)} "
            f"answer_chars={len(guide.answer_text)} cost={time.perf_counter() - guide_start:.3f}s"
        )
        return JSONResponse(
            response_payload(
                device=safe_device,
                image_id=image_id,
                observation=observation,
                guide=guide,
            )
        )

    # =========================================================================
    # 一次性 WAV 回显接口（测试用）
    # =========================================================================

    @app.post("/ai/wav")
    async def ai_wav_oneshot(request: Request) -> Response:
        """一次性 WAV 回显接口。

        接收 WAV 并直接返回，用于快速测试音频通道。
        """
        body = await request.body()
        await log_request(request, body, "one_shot")
        if parse_wav(body) is None:
            return Response(b"expected audio/wav", status_code=400, media_type="text/plain")
        validate_and_log_wav(body, app.state.save_dir, "HTTP one-shot")
        return Response(body, media_type="audio/wav")

    return app


# =============================================================================
# HTTP 服务器启动
# =============================================================================

def run_http(
    host: str,
    port: int,
    wav_save_dir: Path,
    jpg_save_dir: Path,
    ai_reply_repeat: int,
    ai_reply_extra_chunk: bool,
) -> None:
    """启动 FastAPI HTTP 服务器。

    Args:
        host: 绑定地址
        port: 绑定端口
        wav_save_dir: WAV 保存目录
        jpg_save_dir: JPEG 保存目录
        ai_reply_repeat: AI 回复重复次数
        ai_reply_extra_chunk: 是否添加额外数据块
    """
    app = create_http_app(wav_save_dir, jpg_save_dir, ai_reply_repeat, ai_reply_extra_chunk)
    log(f"FastAPI AI WAV + 相机 JPEG 测试服务监听 {host}:{port}")
    log(f"AI 基础 URL: http://<PC_LAN_IP>:{port}")
    log(f"AI reply repeat={max(ai_reply_repeat, 1)} extra_chunk={int(ai_reply_extra_chunk)}")
    log(f"相机上传 URL: http://<PC_LAN_IP>:{port}/camera/upload")
    uvicorn.run(app, host=host, port=port, log_level="warning")


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    """主函数：解析命令行参数并同时启动 UDP 和 HTTP 服务。

    UDP 和 HTTP 分别在独立守护线程中运行。
    主线程等待 Ctrl+C 信号后退出。
    """
    parser = argparse.ArgumentParser(description="Walkie 业务测试服务器")
    parser.add_argument("--host", default=DEFAULT_BIND_HOST, help="绑定地址")
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT, help="WTK1 UDP 监听端口")
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT, help="AI WAV HTTP 端口")
    parser.add_argument("--wav-save-dir", default=str(DEFAULT_WAV_SAVE_DIR), help="接收到的 WAV 文件保存目录")
    parser.add_argument("--jpg-save-dir", default=str(DEFAULT_JPG_SAVE_DIR), help="接收到的 JPEG 文件保存目录")
    parser.add_argument(
        "--ai-reply-repeat",
        type=int,
        default=DEFAULT_AI_REPLY_REPEAT,
        help="在 AI 回复 WAV 中重复上传的 PCM 数据次数",
    )
    parser.add_argument(
        "--ai-reply-extra-chunk",
        action="store_true",
        default=DEFAULT_AI_REPLY_EXTRA_CHUNK,
        help="在回复数据前插入 JUNK 块，用于测试非 44 字节 WAV 数据偏移",
    )
    args = parser.parse_args()

    # 启动 UDP 服务器（守护线程）
    threading.Thread(target=run_udp, args=(args.host, args.udp_port), daemon=True).start()
    # 启动 HTTP 服务器（守护线程）
    threading.Thread(
        target=run_http,
        args=(
            args.host,
            args.http_port,
            Path(args.wav_save_dir),
            Path(args.jpg_save_dir),
            args.ai_reply_repeat,
            args.ai_reply_extra_chunk,
        ),
        daemon=True,
    ).start()

    log("按 Ctrl+C 停止")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        log("已停止")


if __name__ == "__main__":
    main()
