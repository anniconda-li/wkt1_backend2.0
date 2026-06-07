"""WTK1 对讲机后端服务入口。

这是整个应用的主入口文件，负责：
1. 加载项目环境变量配置
2. 从环境变量读取运行参数（WAV 保存目录、JPEG 保存目录、AI 回复参数等）
3. 创建并启动 FastAPI HTTP 应用

启动方式：
    python main.py
    或通过 uvicorn 直接启动：uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

import core.config  # noqa: F401 - 加载项目 .env 环境变量
from server.walkie_app import (
    DEFAULT_AI_REPLY_EXTRA_CHUNK,
    DEFAULT_AI_REPLY_REPEAT,
    DEFAULT_JPG_SAVE_DIR,
    DEFAULT_WAV_SAVE_DIR,
    create_http_app,
)

# 创建 FastAPI 应用实例，配置从环境变量读取，未设置时使用默认值
app = create_http_app(
    # WAV 音频文件保存目录
    Path(os.getenv("WAV_SAVE_DIR", str(DEFAULT_WAV_SAVE_DIR))),
    # JPEG 图片文件保存目录
    Path(os.getenv("JPG_SAVE_DIR", str(DEFAULT_JPG_SAVE_DIR))),
    # AI 回复 WAV 中重复 PCM 数据的次数
    int(os.getenv("AI_REPLY_REPEAT", str(DEFAULT_AI_REPLY_REPEAT))),
    # 是否在 AI 回复 WAV 中插入额外数据块（用于测试非标准 WAV 偏移）
    os.getenv("AI_REPLY_EXTRA_CHUNK", str(int(DEFAULT_AI_REPLY_EXTRA_CHUNK))).strip().lower()
    in {"1", "true", "yes", "on"},
)
