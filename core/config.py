"""项目配置模块。

负责加载 .env 环境变量文件，并配置 Windows 平台下的 asyncio 事件循环策略。
该模块在导入时会自动执行配置初始化。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 依赖应从 requirements.txt 安装
    load_dotenv = None

# 项目根目录，即本文件向上两级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# .env 环境变量文件路径
ENV_PATH = PROJECT_ROOT / ".env"


def configure_asyncio_for_windows() -> None:
    """为 Windows 平台配置 asyncio 事件循环策略。

    Windows 默认使用 ProactorEventLoop，但某些库（如 UDP 套接字）
    需要 SelectorEventLoop。此函数检测平台并自动切换。
    """
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def load_project_env() -> None:
    """加载项目根目录下的 .env 环境变量文件。

    如果 python-dotenv 未安装则静默跳过。
    """
    if load_dotenv is not None:
        load_dotenv(ENV_PATH)


# 模块导入时自动执行配置
configure_asyncio_for_windows()
load_project_env()
