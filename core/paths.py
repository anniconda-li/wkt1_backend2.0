"""项目路径管理模块。

定义项目中所有关键目录和文件路径常量，包括：
- 知识库目录（展品资料、配置文件、参考资料）
- 测试数据目录
- 临时文件目录（相机图片、音频文件、调试输出）
- 运行时目录
- 向后兼容的旧版路径别名

同时提供目录自动创建和默认测试图片初始化功能。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from core.config import PROJECT_ROOT

# =============================================================================
# 知识库相关路径
# =============================================================================
# 知识库根目录
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
# 知识库配置文件目录
KNOWLEDGE_CONFIG_DIR = KNOWLEDGE_DIR / "config"
# 博物馆参考资料目录
MUSEUM_REFS_DIR = KNOWLEDGE_DIR / "refs"
# 展品知识文档目录
EXHIBITS_KNOWLEDGE_DIR = KNOWLEDGE_DIR / "exhibits"
# 配置目录（与 KNOWLEDGE_CONFIG_DIR 相同，便于导入）
CONFIG_DIR = KNOWLEDGE_CONFIG_DIR

# =============================================================================
# 测试相关路径
# =============================================================================
TESTS_DIR = PROJECT_ROOT / "tests"
TEST_DATA_DIR = TESTS_DIR / "data"
TEST_CAMERA_DIR = TEST_DATA_DIR / "camera"
TEST_AUDIO_DIR = TEST_DATA_DIR / "audio"

# =============================================================================
# 临时文件目录（运行时产生）
# =============================================================================
TMP_DIR = PROJECT_ROOT / "tmp"

# 相机临时目录
TMP_CAMERA_DIR = TMP_DIR / "camera"
# 接收到的原始相机图片
TMP_CAMERA_RECEIVED_DIR = TMP_CAMERA_DIR / "received"
# 视觉预处理后的图片
TMP_CAMERA_PREPROCESS_DIR = TMP_CAMERA_DIR / "preprocess"

# 音频临时目录
TMP_AUDIO_DIR = TMP_DIR / "audio"
# 接收到的音频文件
TMP_AUDIO_RECEIVED_DIR = TMP_AUDIO_DIR / "received"
# 生成的 AI 回复音频
TMP_AUDIO_REPLIES_DIR = TMP_AUDIO_DIR / "replies"

# 调试临时目录
TMP_DEBUG_DIR = TMP_DIR / "debug"
# 调试音频目录
TMP_DEBUG_AUDIO_DIR = TMP_DEBUG_DIR / "audio"

# =============================================================================
# 向后兼容的路径别名
# =============================================================================
# 新代码应使用 TMP_AUDIO_RECEIVED_DIR 和 TMP_AUDIO_REPLIES_DIR
TMP_AUDIO_RECEIVED_WAV_DIR = TMP_AUDIO_RECEIVED_DIR
TMP_AUDIO_REPLY_WAV_DIR = TMP_AUDIO_REPLIES_DIR
TMP_AUDIO_DEBUG_REPLY_WAV_DIR = TMP_DEBUG_AUDIO_DIR

# =============================================================================
# 旧版临时目录路径（仅保留用于旧调用者和一次性迁移）
# =============================================================================
# 运行时代码应使用 tmp/camera、tmp/audio 和 tmp/debug
LEGACY_CAMERA_PREPROCESS_DIR = TMP_DIR / "camera_preprocess"
LEGACY_CAMERA_PREPROCESS_TEST_DIR = TMP_DIR / "camera_preprocess_test"
LEGACY_LATEST_DIR = TMP_DIR / "latest"
LEGACY_PHOTOS_DIR = TMP_DIR / "photos"
LEGACY_RECEIVED_JPG_DIR = TMP_DIR / "received_jpg"
LEGACY_RECEIVED_WAV_DIR = TMP_DIR / "received_wav"
LEGACY_REPLY_WAV_DIR = TMP_DIR / "reply_wav"
LEGACY_DEBUG_REPLY_WAV_DIR = TMP_DIR / "debug_reply_wav"
LEGACY_TEST_AI_CANCEL_DIR = TMP_DIR / "test_ai_cancel"
LEGACY_TEST_JPG_DIR = TMP_DIR / "test_jpg"
# 旧版测试图片路径
LEGACY_CAMERA_TEST_IMAGE = LEGACY_RECEIVED_JPG_DIR / "camera_upload_20260603_165431_081287.jpg"
LEGACY_NORMALIZED_CAMERA_TEST_IMAGE = TMP_CAMERA_DIR / "test" / "camera_upload_20260603_165431_081287.jpg"
LEGACY_NAMED_TEST_IMAGE = TMP_CAMERA_DIR / "test" / "test_exhibit.jpg"

# =============================================================================
# 默认测试图片路径
# =============================================================================
DEFAULT_CAMERA_TEST_IMAGE = Path(
    os.getenv(
        "DEFAULT_CAMERA_TEST_IMAGE",
        str(TEST_CAMERA_DIR / "test_exhibit.jpg"),
    )
)
# 如果不是绝对路径，则相对于项目根目录解析
if not DEFAULT_CAMERA_TEST_IMAGE.is_absolute():
    DEFAULT_CAMERA_TEST_IMAGE = PROJECT_ROOT / DEFAULT_CAMERA_TEST_IMAGE

# =============================================================================
# 运行时需要创建的目录列表
# =============================================================================
RUNTIME_DIRS = (
    TMP_CAMERA_RECEIVED_DIR,
    TMP_CAMERA_PREPROCESS_DIR,
    TMP_AUDIO_RECEIVED_DIR,
    TMP_AUDIO_REPLIES_DIR,
    TMP_DEBUG_DIR,
)

# =============================================================================
# 博物馆展品参考 ID 列表
# =============================================================================
MUSEUM_REF_IDS = (
    "yingguo_yuying",         # 应国玉鹰
    "panlongniu_daigai_tonghe",  # 蟠龙钮带盖铜盒
    "denggong_gui",           # 邓公簋
    "lushan_huaci_sanzuxi",   # 鲁山花瓷三足洗
    "shuyao_chuilinwen_shengding",  # 竖窑垂鳞纹升鼎
)

# =============================================================================
# 所有项目目录（用于一次性创建）
# =============================================================================
PROJECT_DIRS = (
    KNOWLEDGE_DIR,
    KNOWLEDGE_CONFIG_DIR,
    MUSEUM_REFS_DIR,
    EXHIBITS_KNOWLEDGE_DIR,
    CONFIG_DIR,
    TESTS_DIR,
    TEST_DATA_DIR,
    TEST_CAMERA_DIR,
    TEST_AUDIO_DIR,
    *(MUSEUM_REFS_DIR / ref_id for ref_id in MUSEUM_REF_IDS),
    *RUNTIME_DIRS,
)

# 旧版运行时目录
LEGACY_RUNTIME_DIRS = (
    LEGACY_CAMERA_PREPROCESS_DIR,
    LEGACY_CAMERA_PREPROCESS_TEST_DIR,
    LEGACY_LATEST_DIR,
    LEGACY_PHOTOS_DIR,
    LEGACY_RECEIVED_JPG_DIR,
    LEGACY_RECEIVED_WAV_DIR,
    LEGACY_REPLY_WAV_DIR,
    LEGACY_DEBUG_REPLY_WAV_DIR,
    LEGACY_TEST_AI_CANCEL_DIR,
    LEGACY_TEST_JPG_DIR,
)


def ensure_project_dirs() -> None:
    """创建所有项目需要的目录。

    包括知识库目录、测试目录和运行时临时目录，
    并确保默认测试图片存在。
    """
    for path in PROJECT_DIRS:
        path.mkdir(parents=True, exist_ok=True)
    ensure_default_camera_test_image()


def ensure_runtime_dirs() -> None:
    """确保运行时需要的目录都存在。

    调用 ensure_project_dirs() 完成实际工作。
    """
    ensure_project_dirs()


def ensure_default_camera_test_image() -> dict[str, object]:
    """确保默认相机测试图片可用。

    优先尝试从旧版路径复制测试图片到新位置。
    如果目标已存在则跳过复制。

    Returns:
        dict: 包含源路径、目标路径和是否已复制标志的信息字典
    """
    target = DEFAULT_CAMERA_TEST_IMAGE
    # 按优先级尝试的源图片候选路径
    source_candidates = (
        LEGACY_NAMED_TEST_IMAGE,
        LEGACY_NORMALIZED_CAMERA_TEST_IMAGE,
        LEGACY_CAMERA_TEST_IMAGE,
    )
    copied = False
    target.parent.mkdir(parents=True, exist_ok=True)
    source = next((path for path in source_candidates if path.exists()), source_candidates[-1])
    if not target.exists():
        existing_source = next((path for path in source_candidates if path.exists()), None)
        if existing_source is not None:
            shutil.copy2(existing_source, target)
            source = existing_source
            copied = True
    info = {
        "source_test_image": str(source),
        "target_test_image": str(target),
        "copied_from_legacy": copied,
    }
    print(
        "[PATHS] "
        f"source_test_image={info['source_test_image']} "
        f"target_test_image={info['target_test_image']} "
        f"copied_from_legacy={str(copied).lower()}",
        flush=True,
    )
    return info


def env_path(name: str, default: Path) -> Path:
    """从环境变量读取路径，支持相对路径自动转换为绝对路径。

    Args:
        name: 环境变量名称
        default: 默认路径

    Returns:
        Path: 解析后的绝对路径
    """
    value = os.getenv(name, "").strip()
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path
