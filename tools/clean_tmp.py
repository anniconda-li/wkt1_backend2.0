"""临时文件清理工具。

清理项目 tmp/ 目录下的各类临时文件：
- 相机预处理图片
- 测试中间文件
- 音频调试文件
- 旧版临时目录

清理后自动重新创建运行时需要的目录结构。
包含安全保护：只清理 tmp/ 目录下的内容，不会误删其他位置。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.paths import (
    DEFAULT_CAMERA_TEST_IMAGE,
    RUNTIME_DIRS,
    TMP_CAMERA_PREPROCESS_DIR,
    TMP_DEBUG_DIR,
    TMP_DIR,
    ensure_project_dirs,
)

# 需要清理的目录和文件列表
CLEAN_PATHS = (
    TMP_CAMERA_PREPROCESS_DIR,
    TMP_DIR / "camera" / "latest",
    TMP_DIR / "camera" / "test",
    TMP_DIR / "audio" / "debug_reply_wav",
    TMP_DIR / "audio" / "received_wav",
    TMP_DIR / "audio" / "reply_wav",
    TMP_DIR / "runs",
    TMP_DEBUG_DIR / "test_ai_cancel",
    TMP_DIR / "test_ai_cancel",
)


def _safe_remove(path: Path) -> bool:
    """安全删除文件或目录（仅限 tmp/ 目录下）。

    安全保护：
    - 路径必须在 TMP_DIR 下
    - 不会删除 TMP_DIR 本身

    Args:
        path: 要删除的路径

    Returns:
        bool: 是否实际执行了删除操作

    Raises:
        RuntimeError: 尝试删除 tmp/ 目录外的路径
    """
    if not path.exists():
        return False
    tmp_root = TMP_DIR.resolve()
    resolved = path.resolve()
    # 安全检查：只允许删除 tmp/ 目录下的内容
    if resolved == tmp_root or tmp_root not in resolved.parents:
        raise RuntimeError(f"拒绝删除 tmp/ 目录外的路径: {resolved}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def main() -> int:
    """主函数：清理临时文件并重建运行时目录。

    Returns:
        int: 总是返回 0
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    ensure_project_dirs()

    # 逐个清理
    removed = []
    for path in CLEAN_PATHS:
        if _safe_remove(path):
            removed.append(str(path))

    # 确保清理后运行时目录仍然存在
    ensure_project_dirs()

    # 输出清理结果
    print("[OK] tmp/ 项目目录已确保")
    print(f"[OK] runtime_dirs={len(RUNTIME_DIRS)}")
    print(f"[OK] default_test_image={DEFAULT_CAMERA_TEST_IMAGE}")
    print(f"[OK] removed={removed if removed else 'none'}")
    print("[INFO] 固定的相机测试图片位于 tests/data/camera")
    return 0


if __name__ == "__main__":
    sys.exit(main())
