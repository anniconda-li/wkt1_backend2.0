"""相机导游调试脚本。

命令行工具，用于端到端测试"视觉识别 → 知识库检索 → 导游讲解"完整链路。

用法：
    python tests/scripts/camera_guide.py --image path/to/image.jpg --text "这是什么"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.config  # noqa: E402,F401 - 加载项目 .env 环境变量
from core.paths import DEFAULT_CAMERA_TEST_IMAGE, ensure_project_dirs
from services.bailian_app_service import BailianAppService
from tools.camera_guide_debug import DEFAULT_CAMERA_GUIDE_TEST_TEXT, run_camera_guide_test
from services.vision_service import VisionService


def main() -> int:
    """主函数：解析参数并运行相机导游测试。

    Returns:
        int: 0 表示成功，1 表示失败
    """
    parser = argparse.ArgumentParser(description="运行一次相机导游调试流程")
    parser.add_argument(
        "--image",
        default=str(DEFAULT_CAMERA_TEST_IMAGE),
        help="测试图片路径，默认使用 DEFAULT_CAMERA_TEST_IMAGE",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_CAMERA_GUIDE_TEST_TEXT,
        help="固定的用户提问",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    result = asyncio.run(
        run_camera_guide_test(
            vision_service=VisionService(),
            bailian_app_service=BailianAppService(),
            test_image_path=Path(args.image),
            user_text=args.text,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
