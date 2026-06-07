"""博物馆参考图片检查工具。

检查 museum_vision_candidates.json 中所有候选展品的参考图片是否存在。
输出 TSV 格式的检查结果，包括展品 ID、标准名称、图片路径和存在状态。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.paths import CONFIG_DIR, ensure_project_dirs

# 候选展品配置文件路径
CANDIDATES_PATH = CONFIG_DIR / "museum_vision_candidates.json"


def main() -> int:
    """主函数：检查所有候选展品的参考图片是否存在。

    Returns:
        int: 总是返回 0
    """
    ensure_project_dirs()
    candidates = _load_candidates(CANDIDATES_PATH)
    total_images = 0
    missing_images = 0

    # 输出 TSV 格式的表头
    print("id\tstandard_name\timage_path\texists")
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        standard_name = str(candidate.get("standard_name") or candidate.get("name") or "")
        reference_images = candidate.get("reference_images")
        if not isinstance(reference_images, list):
            reference_images = []
        for image_path in reference_images:
            image_text = str(image_path)
            exists = (PROJECT_ROOT / image_text).exists()
            total_images += 1
            if not exists:
                missing_images += 1
            print(f"{candidate_id}\t{standard_name}\t{image_text}\t{str(exists).lower()}")

    # 输出汇总统计
    print()
    print(f"total_candidates={len(candidates)}")
    print(f"total_reference_images={total_images}")
    print(f"missing_images={missing_images}")
    return 0


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    """从 JSON 文件加载候选展品列表。

    包含完善的错误处理：文件不存在、JSON 格式错误、非列表格式。

    Returns:
        list[dict]: 符合条件的候选展品字典列表
    """
    if not path.exists():
        print(f"[ERROR] 候选展品文件不存在: {path}")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] 无效的 JSON: {path} error={exc}")
        return []
    if not isinstance(data, list):
        print(f"[ERROR] 候选展品 JSON 必须是列表: {path}")
        return []
    return [item for item in data if isinstance(item, dict)]


if __name__ == "__main__":
    sys.exit(main())
