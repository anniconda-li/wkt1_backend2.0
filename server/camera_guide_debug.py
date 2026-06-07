"""相机导游调试模块。

提供端到端的相机导游测试功能：
1. 调用视觉服务分析图片（VisionService）
2. 将视觉结果重写为知识库检索 prompt
3. 调用百炼应用生成导游讲解（BailianAppService）

主要用于调试和验证"拍照→视觉识别→知识库导游"这条完整链路。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.paths import DEFAULT_CAMERA_TEST_IMAGE
from services.bailian_app_service import FALLBACK_TEXT, BailianAppService
from services.vision_service import VisionJsonParseError, VisionService

# 默认测试提问文本
DEFAULT_CAMERA_GUIDE_TEST_TEXT = "这是什么"
logger = logging.getLogger(__name__)


def build_camera_guide_prompt(user_text: str, vision_result: dict[str, Any]) -> str:
    """根据视觉分析结果和用户问题构建知识库检索 prompt。

    将视觉模型的输出（类别、关键词、视觉描述）重组为
    适合百炼知识库应用检索和回答的提示词。

    Args:
        user_text: 用户的问题（如"这是什么"）
        vision_result: 视觉分析结果字典（来自 analyze_for_guide_context）

    Returns:
        str: 重写后的知识库检索 prompt
    """
    visual_summary = str(vision_result.get("visual_summary") or "").strip() or "图片中展品信息不清。"
    keywords = "、".join(_str_list(vision_result.get("search_keywords"))) or "无明确关键词"
    category = str(vision_result.get("category") or "无法判断").strip() or "无法判断"
    return (
        "游客拍到一件展品。图片视觉描述：\n"
        f"{visual_summary}\n\n"
        "检索关键词：\n"
        f"{keywords}\n\n"
        f'用户问："{user_text}"\n\n'
        "请根据知识库中同一文物条目的视觉检索描述、文物基础信息和导游讲解回答。\n"
        "具体展品名称必须使用知识库原文中的标准名称或别名，不得自造名称。\n"
        f"如果无法匹配到具体文物，请只说明它可能属于{category}类展品，不要编造具体名称。\n"
        "回答控制在 80 字以内，适合语音播报。"
    )


async def run_camera_guide_test(
    *,
    vision_service: VisionService,
    bailian_app_service: BailianAppService,
    test_image_path: Path = DEFAULT_CAMERA_TEST_IMAGE,
    user_text: str = DEFAULT_CAMERA_GUIDE_TEST_TEXT,
) -> dict[str, Any]:
    """运行一次完整的相机导游测试。

    流程：
    1. 检查测试图片是否存在
    2. 视觉分析（analyze_for_guide_context）
    3. 重写 prompt（build_camera_guide_prompt）
    4. 调用百炼应用获取导游讲解
    5. 返回包含所有中间结果和耗时统计的字典

    Args:
        vision_service: 视觉服务实例
        bailian_app_service: 百炼应用服务实例
        test_image_path: 测试图片路径
        user_text: 模拟用户提问

    Returns:
        dict: 结果字典，ok=True 表示成功
    """
    total_start = time.perf_counter()
    test_image_path = Path(test_image_path)
    # 检查图片是否存在
    if not test_image_path.exists():
        return _failure(
            stage="image_not_found",
            error_type="FileNotFoundError",
            error=(
                f"默认测试图片不存在：{test_image_path}；请把测试图片放到 "
                "tests/data/camera/test_exhibit.jpg"
            ),
            test_image_path=test_image_path,
            total_start=total_start,
        )

    # 第 1 步：视觉分析
    vision_start = time.perf_counter()
    try:
        vision_result = await asyncio.to_thread(vision_service.analyze_for_guide_context, test_image_path)
    except VisionJsonParseError as exc:
        # JSON 解析失败，记录调试日志
        _log_camera_guide_debug(
            {
                "test_image_path": str(test_image_path),
                "user_text": user_text,
                "vision_elapsed_ms": _elapsed_ms(vision_start),
                "raw_response": _preview_text(exc.raw_response, 500),
                "total_elapsed_ms": _elapsed_ms(total_start),
            }
        )
        return _failure(
            stage="vision",
            error_type=type(exc).__name__,
            error=str(exc),
            test_image_path=test_image_path,
            total_start=total_start,
            extra={"raw_response": exc.raw_response},
        )
    except Exception as exc:
        _log_camera_guide_debug(
            {
                "test_image_path": str(test_image_path),
                "user_text": user_text,
                "vision_elapsed_ms": _elapsed_ms(vision_start),
                "bailian_error_type": "",
                "total_elapsed_ms": _elapsed_ms(total_start),
            }
        )
        return _failure(
            stage="vision",
            error_type=type(exc).__name__,
            error=str(exc),
            test_image_path=test_image_path,
            total_start=total_start,
        )
    vision_elapsed_ms = _elapsed_ms(vision_start)

    # 第 2 步：重写 prompt 并调用百炼应用
    rewritten_prompt = build_camera_guide_prompt(user_text, vision_result)
    bailian_start = time.perf_counter()
    try:
        bailian_answer = await bailian_app_service.ask_async(rewritten_prompt)
    except Exception as exc:
        bailian_elapsed_ms = _elapsed_ms(bailian_start)
        _log_camera_guide_debug(
            _debug_log_payload(
                test_image_path=test_image_path,
                user_text=user_text,
                vision_elapsed_ms=vision_elapsed_ms,
                vision_result=vision_result,
                rewritten_prompt=rewritten_prompt,
                bailian_elapsed_ms=bailian_elapsed_ms,
                bailian_error_type=type(exc).__name__,
                total_elapsed_ms=_elapsed_ms(total_start),
            )
        )
        return _failure(
            stage="bailian",
            error_type=type(exc).__name__,
            error=str(exc),
            test_image_path=test_image_path,
            total_start=total_start,
        )
    bailian_elapsed_ms = _elapsed_ms(bailian_start)

    # 检查是否为降级回复
    if bailian_answer == FALLBACK_TEXT:
        total_elapsed_ms = _elapsed_ms(total_start)
        _log_camera_guide_debug(
            _debug_log_payload(
                test_image_path=test_image_path,
                user_text=user_text,
                vision_elapsed_ms=vision_elapsed_ms,
                vision_result=vision_result,
                rewritten_prompt=rewritten_prompt,
                bailian_elapsed_ms=bailian_elapsed_ms,
                bailian_error_type="BailianFallback",
                total_elapsed_ms=total_elapsed_ms,
            )
        )
        return _failure(
            stage="bailian",
            error_type="BailianFallback",
            error=bailian_answer,
            test_image_path=test_image_path,
            total_start=total_start,
        )

    # 成功完成
    total_elapsed_ms = _elapsed_ms(total_start)
    _log_camera_guide_debug(
        _debug_log_payload(
            test_image_path=test_image_path,
            user_text=user_text,
            vision_elapsed_ms=vision_elapsed_ms,
            vision_result=vision_result,
            rewritten_prompt=rewritten_prompt,
            bailian_elapsed_ms=bailian_elapsed_ms,
            bailian_error_type="",
            total_elapsed_ms=total_elapsed_ms,
        )
    )
    return {
        "ok": True,
        "test_image_path": str(test_image_path),
        "user_text": user_text,
        "vision_result": vision_result,
        "rewritten_prompt": rewritten_prompt,
        "rewritten_prompt_len": len(rewritten_prompt),
        "bailian_answer": bailian_answer,
        "timing": {
            "vision_elapsed_ms": vision_elapsed_ms,
            "bailian_elapsed_ms": bailian_elapsed_ms,
            "total_elapsed_ms": total_elapsed_ms,
        },
    }


def _debug_log_payload(
    *,
    test_image_path: Path,
    user_text: str,
    vision_elapsed_ms: int,
    vision_result: dict[str, Any],
    rewritten_prompt: str,
    bailian_elapsed_ms: int,
    bailian_error_type: str,
    total_elapsed_ms: int,
) -> dict[str, Any]:
    """构建调试日志负载（截取关键字段，避免日志过长）。"""
    return {
        "test_image_path": str(test_image_path),
        "user_text": user_text,
        "vision_elapsed_ms": vision_elapsed_ms,
        "vision_category": vision_result.get("category"),
        "vision_confidence": vision_result.get("confidence"),
        "visual_summary": vision_result.get("visual_summary"),
        "search_keywords": vision_result.get("search_keywords"),
        "rewritten_prompt_len": len(rewritten_prompt),
        "rewritten_prompt_preview": _preview_text(rewritten_prompt, 500),
        "bailian_elapsed_ms": bailian_elapsed_ms,
        "bailian_error_type": bailian_error_type,
        "total_elapsed_ms": total_elapsed_ms,
    }


def _failure(
    *,
    stage: str,
    error_type: str,
    error: str,
    test_image_path: Path,
    total_start: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建失败结果字典。"""
    data: dict[str, Any] = {
        "ok": False,
        "stage": stage,
        "error_type": error_type,
        "error": error,
        "test_image_path": str(test_image_path),
        "timing": {"total_elapsed_ms": _elapsed_ms(total_start)},
    }
    if extra:
        data.update(extra)
    return data


def _log_camera_guide_debug(payload: dict[str, Any]) -> None:
    """输出相机导游调试日志（JSON 格式）。"""
    text = json.dumps(payload, ensure_ascii=False)
    logger.info("[CAMERA-GUIDE-DEBUG] %s", text)
    print(f"[CAMERA-GUIDE-DEBUG] {text}", flush=True)


def _elapsed_ms(start: float) -> int:
    """计算从 start 到现在的毫秒数。"""
    return int((time.perf_counter() - start) * 1000)


def _preview_text(text: str, limit: int) -> str:
    """截取文本预览，特殊字符转义。"""
    normalized = (text or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _str_list(value: Any) -> list[str]:
    """将输入值转为非空字符串列表。"""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
