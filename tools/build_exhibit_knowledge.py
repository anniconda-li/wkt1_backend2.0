"""展品知识库构建工具。

从展品参考图片和候选配置文件出发，通过视觉模型生成每件展品的：
1. 详细视觉检索描述（detailed_visual_description）
2. 视觉检索关键词（visual_keywords）
3. 名称约束规则（name_constraints）

并将结果导出为 Markdown 知识文档，供百炼知识库应用使用。

工作流程：
1. 加载 museum_vision_candidates.json 候选配置
2. 对每个候选展品，使用参考图片调用视觉模型生成视觉检索索引
3. 将结果保存到 museum_vision_index.json
4. 导出每个展品为独立的 .md 知识文档
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.config  # noqa: E402,F401 - 加载项目 .env 环境变量
from core.paths import CONFIG_DIR, EXHIBITS_KNOWLEDGE_DIR, ensure_project_dirs

# 配置文件路径
CANDIDATES_PATH = CONFIG_DIR / "museum_vision_candidates.json"
VISION_INDEX_PATH = CONFIG_DIR / "museum_vision_index.json"
LOG_PREFIX = "[BUILD-EXHIBIT-KNOWLEDGE]"


def main() -> int:
    """主函数：构建展品知识库。

    Returns:
        int: 0 成功，1 有视觉识别失败
    """
    parser = argparse.ArgumentParser(description="从参考图片生成展品知识库 Markdown 文档")
    parser.add_argument("--id", dest="candidate_id", default="", help="仅处理指定展品 ID")
    parser.add_argument("--limit", type=int, default=0, help="最多处理的展品数量")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有的视觉索引条目")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置和图片，不写文件")
    parser.add_argument("--export-only", action="store_true", help="仅从已有视觉索引导出 Markdown")
    args = parser.parse_args()

    ensure_project_dirs()
    EXHIBITS_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    # 加载候选展品和已有索引
    candidates = _select_candidates(_load_candidates(CANDIDATES_PATH), args.candidate_id, args.limit)
    index_by_id = _load_index(VISION_INDEX_PATH)
    _log(f"已加载候选展品 count={len(candidates)}")

    # 统计信息
    summary = {
        "total_candidates": len(candidates),
        "processed": 0,
        "skipped_existing": 0,
        "missing_images": 0,
        "vision_success": 0,
        "vision_failed": 0,
        "markdown_exported": 0,
    }

    if args.dry_run:
        _dry_run(candidates, summary)
        _print_summary(summary)
        return 0

    # 逐展品处理
    for candidate in candidates:
        candidate_id = _candidate_id(candidate)
        standard_name = _standard_name(candidate)
        summary["processed"] += 1
        _log(f"处理 id={candidate_id} name={standard_name}")

        image_path = _first_existing_reference_image(candidate)
        if image_path is None:
            summary["missing_images"] += 1
            for ref in _reference_images(candidate) or [""]:
                _log(f"图片缺失 path={ref}")
            if candidate_id not in index_by_id:
                index_by_id[candidate_id] = _entry_from_candidate(
                    candidate,
                    None,
                    parse_ok=False,
                    error="缺少参考图片",
                )
        else:
            _log(f"图片存在 path={_project_relative(image_path)}")

        existing = index_by_id.get(candidate_id)

        if args.export_only:
            # 仅导出模式：不调用视觉模型
            if existing is None:
                index_by_id[candidate_id] = _entry_from_candidate(
                    candidate,
                    image_path,
                    parse_ok=False,
                    error="未索引",
                )
            summary["markdown_exported"] += _export_one(candidate, index_by_id[candidate_id])
            continue

        if image_path is not None and existing and existing.get("parse_ok") is True and not args.overwrite:
            # 跳过已有成功索引的展品
            summary["skipped_existing"] += 1
            _log(f"跳过已有视觉索引 id={candidate_id}")
        elif image_path is not None:
            # 调用视觉模型生成索引
            _log(f"视觉模型开始 id={candidate_id}")
            start = time.perf_counter()
            try:
                vision_result = _call_vision_model(candidate, image_path)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                index_by_id[candidate_id] = _entry_from_candidate(
                    candidate,
                    image_path,
                    parse_ok=True,
                    error="",
                    result=vision_result,
                )
                summary["vision_success"] += 1
                _log(f"视觉模型完成 id={candidate_id} elapsed_ms={elapsed_ms} parse_ok=true")
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                index_by_id[candidate_id] = _entry_from_candidate(
                    candidate,
                    image_path,
                    parse_ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
                summary["vision_failed"] += 1
                _log(f"视觉模型完成 id={candidate_id} elapsed_ms={elapsed_ms} parse_ok=false")

        # 保存视觉索引
        VISION_INDEX_PATH.write_text(json.dumps(index_by_id, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"索引已保存 path={_project_relative(VISION_INDEX_PATH)}")
        # 导出 Markdown 知识文档
        summary["markdown_exported"] += _export_one(candidate, index_by_id[candidate_id])

    _print_summary(summary)
    return 0 if summary["vision_failed"] == 0 else 1


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    """从 JSON 文件加载候选展品列表。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"候选展品 JSON 必须是列表: {path}")
    return [item for item in data if isinstance(item, dict)]


def _load_index(path: Path) -> dict[str, dict[str, Any]]:
    """加载已有的视觉索引文件。

    兼容两种格式：
    - 新版：{"entries": [...]}
    - 旧版：{"candidate_id": {...}}
    """
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return {
            str(entry.get("candidate_id") or entry.get("id") or ""): _normalize_entry(entry)
            for entry in data["entries"]
            if isinstance(entry, dict) and (entry.get("candidate_id") or entry.get("id"))
        }
    if isinstance(data, dict):
        normalized: dict[str, dict[str, Any]] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                normalized[str(key)] = _normalize_entry(value)
        return normalized
    return {}


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """规范化索引条目，统一字段名和填充默认值。"""
    candidate_id = str(entry.get("candidate_id") or entry.get("id") or "").strip()
    vision_result = entry.get("vision_result") if isinstance(entry.get("vision_result"), dict) else {}
    detailed = str(entry.get("detailed_visual_description") or "").strip()
    if not detailed:
        detailed = str(vision_result.get("detailed_visual_description") or vision_result.get("visual_description") or "").strip()
    if not detailed:
        detailed = _join_old_visual_fields(vision_result)
    return {
        "candidate_id": candidate_id,
        "standard_name": str(entry.get("standard_name") or vision_result.get("standard_name") or "").strip(),
        "aliases": _str_list(entry.get("aliases") or vision_result.get("aliases")),
        "category": str(entry.get("category") or vision_result.get("category") or "").strip(),
        "reference_image_used": str(entry.get("reference_image_used") or entry.get("reference_image") or "").strip(),
        "detailed_visual_description": detailed,
        "visual_keywords": _str_list(entry.get("visual_keywords") or vision_result.get("visual_keywords")),
        "name_constraints": _str_list(entry.get("name_constraints") or vision_result.get("name_constraints") or vision_result.get("negative_notes")),
        "generated_at": str(entry.get("generated_at") or _utc_now()).strip(),
        "parse_ok": bool(entry.get("parse_ok", entry.get("status") == "ok")),
        "error": str(entry.get("error") or "").strip(),
    }


def _select_candidates(candidates: list[dict[str, Any]], candidate_id: str, limit: int) -> list[dict[str, Any]]:
    """按 ID 或数量限制筛选候选展品。"""
    selected = candidates
    if candidate_id:
        selected = [candidate for candidate in selected if _candidate_id(candidate) == candidate_id]
    if limit > 0:
        selected = selected[:limit]
    return selected


def _dry_run(candidates: list[dict[str, Any]], summary: dict[str, int]) -> None:
    """预检查模式：验证配置和参考图片是否存在。"""
    for candidate in candidates:
        candidate_id = _candidate_id(candidate)
        _log(f"处理 id={candidate_id} name={_standard_name(candidate)}")
        summary["processed"] += 1
        refs = _reference_images(candidate)
        if not refs:
            summary["missing_images"] += 1
            _log("图片缺失 path=")
            continue
        for ref in refs:
            if _project_path(ref).exists():
                _log(f"图片存在 path={ref}")
            else:
                summary["missing_images"] += 1
                _log(f"图片缺失 path={ref}")


def _call_vision_model(candidate: dict[str, Any], image_path: Path) -> dict[str, Any]:
    """调用视觉模型生成展品的视觉检索描述。

    Args:
        candidate: 候选展品配置字典
        image_path: 参考图片路径

    Returns:
        dict: 视觉模型生成的检索描述结果

    Raises:
        RuntimeError: API Key 未配置或调用失败
        ValueError: 不支持的 provider 或返回非 JSON
    """
    provider = os.getenv("VISION_PROVIDER", "dashscope").strip().lower()
    if provider == "mock":
        return _mock_vision_result(candidate)
    if provider != "dashscope":
        raise ValueError(f"不支持的 VISION_PROVIDER: {provider}")

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置，无法调用视觉模型")

    import dashscope

    dashscope.api_key = api_key
    # 调用多模态视觉模型
    response = dashscope.MultiModalConversation.call(
        model=os.getenv("VISION_MODEL", "qwen-vl-plus").strip(),
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": _image_data_url(image_path)},
                    {"text": _build_vision_prompt(candidate)},
                ],
            }
        ],
    )
    data = _response_to_dict(response)
    status_code = data.get("status_code", getattr(response, "status_code", None))
    if status_code not in (None, 200):
        message = data.get("message", getattr(response, "message", ""))
        code = data.get("code", getattr(response, "code", ""))
        raise RuntimeError(f"视觉模型调用失败 status={status_code} code={code} message={message}")

    text = _extract_response_text(data)
    parsed = _extract_json_object(text)
    if not parsed:
        raise ValueError(f"视觉模型返回非 JSON：{_preview_text(text, 500)}")
    return _coerce_vision_result(candidate, parsed)


def _build_vision_prompt(candidate: dict[str, Any]) -> str:
    """构建发送给视觉模型的展品检索描述生成 prompt。

    强调只根据图片可见内容描述，不编造文物历史信息。
    """
    standard_name = _standard_name(candidate)
    aliases = "、".join(_aliases(candidate)) or "无"
    category = str(candidate.get("category") or "").strip()
    return f"""你是博物馆文物"视觉检索索引"生成助手。

你的任务不是写导游讲解，也不是判断文物历史信息，而是根据标准文物图片，生成一段适合后续图像检索和文本匹配的详细视觉描述。

请只根据图片中真实可见的内容描述，不要编造年代、出土地、文物等级、用途、历史故事或价值评价。

我会提供该文物的标准名称、别名和类别。你只能使用这些名称，不得根据图像自行创造新的文物名称。

展品标准名称：{standard_name}
别名：{aliases}
类别：{category}

请只输出 JSON，不要输出 Markdown，不要输出解释文字：

{{
  "standard_name": "{standard_name}",
  "aliases": [],
  "category": "{category}",
  "detailed_visual_description": "",
  "visual_keywords": [],
  "name_constraints": []
}}

字段要求：

1. detailed_visual_description 是最重要字段。
2. detailed_visual_description 必须是一段连贯的详细视觉描述，150～300 字。
3. 描述重点包括：整体形态、轮廓、结构、颜色、材质观感、表面纹饰、特殊部件、容易被视觉模型识别到的特征。
4. 不要堆砌重复短语。
5. 不要写导游讲解。
6. 不要写历史背景。
7. 不要编造不可见信息。
8. visual_keywords 控制在 8～20 个，必须是视觉检索相关词。
9. name_constraints 写名称约束，强调只能使用标准名称或别名，不得自造名称。
10. 如果图片不清晰，也要说明可见特征，但不要补全看不见的细节。"""


def _coerce_vision_result(candidate: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """规范化视觉模型返回的结果数据。"""
    detailed = str(data.get("detailed_visual_description") or data.get("visual_description") or "").strip()
    if not detailed:
        detailed = _join_old_visual_fields(data)
    return {
        "standard_name": _standard_name(candidate),
        "aliases": _str_list(data.get("aliases")) or _aliases(candidate),
        "category": str(data.get("category") or candidate.get("category") or "").strip(),
        "detailed_visual_description": detailed,
        "visual_keywords": _str_list(data.get("visual_keywords"))[:20],
        "name_constraints": _str_list(data.get("name_constraints"))[:20],
    }


def _mock_vision_result(candidate: dict[str, Any]) -> dict[str, Any]:
    """生成 mock 视觉检索描述（用于开发测试）。"""
    standard_name = _standard_name(candidate)
    aliases = _aliases(candidate)
    category = str(candidate.get("category") or "").strip()
    return {
        "standard_name": standard_name,
        "aliases": aliases,
        "category": category,
        "detailed_visual_description": (
            f"{standard_name}参考图的模拟视觉检索描述。这里应记录器物整体形态、轮廓结构、颜色材质、"
            "表面纹饰、特殊部件和容易被照片识别到的特征，用于验证构建流程。"
        ),
        "visual_keywords": [item for item in [category, standard_name, *aliases] if item][:20],
        "name_constraints": [
            f'具体展品名称只能使用标准名称"{standard_name}"或配置中的别名。',
            "不得根据视觉特征拼接新的展品名称。",
        ],
    }


def _entry_from_candidate(
    candidate: dict[str, Any],
    image_path: Path | None,
    *,
    parse_ok: bool,
    error: str,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从候选展品和视觉结果构建索引条目。"""
    result = result or {}
    standard_name = _standard_name(candidate)
    aliases = _aliases(candidate)
    return {
        "candidate_id": _candidate_id(candidate),
        "standard_name": standard_name,
        "aliases": aliases,
        "category": str(candidate.get("category") or "").strip(),
        "reference_image_used": "" if image_path is None else _project_relative(image_path),
        "detailed_visual_description": str(result.get("detailed_visual_description") or "").strip(),
        "visual_keywords": _str_list(result.get("visual_keywords")),
        "name_constraints": _str_list(result.get("name_constraints")) or _default_name_constraints(standard_name),
        "generated_at": _utc_now(),
        "parse_ok": parse_ok,
        "error": error,
    }


def _export_one(candidate: dict[str, Any], entry: dict[str, Any]) -> int:
    """导出一个展品的 Markdown 知识文档。

    Returns:
        int: 导出数量（1）
    """
    candidate_id = _candidate_id(candidate)
    output_path = EXHIBITS_KNOWLEDGE_DIR / f"{candidate_id}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_markdown(candidate, entry), encoding="utf-8")
    _log(f"Markdown 已导出 path={_project_relative(output_path)}")
    return 1


def _build_markdown(candidate: dict[str, Any], entry: dict[str, Any]) -> str:
    """构建展品的完整 Markdown 知识文档。

    包含：标准名称、别名、类别、视觉检索描述、检索关键词、
    文物基础信息、导游讲解、名称约束、资料来源。
    """
    candidate_id = _candidate_id(candidate)
    standard_name = _standard_name(candidate)
    aliases = _aliases(candidate)
    category = str(candidate.get("category") or "").strip()
    basic_info = candidate.get("basic_info") if isinstance(candidate.get("basic_info"), dict) else {}
    detailed = str(entry.get("detailed_visual_description") or "").strip() or "暂无视觉检索描述。"
    keywords = _str_list(entry.get("visual_keywords"))
    constraints = _str_list(entry.get("name_constraints"))
    source_urls = _str_list(candidate.get("source_urls"))
    guide_text = str(candidate.get("guide_text") or "").strip()
    if not guide_text:
        guide_text = _fallback_guide_text(candidate, detailed)

    lines = [
        f"# {standard_name}",
        "",
        f"文物ID：{candidate_id}  ",
        f"标准名称：{standard_name}  ",
        f"别名：{'、'.join(aliases) if aliases else '无'}  ",
        f"类别：{category or '未知'}",
        "",
        "## 视觉检索描述",
        "",
        detailed,
        "",
        "## 视觉检索关键词",
        "",
        "、".join(keywords) if keywords else "暂无",
        "",
        "## 文物基础信息",
        "",
        f"年代：{_basic_value(basic_info, 'dynasty', '未知')}  ",
        f"用途：{_basic_value(basic_info, 'usage', '未知')}  ",
        f"材质：{_basic_value(basic_info, 'material', '未知')}  ",
        f"馆藏：{_basic_value(basic_info, 'collection', '平顶山市博物馆')}  ",
        f"出土信息：{_basic_value(basic_info, 'excavation', '暂无明确资料')}",
        "",
        "## 导游讲解",
        "",
        guide_text,
        "",
        "## 名称约束",
        "",
        "具体展品名称只能使用知识库中的标准名称或别名。",
        "",
        f"标准名称：{standard_name}  ",
        f"允许别名：{'、'.join(aliases) if aliases else '无'}",
        "",
        "不得根据视觉描述、类别、形状、材质、年代、地区等信息自行拼接新的展品名称。",
        "",
        "\n".join(constraints) if constraints else "",
        "",
    ]
    if source_urls:
        lines.extend(["## 资料来源", ""])
        lines.extend(f"- {url}" for url in source_urls)
        lines.append("")
    return "\n".join(lines)


def _fallback_guide_text(candidate: dict[str, Any], detailed_visual_description: str) -> str:
    """生成本地降级导游讲解文本（不依赖 LLM）。

    使用展品基本信息和视觉描述拼接一个简单的讲解文案。
    """
    standard_name = _standard_name(candidate)
    category = str(candidate.get("category") or "文物").strip()
    basic_info = candidate.get("basic_info") if isinstance(candidate.get("basic_info"), dict) else {}
    material = str(basic_info.get("material") or "").strip()
    usage = str(basic_info.get("usage") or "").strip()
    parts = [f"{standard_name}是一件{category}类展品"]
    if material:
        parts.append(f"材质为{material}")
    if usage:
        parts.append(f"可按{usage}理解")
    answer = "，".join(parts) + "。"
    if detailed_visual_description.strip() and detailed_visual_description.strip() != "暂无视觉检索描述。":
        answer += " 参观时可以重点观察它的整体器形、结构比例、表面纹饰和特殊部件。"
    else:
        answer += " 参观时可以重点观察它的器形、材质和表面装饰。"
    answer += " 具体年代、出土信息和名称请以现场展签为准。"
    return answer


def _first_existing_reference_image(candidate: dict[str, Any]) -> Path | None:
    """获取候选展品的第一张存在的参考图片路径。"""
    for ref in _reference_images(candidate):
        path = _project_path(ref)
        if path.exists():
            return path
    return None


def _reference_images(candidate: dict[str, Any]) -> list[str]:
    """获取候选展品的参考图片路径列表。"""
    refs = candidate.get("reference_images")
    if not isinstance(refs, list):
        return []
    return [str(ref).strip() for ref in refs if str(ref).strip()]


def _candidate_id(candidate: dict[str, Any]) -> str:
    """获取候选展品 ID。"""
    return str(candidate.get("id") or "").strip()


def _standard_name(candidate: dict[str, Any]) -> str:
    """获取候选展品的标准名称。"""
    return str(candidate.get("standard_name") or candidate.get("name") or "").strip()


def _aliases(candidate: dict[str, Any]) -> list[str]:
    """获取候选展品的别名列表。"""
    return _str_list(candidate.get("aliases"))


def _basic_value(basic_info: dict[str, Any], key: str, fallback: str) -> str:
    """从文物基础信息字典中安全获取值。"""
    return str(basic_info.get(key) or "").strip() or fallback


def _default_name_constraints(standard_name: str) -> list[str]:
    """生成默认的名称约束规则。"""
    return [
        f'具体展品名称只能使用标准名称"{standard_name}"或配置中的别名。',
        "不得根据视觉描述、类别、形状、材质、年代、地区等信息自行拼接新的展品名称。",
    ]


def _project_path(value: str) -> Path:
    """将字符串路径转为绝对路径（相对路径相对于项目根目录）。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _project_relative(path: Path) -> str:
    """将绝对路径转为相对于项目根目录的路径字符串。"""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _image_data_url(image_path: Path) -> str:
    """将图片编码为 base64 data URL。"""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _response_to_dict(response: Any) -> dict[str, Any]:
    """将 DashScope API 响应转为字典。"""
    if isinstance(response, dict):
        return response
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    data: dict[str, Any] = {}
    for name in ("status_code", "code", "message", "output", "usage", "request_id"):
        if hasattr(response, name):
            data[name] = getattr(response, name)
    return data


def _extract_response_text(value: Any) -> str:
    """递归提取 API 响应中的文本内容。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        output = value.get("output")
        if isinstance(output, dict):
            choices = output.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    text = _extract_response_text(choice)
                    if text:
                        return text
            text = output.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        message = value.get("message")
        if isinstance(message, dict):
            text = _extract_response_text(message)
            if text:
                return text
        content = value.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "".join(parts).strip()
        if isinstance(content, str) and content.strip():
            return content.strip()
        for child in value.values():
            text = _extract_response_text(child)
            if text:
                return text
    if isinstance(value, list):
        for child in value:
            text = _extract_response_text(child)
            if text:
                return text
    return ""


def _extract_json_object(text: str) -> dict[str, Any]:
    """从文本中提取 JSON 对象（去除 Markdown 代码块标记后）。"""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _join_old_visual_fields(value: dict[str, Any]) -> str:
    """合并旧版视觉描述字段为单一字符串。"""
    parts = []
    for key in (
        "visual_description",
        "shape_features",
        "material_color_features",
        "decoration_features",
        "possible_user_descriptions",
        "negative_notes",
    ):
        item = value.get(key)
        if isinstance(item, list):
            parts.extend(str(part).strip() for part in item if str(part).strip())
        elif isinstance(item, str) and item.strip():
            parts.append(item.strip())
    return "；".join(parts)


def _str_list(value: Any) -> list[str]:
    """将输入转为非空字符串列表。"""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _preview_text(text: str, limit: int) -> str:
    """截取文本预览，特殊字符转义。"""
    normalized = (text or "").replace("\r", "\\r").replace("\n", "\\n")
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."


def _utc_now() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log(message: str) -> None:
    """输出带前缀的日志。"""
    print(f"{LOG_PREFIX} {message}", flush=True)


def _print_summary(summary: dict[str, int]) -> None:
    """打印构建统计摘要。"""
    print("summary:")
    for key in (
        "total_candidates",
        "processed",
        "skipped_existing",
        "missing_images",
        "vision_success",
        "vision_failed",
        "markdown_exported",
    ):
        print(f"{key}={summary.get(key, 0)}")


if __name__ == "__main__":
    sys.exit(main())
