"""文物检索服务模块。

桥接视觉描述（VisualDescription）与文物候选库，
优先在本地对少量重点展品做视觉特征匹配，必要时回退到百炼知识库检索。

核心流程：
1. 从 VisualDescription 拼合本地检索文本
2. 与 museum_vision_candidates.json 的视觉特征、别名、同义词打分
3. 高置信度直接返回 ArtifactMatchResult
4. 低置信度时可按配置回退百炼视觉检索应用
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.paths import KNOWLEDGE_CONFIG_DIR
from services.bailian_app_service import BailianAppService
from services.vision_service import VisualDescription

DEFAULT_CANDIDATES_PATH = KNOWLEDGE_CONFIG_DIR / "museum_vision_candidates.json"
DEFAULT_LOCAL_MIN_CONFIDENCE = 0.58
DEFAULT_LOCAL_MIN_MARGIN = 0.08


@dataclass(frozen=True)
class ArtifactMatchResult:
    """知识库检索匹配结果。

    由百炼视觉检索应用返回，包含匹配到的文物 ID 和置信度。

    Attributes:
        match_id: 匹配到的文物 ID（如 yingguo_yuying），none 表示未匹配
        match_name: 匹配到的文物名称（如 应国玉鹰）
        confidence: 匹配置信度 (0.0~1.0)
        evidence: 匹配依据说明
        raw_response: 百炼应用原始返回文本（调试用）
    """
    match_id: str = "none"
    match_name: str = "无"
    confidence: float = 0.0
    evidence: str = ""
    raw_response: str = ""
    provider: str = ""

    @property
    def is_matched(self) -> bool:
        """是否匹配置信度足够高，可以进行具体讲解。"""
        return self.match_id != "none" and self.confidence >= 0.6

    def to_dict(self) -> dict[str, Any]:
        """转为字典格式。"""
        return asdict(self)


class ArtifactSearchService:
    """文物检索服务。

    将 VisualDescription 发送到百炼视觉检索应用（挂视觉指纹知识库），
    获取最匹配的文物 ID 和名称。

    Attributes:
        bailian: 百炼视觉检索应用服务实例
    """

    def __init__(
        self,
        bailian_vision_service: BailianAppService,
        *,
        candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    ):
        """初始化文物检索服务。

        Args:
            bailian_vision_service: 百炼视觉检索应用实例（挂视觉指纹知识库）
            candidates_path: 本地候选展品配置文件路径
        """
        self.bailian = bailian_vision_service
        self.candidates_path = candidates_path
        self.candidates = self._load_candidates(candidates_path)

    def search(self, desc: VisualDescription) -> ArtifactMatchResult:
        """同步检索最匹配的文物。

        Web 路由应优先使用 ``search_async``。

        Args:
            desc: 视觉描述

        Returns:
            ArtifactMatchResult: 匹配结果
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("ArtifactSearchService.search() 不能在事件循环中调用，请改用 search_async()")
        return asyncio.run(self.search_async(desc))

    async def search_async(self, desc: VisualDescription) -> ArtifactMatchResult:
        """异步检索最匹配的文物。

        Args:
            desc: 视觉描述

        Returns:
            ArtifactMatchResult: 匹配结果
        """
        total_start = time.perf_counter()
        provider = os.getenv("ARTIFACT_SEARCH_PROVIDER", "local").strip().lower() or "local"
        min_confidence = _env_float("ARTIFACT_LOCAL_MIN_CONFIDENCE", DEFAULT_LOCAL_MIN_CONFIDENCE)
        min_margin = _env_float("ARTIFACT_LOCAL_MIN_MARGIN", DEFAULT_LOCAL_MIN_MARGIN)

        local_result, runner_up = self._search_local(desc)
        margin = local_result.confidence - runner_up.confidence
        print(
            f"[SEARCH-LOCAL] match_id={local_result.match_id} match_name={local_result.match_name} "
            f"confidence={local_result.confidence:.2f} runner_up={runner_up.match_id}:{runner_up.confidence:.2f} "
            f"margin={margin:.2f} provider={provider}",
            flush=True,
        )

        if provider == "local":
            print(
                f"[SEARCH] provider=local match_id={local_result.match_id} "
                f"match_name={local_result.match_name} confidence={local_result.confidence:.2f} "
                f"cost={time.perf_counter() - total_start:.3f}s",
                flush=True,
            )
            return local_result

        if provider == "hybrid" and local_result.confidence >= min_confidence and margin >= min_margin:
            print(
                f"[SEARCH] provider=hybrid-local match_id={local_result.match_id} "
                f"match_name={local_result.match_name} confidence={local_result.confidence:.2f} "
                f"cost={time.perf_counter() - total_start:.3f}s",
                flush=True,
            )
            return local_result

        if provider not in {"bailian", "hybrid"}:
            print(f"[SEARCH] 未知 ARTIFACT_SEARCH_PROVIDER={provider!r}，使用本地结果", flush=True)
            return local_result

        return await self._search_bailian(desc, total_start)

    async def _search_bailian(self, desc: VisualDescription, total_start: float) -> ArtifactMatchResult:
        """调用百炼视觉检索应用。"""
        prompt = self._build_search_prompt(desc)
        print(f"[SEARCH] 检索 prompt 长度={len(prompt)}", flush=True)

        try:
            response = await self.bailian.ask_async(prompt)
        except Exception as exc:
            print(f"[SEARCH] 百炼调用异常 error={exc}", flush=True)
            return ArtifactMatchResult(evidence=f"检索调用异常: {exc}")

        result = self._parse_response(response)
        result = ArtifactMatchResult(
            match_id=result.match_id,
            match_name=result.match_name,
            confidence=result.confidence,
            evidence=result.evidence,
            raw_response=result.raw_response,
            provider="bailian",
        )
        print(
            f"[SEARCH] match_id={result.match_id} match_name={result.match_name} "
            f"confidence={result.confidence:.2f} cost={time.perf_counter() - total_start:.3f}s",
            flush=True,
        )
        return result

    def _build_search_prompt(self, desc: VisualDescription) -> str:
        """构建知识库检索 prompt。

        将 VisualDescription 的各字段组合为检索文本，
        发送到百炼视觉检索应用进行语义匹配。

        Args:
            desc: 视觉描述

        Returns:
            str: 检索 prompt
        """
        parts = []
        parts.append("请在知识库中找到与以下视觉描述最匹配的文物。")
        parts.append("")
        parts.append(f"视觉描述：{desc.visual_description}")
        parts.append(f"类别：{desc.category}")
        if desc.shape_features:
            parts.append(f"形态特征：{' '.join(desc.shape_features)}")
        if desc.decoration_features:
            parts.append(f"纹饰特征：{' '.join(desc.decoration_features)}")
        if desc.color_material:
            parts.append(f"颜色材质：{' '.join(desc.color_material)}")
        if desc.search_keywords:
            parts.append(f"关键词：{' '.join(desc.search_keywords)}")
        if desc.risk:
            parts.append(f"注意：{desc.risk}")

        parts.append("")
        parts.append("规则：")
        parts.append("- 只根据视觉描述的相似度匹配，不依赖年代、历史等非视觉信息")
        parts.append("- 有充分匹配依据时才给出匹配，否则返回无匹配")
        parts.append("- 不要编造文物名称，只使用知识库中已有的标准名称")
        parts.append("- 只返回 JSON，不要额外文字")
        parts.append("")
        parts.append('匹配成功返回：{"match_id":"yingguo_yuying","match_name":"应国玉鹰","confidence":0.85,"evidence":"双翼展开、浅色玉质、线刻纹饰与知识库描述高度吻合"}')
        parts.append('无匹配返回：{"match_id":"none","match_name":"无","confidence":0.0,"evidence":"知识库中无匹配的视觉描述"}')

        return "\n".join(parts)

    def _load_candidates(self, path: Path) -> list[dict[str, Any]]:
        """加载本地候选展品配置。"""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[SEARCH-LOCAL] 候选配置加载失败 path={path} error={exc}", flush=True)
            return []
        if not isinstance(data, list):
            print(f"[SEARCH-LOCAL] 候选配置不是列表 path={path}", flush=True)
            return []
        candidates = [item for item in data if isinstance(item, dict)]
        print(f"[SEARCH-LOCAL] 已加载候选展品 count={len(candidates)} path={path}", flush=True)
        return candidates

    def _search_local(self, desc: VisualDescription) -> tuple[ArtifactMatchResult, ArtifactMatchResult]:
        """基于少量候选展品的本地视觉特征打分。"""
        if not self.candidates:
            return (
                ArtifactMatchResult(evidence="本地候选配置为空", provider="local"),
                ArtifactMatchResult(provider="local"),
            )

        query_text = _normalize_text(
            "\n".join(
                [
                    desc.category,
                    desc.visual_description,
                    " ".join(desc.shape_features),
                    " ".join(desc.decoration_features),
                    " ".join(desc.color_material),
                    " ".join(desc.search_keywords),
                ]
            )
        )

        scored = [self._score_candidate(desc, query_text, candidate) for candidate in self.candidates]
        scored.sort(key=lambda item: item[0].confidence, reverse=True)
        best = scored[0][0]
        runner_up = scored[1][0] if len(scored) > 1 else ArtifactMatchResult(provider="local")
        if best.confidence < _env_float("ARTIFACT_LOCAL_ABS_MIN_CONFIDENCE", 0.35):
            return (
                ArtifactMatchResult(
                    evidence=f"本地匹配置信度过低，最佳候选 {best.match_name}={best.confidence:.2f}",
                    provider="local",
                ),
                runner_up,
            )
        return best, runner_up

    def _score_candidate(
        self,
        desc: VisualDescription,
        query_text: str,
        candidate: dict[str, Any],
    ) -> tuple[ArtifactMatchResult, list[str]]:
        """对单个候选展品打分。"""
        score = 0.0
        evidence: list[str] = []
        category = str(candidate.get("category") or "").strip()

        if desc.category and desc.category not in {"无法判断", "未知"}:
            if category == desc.category:
                score += 0.24
                evidence.append(f"类别匹配:{category}")
            else:
                score -= 0.18

        for term in _candidate_terms(candidate, ("standard_name", "name")):
            if _term_matches(query_text, term):
                score += 0.20
                evidence.append(f"名称:{term}")

        for term in _candidate_terms(candidate, ("aliases",)):
            if _term_matches(query_text, term):
                score += 0.16
                evidence.append(f"别名:{term}")

        for term in _candidate_terms(candidate, ("visual_features",)):
            if _term_matches(query_text, term):
                score += 0.14
                evidence.append(f"视觉:{term}")

        for term in _candidate_terms(candidate, ("local_match_terms",)):
            if _term_matches(query_text, term):
                score += 0.12
                evidence.append(f"同义:{term}")

        for term in _candidate_terms(candidate, ("kb_keywords",)):
            if _term_matches(query_text, term):
                score += 0.07
                evidence.append(f"关键词:{term}")

        for term in _candidate_terms(candidate, ("negative_terms",)):
            if _term_matches(query_text, term):
                score -= 0.22
                evidence.append(f"排除:{term}")

        try:
            priority = float(candidate.get("priority") or 0)
        except (TypeError, ValueError):
            priority = 0.0
        score += min(max(priority, 0.0), 100.0) / 1000.0

        confidence = max(0.0, min(score, 0.98))
        match_id = str(candidate.get("id") or "none").strip() or "none"
        match_name = str(candidate.get("standard_name") or candidate.get("name") or "无").strip() or "无"
        evidence_text = "；".join(evidence[:8]) if evidence else "未命中明确视觉特征"
        return (
            ArtifactMatchResult(
                match_id=match_id,
                match_name=match_name,
                confidence=confidence,
                evidence=evidence_text,
                provider="local",
            ),
            evidence,
        )

    def _parse_response(self, text: str) -> ArtifactMatchResult:
        """解析百炼视觉检索应用的响应。

        尝试从响应中提取 JSON 匹配结果。

        Args:
            text: 百炼应用返回的原始文本

        Returns:
            ArtifactMatchResult: 解析后的匹配结果
        """
        if not text or not text.strip():
            return ArtifactMatchResult(evidence="检索返回为空", raw_response=text)

        cleaned = text.strip()
        # 去除 Markdown 代码块包装
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # 正则提取第一个 JSON 对象
            match = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not match:
                return ArtifactMatchResult(
                    evidence=f"无法解析检索结果: {text[:200]}",
                    raw_response=text,
                )
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return ArtifactMatchResult(
                    evidence=f"JSON 解析失败: {text[:200]}",
                    raw_response=text,
                )

        if not isinstance(data, dict):
            return ArtifactMatchResult(evidence="检索结果格式异常", raw_response=text)

        match_id = str(data.get("match_id") or "none").strip()
        match_name = str(data.get("match_name") or "无").strip()
        if match_id == "none":
            match_name = "无"

        confidence = 0.0
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            pass
        confidence = max(0.0, min(1.0, confidence))

        evidence = str(data.get("evidence") or "").strip()

        return ArtifactMatchResult(
            match_id=match_id,
            match_name=match_name,
            confidence=confidence,
            evidence=evidence,
            raw_response=text,
            provider="bailian",
        )


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _normalize_text(text: str) -> str:
    """归一化中文检索文本。"""
    return re.sub(r"\s+", "", (text or "").lower())


def _candidate_terms(candidate: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    """从候选配置里提取检索词。"""
    terms: list[str] = []
    for key in keys:
        value = candidate.get(key)
        if isinstance(value, str):
            terms.append(value)
        elif isinstance(value, list):
            terms.extend(str(item) for item in value if str(item).strip())
    return terms


def _term_matches(query_text: str, term: str) -> bool:
    """判断一个候选特征词是否命中视觉描述。"""
    normalized = _normalize_text(term)
    if not normalized:
        return False
    # 单字特征词容易误伤普通描述，例如“流”会命中“线条流畅”。
    # 候选库里应使用“长流”“器身有流”“青铜簋”等更稳定的词组。
    if len(normalized) < 2:
        return False
    if normalized in query_text:
        return True
    parts = [
        part
        for part in re.split(r"[、，,;/；\s]|或|和|与|及", normalized)
        if len(part) >= 2
    ]
    if parts and any(part in query_text for part in parts):
        return True
    return False
