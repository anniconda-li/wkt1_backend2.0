"""拍照导游服务模块。

根据视觉识别结果（VisionObservation），生成适合语音播报的导游讲解。

核心决策逻辑（choose_mode_with_reason）：
1. 候选置信度 >= 0.8 + 安全级别 certain/likely → SPECIFIC_MODE（具体讲解）
2. 候选置信度 0.6~0.8 → POSSIBLE_MODE（可能讲解，措辞保守）
3. 有候选但置信度不足或无候选但类别已知 → CATEGORY_MODE（类别引导）
4. 无法识别 + 需要重拍 → RETAKE_MODE（提示重拍）

每个模式都有对应的 LLM 回答策略和本地降级文案。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.bailian_app_service import FALLBACK_TEXT, BailianAppService
from services.vision_service import MuseumVisionCandidate, VisionObservation, load_vision_candidates

# 导游讲解模式
SPECIFIC_MODE = "specific_explain"     # 具体展品讲解（高置信度）
POSSIBLE_MODE = "possible_explain"     # 可能展品讲解（中置信度，保守措辞）
CATEGORY_MODE = "category_guide"       # 类别引导（无具体候选）
RETAKE_MODE = "retake_request"         # 提示重拍
# 知识库无相关内容的标记
NO_KB_MARKER = "知识库无相关内容"

# 各类别的讲解主题提示
CATEGORY_THEMES = {
    "玉器": "应国文化、身份、礼仪、审美",
    "陶瓷": "鲁山花瓷、郏县钧瓷、地方陶瓷文化",
    "青铜器": "古代礼制、贵族生活、应国文化",
    "石器": "早期生产生活、工具痕迹、材质和用途",
    "书画": "题材、笔墨、章法、地方文化记忆",
    "建筑构件": "建筑工艺、装饰寓意、空间礼制",
    "展厅": "展览主题、参观路线、平顶山历史脉络",
}

# 各类别的本地降级讲解文案（LLM 不可用时使用）
LOCAL_CATEGORY_GUIDES = {
    "玉器": "这张照片更像玉器类展品。看玉器，可以先看颜色和温润感，再看造型是不是和身份、礼仪有关。平顶山一带的应国文化里，玉器常能帮助我们理解贵族审美和礼制。要不要再靠近拍一张细节？",
    "陶瓷": "这张照片更像陶瓷类展品。看陶瓷，可以先看器形，再看釉色、纹饰和口沿足底。平顶山周边有鲁山花瓷、郏县钧瓷等陶瓷文化线索，能看出地方工艺的变化。想不想继续听陶瓷怎么看？",
    "青铜器": "这张照片更像青铜器类展品。看青铜器，可以先看器形用途，再看纹饰和锈色。它常和古代礼制、贵族生活、应国文化有关，不一定要认出具体名字，也能读出身份和仪式感。",
    "石器": "这张照片更像石器类展品。看石器，可以注意材质、边缘磨损和形状用途。它们常指向早期生产生活，比如切割、打磨或祭祀场景。要不要换个角度拍清楚轮廓？",
    "书画": "这张照片更像书画类展品。看书画，可以先看题材，再看线条、墨色、留白和题跋印章。即使看不清具体作者，也能从画面气息和内容理解它想表达的文化趣味。",
    "建筑构件": "这张照片更像建筑构件。看这类展品，可以观察纹样、榫卯或装饰位置，想象它原来在建筑中的功能。它往往连接着工艺、礼制和审美。需要我继续讲建筑构件怎么看吗？",
    "展厅": "这张照片更像展厅环境。可以先看展厅主题、展线方向和展柜分布，再选择一个展品靠近拍。平顶山博物馆的参观，可以沿着历史脉络看城市、应国文化和地方工艺。",
}

# 提示重拍的默认文案
RETAKE_ANSWER = "这张照片信息不太够。请把展品放在画面中间，靠近一点，避开展柜反光后重拍。"


@dataclass(frozen=True)
class PhotoGuideResult:
    """拍照导游结果。

    Attributes:
        mode: 讲解模式（specific_explain/possible_explain/category_guide/retake_request）
        grounded: 回答是否基于知识库知识（True）还是降级文案（False）
        answer_text: 适合语音播报的导游回答文本
        gate_reason: 模式选择原因（便于调试）
    """
    mode: str
    grounded: bool
    answer_text: str
    gate_reason: str


class PhotoGuideService:
    """拍照导游服务。

    接收 VisionObservation 视觉结果，通过多级决策选择讲解模式，
    调用 LLM 生成导游讲解，或用本地降级文案兜底。

    Attributes:
        bailian_app_service: 百炼 AI 应用服务（可选，用于调用 LLM）
        candidates: 展品候选列表
        candidates_by_id: 按 ID 索引的候选字典
    """

    def __init__(
        self,
        bailian_app_service: BailianAppService | None = None,
        candidates: list[MuseumVisionCandidate] | None = None,
    ):
        """初始化拍照导游服务。

        Args:
            bailian_app_service: 百炼服务实例，None 时只使用本地降级文案
            candidates: 展品候选列表，None 时从配置文件加载
        """
        self.bailian_app_service = bailian_app_service
        self.candidates = candidates if candidates is not None else load_vision_candidates()
        self.candidates_by_id = {candidate.id: candidate for candidate in self.candidates}

    def build_answer(self, observation: VisionObservation, *, device: str, image_id: str) -> PhotoGuideResult:
        """根据视觉观察结果构建导游回答。

        决策流程（多级降级）：
        1. 判断是否需要重拍 → 直接返回重拍提示
        2. 尝试具体/可能讲解模式（调用 LLM）
           - LLM 返回有效回答 → 直接使用（grounded=True）
           - LLM 不可用 → 本地候选讲解（grounded=False）
        3. 降级到类别引导模式（调用 LLM）
           - LLM 返回有效回答 → 直接使用（grounded=True）
           - LLM 不可用 → 本地类别讲解（grounded=False）

        Args:
            observation: 视觉识别观察结果
            device: 设备标识
            image_id: 图片 ID

        Returns:
            PhotoGuideResult: 导游讲解结果
        """
        mode, gate_reason = choose_mode_with_reason(observation)
        print(f"[CAMERA] 选中模式 mode={mode} gate_reason={gate_reason}", flush=True)

        # 需要重拍
        if mode == RETAKE_MODE:
            return PhotoGuideResult(mode=mode, grounded=False, answer_text=RETAKE_ANSWER, gate_reason=gate_reason)

        # 尝试具体/可能展品讲解
        if mode in {SPECIFIC_MODE, POSSIBLE_MODE}:
            candidate_answer = self._ask_candidate(observation, mode=mode, device=device, image_id=image_id)
            if _has_grounded_answer(candidate_answer):
                return PhotoGuideResult(
                    mode=mode,
                    grounded=True,
                    answer_text=_clean_answer(candidate_answer),
                    gate_reason=gate_reason,
                )
            # LLM 不可用，使用本地文案
            if self.bailian_app_service is None:
                return PhotoGuideResult(
                    mode=mode,
                    grounded=False,
                    answer_text=_local_candidate_answer(observation, mode),
                    gate_reason="降级 本地候选讲解",
                )

        # 降级到类别引导
        category_answer = self._ask_category(observation, device=device, image_id=image_id)
        if _has_grounded_answer(category_answer):
            return PhotoGuideResult(
                mode=CATEGORY_MODE,
                grounded=True,
                answer_text=_clean_answer(category_answer),
                gate_reason="候选回答不可用，降级到类别引导",
            )
        return PhotoGuideResult(
            mode=CATEGORY_MODE,
            grounded=False,
            answer_text=LOCAL_CATEGORY_GUIDES.get(observation.category, RETAKE_ANSWER),
            gate_reason="降级 本地类别讲解",
        )

    def _ask_candidate(self, observation: VisionObservation, *, mode: str, device: str, image_id: str) -> str:
        """调用 LLM 进行具体/可能展品讲解。

        根据匹配置信度调整措辞保守程度：
        - SPECIFIC_MODE: "很可能是"，不建议绝对确定
        - POSSIBLE_MODE: "很像/可能是"，建议以现场说明为准

        Args:
            observation: 视觉观察结果
            mode: 讲解模式
            device: 设备标识
            image_id: 图片 ID

        Returns:
            str: LLM 生成的讲解文本，服务不可用时返回空字符串
        """
        if self.bailian_app_service is None:
            return ""
        candidate = self.candidates_by_id.get(observation.best_candidate_id)
        keywords = "、".join(candidate.kb_keywords if candidate else [observation.best_candidate_name])
        caution = (
            '这次匹配置信度较高，可以说"很可能是"，但仍建议不要说成绝对确定。'
            if mode == SPECIFIC_MODE
            else '这次匹配置信度中等，请使用"很像/可能是/建议以现场说明为准"这类保守措辞。'
        )
        prompt = (
            "游客拍照识别结果："
            f"候选展品：{observation.best_candidate_name}；"
            f"匹配置信度：{observation.candidate_confidence:.2f}；"
            f"视觉依据：{'、'.join(observation.visual_evidence) or '无'}；"
            f"风险：{observation.risk or '无'}；"
            f"知识库关键词：{keywords}。"
            "请根据平顶山市博物馆知识库，用自然中文给游客做简短讲解。"
            f"{caution}"
            "不要编造知识库没有的年代、出土地点、展柜位置。"
            "如果知识库没有相关内容，请只回复：知识库无相关内容。"
            "回答适合语音播报，50到120字，不要Markdown，不要项目符号。"
        )
        return self.bailian_app_service.ask(prompt)

    def _ask_category(self, observation: VisionObservation, *, device: str, image_id: str) -> str:
        """调用 LLM 进行类别引导讲解。

        当无法确认具体展品时，引导游客从类别角度欣赏。

        Args:
            observation: 视觉观察结果
            device: 设备标识
            image_id: 图片 ID

        Returns:
            str: LLM 生成的类别引导文本
        """
        if self.bailian_app_service is None:
            return ""
        themes = CATEGORY_THEMES.get(observation.category, "平顶山博物馆展览主题")
        prompt = (
            "游客拍到的具体文物名称不能可靠确认，不能编造具体文物名称。"
            f'请围绕"{observation.category}"这类展品讲怎么看，并尽量结合知识库相关主题：{themes}。'
            f"照片可见特征：{'、'.join(observation.visible_features) or '无'}。"
            f"不确定风险：{observation.risk or '无'}。"
            "回答适合语音播报，50到120字，不要Markdown，不要项目符号，不要说识别失败。"
        )
        return self.bailian_app_service.ask(prompt)


def choose_mode(observation: VisionObservation) -> str:
    """根据视觉观察结果选择导游讲解模式。

    便捷函数，仅返回模式名称。

    Args:
        observation: 视觉观察结果

    Returns:
        str: 讲解模式
    """
    return choose_mode_with_reason(observation)[0]


def choose_mode_with_reason(observation: VisionObservation) -> tuple[str, str]:
    """根据视觉观察结果选择讲解模式并附带原因。

    决策规则（按优先级）：
    1. 有具体候选 + 置信度 >= 0.8 + 安全级别 certain/likely → SPECIFIC_MODE
    2. 有具体候选 + 置信度 0.6~0.8 → POSSIBLE_MODE
    3. 类别已知（非"未知"）→ CATEGORY_MODE
    4. 无法识别 + need_retake=true → RETAKE_MODE
    5. 默认 → RETAKE_MODE

    Args:
        observation: 视觉观察结果

    Returns:
        tuple[str, str]: (讲解模式, 选择原因)
    """
    if (
        observation.best_candidate_id != "none"
        and observation.candidate_confidence >= 0.8
        and observation.safe_answer_level in {"certain", "likely"}
    ):
        return SPECIFIC_MODE, "候选置信度 >= 0.80 且安全级别为 certain/likely"
    if observation.best_candidate_id != "none" and 0.6 <= observation.candidate_confidence < 0.8:
        return POSSIBLE_MODE, "候选置信度在 0.60 到 0.80 之间"
    if observation.category != "未知":
        return CATEGORY_MODE, "无可信候选但类别已知"
    if observation.best_candidate_id == "none" and observation.category == "未知" and observation.need_retake:
        return RETAKE_MODE, "无候选、未知类别且需要重拍"
    return RETAKE_MODE, "图片信息不足"


def response_payload(
    *,
    device: str,
    image_id: str,
    observation: VisionObservation,
    guide: PhotoGuideResult,
) -> dict[str, Any]:
    """构建相机分析接口的响应体 JSON。

    合并 VisionObservation 的关键字段和 PhotoGuideResult 的回答信息。

    Args:
        device: 设备标识
        image_id: 图片 ID
        observation: 视觉观察结果
        guide: 导游讲解结果

    Returns:
        dict: 完整的 API 响应字典
    """
    data = observation.to_dict()
    return {
        "ok": True,
        "device": device,
        "image_id": image_id,
        "mode": guide.mode,
        "best_candidate_id": data["best_candidate_id"],
        "best_candidate_name": data["best_candidate_name"],
        "candidate_confidence": data["candidate_confidence"],
        "category": data["category"],
        "top_candidates": data["top_candidates"],
        "visible_features": data["visible_features"],
        "visual_evidence": data["visual_evidence"],
        "risk": data["risk"],
        "safe_answer_level": data["safe_answer_level"],
        "need_retake": data["need_retake"] or guide.mode == RETAKE_MODE,
        "answer_text": guide.answer_text,
        "grounded": guide.grounded,
        "gate_reason": guide.gate_reason,
        # 兼容旧版客户端/调试工具
        "scene_type": data["scene_type"],
        "object_category": data["object_category"],
        "visual_features": data["visual_features"],
        "readable_text": data["readable_text"],
        "possible_subject": data["possible_subject"],
        "category_confidence": data["category_confidence"],
        "specific_name_confidence": data["specific_name_confidence"],
    }


def _has_grounded_answer(answer: str) -> bool:
    """检查 LLM 回答是否有效（非空、非降级文本、非"知识库无相关内容"）。

    Args:
        answer: LLM 回答文本

    Returns:
        bool: 回答是否有效
    """
    cleaned = _clean_answer(answer)
    if not cleaned or cleaned == FALLBACK_TEXT:
        return False
    return NO_KB_MARKER not in cleaned


def _clean_answer(answer: str) -> str:
    """清洗回答文本：去除首尾空格，合并多余空白。

    Args:
        answer: 原始回答文本

    Returns:
        str: 清洗后的文本
    """
    return " ".join((answer or "").strip().split())


def _local_candidate_answer(observation: VisionObservation, mode: str) -> str:
    """生成本地降级候选展品讲解（不依赖 LLM）。

    当 LLM 不可用时，根据视觉证据拼接简单的讲解文本。

    Args:
        observation: 视觉观察结果
        mode: 讲解模式

    Returns:
        str: 本地生成的讲解文本
    """
    evidence = "、".join(observation.visual_evidence[:3]) or "外形特征"
    if mode == SPECIFIC_MODE:
        prefix = f"这件展品很可能是{observation.best_candidate_name}"
    else:
        prefix = f"这件展品很像{observation.best_candidate_name}，但图片细节还不够清楚，建议以现场说明为准"
    return f"{prefix}。从照片看，主要依据是{evidence}。你可以先关注它的材质、造型和纹饰，再结合展签确认具体名称。"
