"""语音问答服务模块。

将 ASR（语音识别）→ LLM（大模型问答）→ TTS（语音合成）串联为完整的语音问答链路。

支持两种模式：
- "fixed": 固定回复模式，返回预设文本（用于连通性测试）
- "asr_bailian_app": 完整的 ASR + 百炼 AI 问答流程
"""

from __future__ import annotations

import time
from pathlib import Path

from services.asr_service import transcribe_wav
from services.bailian_app_service import BailianAppService
from services.tts_service import synthesize_wav_16k

# 固定回复模式下的默认回答文本
FIXED_ANSWER = "你好，我是景区导游助手。当前语音回复链路已经打通。"


class VoiceQaService:
    """语音问答服务。

    将音频输入经过 ASR 识别、LLM 处理和 TTS 合成，
    返回文本答案和对应的 WAV 音频。

    Attributes:
        bailian_app_service: 百炼 AI 应用服务实例
    """

    def __init__(self, bailian_app_service: BailianAppService | None = None):
        """初始化语音问答服务。

        Args:
            bailian_app_service: 百炼服务实例，为 None 时仅支持 fixed 模式
        """
        self.bailian_app_service = bailian_app_service

    def process_session(
        self,
        wav_path: str | Path,
        device: str = "walkie-01",
        spot_id: str = "dayanta",
        image_context: str = "",
        mode: str = "fixed",
    ) -> tuple[str, bytes]:
        """处理一个完整的问答会话（语音输入 → 文本回答 → 语音输出）。

        Args:
            wav_path: 用户上传的 WAV 音频文件路径
            device: 设备标识（默认 walkie-01）
            spot_id: 景点标识（默认 dayanta）
            image_context: 图片上下文信息
            mode: 处理模式（"fixed" 或 "asr_bailian_app"）

        Returns:
            tuple[str, bytes]: (回答文本, 回答音频 WAV 数据)
        """
        _asr_text, answer_text = self.process_text_session(
            wav_path,
            device=device,
            spot_id=spot_id,
            image_context=image_context,
            mode=mode,
        )

        # TTS 语音合成
        tts_start = time.perf_counter()
        try:
            reply_wav = synthesize_wav_16k(answer_text)
        except Exception as exc:
            print(f"[AI-TIME] tts={time.perf_counter() - tts_start:.3f}s error={exc}", flush=True)
            raise RuntimeError(f"TTS 失败: {exc}") from exc
        print(f"[AI-TIME] tts={time.perf_counter() - tts_start:.3f}s reply_bytes={len(reply_wav)}", flush=True)
        return answer_text, reply_wav

    def process_text_session(
        self,
        wav_path: str | Path,
        device: str = "walkie-01",
        spot_id: str = "dayanta",
        image_context: str = "",
        mode: str = "fixed",
    ) -> tuple[str, str]:
        """处理问答会话的文本部分（ASR → LLM）。

        只进行语音识别和文本问答，不进行 TTS 合成。

        Args:
            wav_path: 用户上传的 WAV 音频文件路径
            device: 设备标识
            spot_id: 景点标识
            image_context: 图片上下文信息
            mode: 处理模式

        Returns:
            tuple[str, str]: (ASR 识别文本, AI 回答文本)
        """
        wav_path = Path(wav_path)
        print(
            f"[VoiceQaService] process wav={wav_path} device={device} "
            f"spot_id={spot_id} mode={mode} llm_provider=bailian_app",
            flush=True,
        )

        if mode == "fixed":
            # 固定回复模式：不进行实际 ASR，返回预设文本
            asr_text = ""
            answer_text = FIXED_ANSWER
        elif mode == "asr_bailian_app":
            # 完整 ASR + LLM 模式
            asr_start = time.perf_counter()
            try:
                asr_text = transcribe_wav(wav_path)
            except Exception as exc:
                print(f"[AI-TIME] asr={time.perf_counter() - asr_start:.3f}s error={exc}", flush=True)
                raise RuntimeError(f"ASR 失败: {exc}") from exc
            print(f"[AI] asr_text: {asr_text}", flush=True)
            print(f"[AI-TIME] asr={time.perf_counter() - asr_start:.3f}s text_chars={len(asr_text)}", flush=True)
            answer_text = self._ask_llm(
                asr_text,
                device=device,
                spot_id=spot_id,
                image_context=image_context,
            )
            print(f"[AI] answer_text chars: {len(answer_text)}", flush=True)
        else:
            raise ValueError(f"不支持的 TOUR_MODE: {mode}")

        return asr_text, answer_text

    def _ask_llm(
        self,
        question: str,
        *,
        device: str,
        spot_id: str,
        image_context: str,
    ) -> str:
        """调用百炼 LLM 进行问答。

        Args:
            question: 用户问题文本
            device: 设备标识
            spot_id: 景点标识
            image_context: 图片上下文信息

        Returns:
            str: AI 回答文本

        Raises:
            RuntimeError: 百炼服务未配置或调用失败
        """
        if self.bailian_app_service is None:
            raise RuntimeError("百炼应用服务未配置")
        bailian_start = time.perf_counter()
        try:
            answer_text = self.bailian_app_service.ask(question)
        except Exception as exc:
            print(f"[AI-TIME] bailian_app={time.perf_counter() - bailian_start:.3f}s error={exc}", flush=True)
            raise RuntimeError(f"百炼应用调用失败: {exc}") from exc
        print(
            f"[AI-TIME] bailian_app={time.perf_counter() - bailian_start:.3f}s "
            f"answer_chars={len(answer_text)}",
            flush=True,
        )
        return answer_text
