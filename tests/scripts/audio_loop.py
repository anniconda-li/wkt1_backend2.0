"""音频环回测试脚本。

验证完整的 TTS → ASR → LLM → TTS 链路：
1. 将输入文本合成 TTS 音频（或使用已有 WAV）
2. 对音频进行 ASR 识别
3. 调用 LLM 获取回答
4. 将回答合成 TTS 音频

所有中间结果保存到输出目录，用于调试和验证各环节正确性。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import wave
from pathlib import Path

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.config  # noqa: F401 - 加载项目 .env 环境变量
from core.paths import TMP_DEBUG_AUDIO_DIR
from services.asr_service import transcribe_wav
from services.bailian_app_service import BailianAppService
from services.tts_service import synthesize_wav_16k

# 默认测试问题和答案
DEFAULT_QUESTION = "大雁塔有什么故事？"
MOCK_BAILIAN_ANSWER = "大雁塔是西安著名古迹，始建于唐代，最初用于保存玄奘从印度带回的佛经和佛像。"
DEFAULT_OUTPUT_DIR = TMP_DEBUG_AUDIO_DIR / "audio_loop"


def validate_esp_wav(path: Path) -> dict[str, object]:
    """验证 WAV 文件是否符合对讲机设备的格式要求。

    要求：单声道、16-bit、16kHz、无压缩。

    Returns:
        dict: WAV 音频参数信息

    Raises:
        RuntimeError: 格式不符合要求
    """
    with wave.open(str(path), "rb") as wf:
        info = {
            "channels": wf.getnchannels(),
            "sample_width": wf.getsampwidth(),
            "sample_rate": wf.getframerate(),
            "frames": wf.getnframes(),
            "duration": wf.getnframes() / wf.getframerate() if wf.getframerate() else 0.0,
            "compression": wf.getcomptype(),
            "compression_name": wf.getcompname(),
        }

    if info["channels"] != 1:
        raise RuntimeError(f"WAV 声道数必须为 1，当前 {info['channels']}")
    if info["sample_width"] != 2:
        raise RuntimeError(f"WAV 采样位宽必须为 2，当前 {info['sample_width']}")
    if info["sample_rate"] != 16000:
        raise RuntimeError(f"WAV 采样率必须为 16000，当前 {info['sample_rate']}")
    if info["compression"] != "NONE":
        raise RuntimeError(f"WAV 压缩格式必须为 NONE，当前 {info['compression']}")
    return info


def print_wav_info(label: str, path: Path) -> None:
    """打印 WAV 文件信息，带标签前缀。"""
    info = validate_esp_wav(path)
    print(
        f"[{label}] format: "
        f"channels={info['channels']} "
        f"sample_width={info['sample_width']} "
        f"sample_rate={info['sample_rate']} "
        f"frames={info['frames']} "
        f"duration={info['duration']:.3f}s "
        f"compression={info['compression']} ({info['compression_name']})"
    )


def get_bailian_answer(asr_text: str) -> str:
    """调用百炼 AI 获取回答。"""
    return BailianAppService().ask(asr_text)


def prepare_output_dir(path_text: str) -> Path:
    """准备输出目录：创建并清空已有内容。

    Args:
        path_text: 目录路径（支持相对路径）

    Returns:
        Path: 已清空的绝对路径目录
    """
    output_dir = Path(path_text)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # 清空目录
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return output_dir


def main() -> None:
    """主函数：运行音频环回测试。"""
    parser = argparse.ArgumentParser(description="运行本地 TTS → ASR → 百炼 → TTS 环回测试")
    parser.add_argument("--text", default=DEFAULT_QUESTION, help="输入问题文本")
    parser.add_argument("--wav", default="", help="已有问题 WAV 路径，提供后跳过问题 TTS")
    parser.add_argument("--answer", default="", help="手动指定回答文本，提供后跳过百炼调用")
    parser.add_argument("--mock-bailian", action="store_true", help="使用本地 mock 回答")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)), help="输出目录")
    args = parser.parse_args()

    output_dir = prepare_output_dir(args.out_dir)
    # 定义输出文件路径
    question_text_path = output_dir / "question.txt"
    asr_text_path = output_dir / "asr_text.txt"
    answer_text_path = output_dir / "answer.txt"
    question_wav_path = output_dir / "question.wav"
    reply_wav_path = output_dir / "reply.wav"

    # 第 1 步：问题文本 → TTS 音频
    question_text = args.text.strip() or DEFAULT_QUESTION
    question_text_path.write_text(question_text, encoding="utf-8")

    if args.wav:
        question_wav_path = Path(args.wav)
        if not question_wav_path.exists():
            raise FileNotFoundError(question_wav_path)
        validate_esp_wav(question_wav_path)
    else:
        question_wav_bytes = synthesize_wav_16k(question_text)
        question_wav_path.write_bytes(question_wav_bytes)
        validate_esp_wav(question_wav_path)

    print(f"[QUESTION] text: {question_text}")
    print(f"[QUESTION] wav: {question_wav_path}")
    print_wav_info("QUESTION", question_wav_path)

    # 第 2 步：ASR 识别
    asr_text = transcribe_wav(question_wav_path)
    asr_text_path.write_text(asr_text, encoding="utf-8")
    print(f"[ASR] text: {asr_text}")

    # 第 3 步：LLM 回答
    if args.answer.strip():
        answer_text = args.answer.strip()
    elif args.mock_bailian:
        answer_text = MOCK_BAILIAN_ANSWER
    else:
        answer_text = get_bailian_answer(asr_text)

    answer_text_path.write_text(answer_text, encoding="utf-8")
    print(f"[BAILIAN] answer: {answer_text}")

    # 第 4 步：回答文本 → TTS 音频
    reply_wav_bytes = synthesize_wav_16k(answer_text)
    reply_wav_path.write_bytes(reply_wav_bytes)
    validate_esp_wav(reply_wav_path)
    print(f"[TTS] reply wav: {reply_wav_path}")
    print_wav_info("TTS", reply_wav_path)
    print("[OK] 音频环回测试通过")


if __name__ == "__main__":
    main()
