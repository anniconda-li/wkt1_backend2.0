"""WAV 流水线调试脚本。

从本地 WAV 文件运行 ASR → LLM → TTS 完整链路调试。
每一步都有独立的耗时统计和错误处理。

用法：
    python tests/scripts/wav_pipeline.py --wav path/to/audio.wav --device debug-server
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import core.config  # noqa: F401 - 加载项目 .env 环境变量
from core.paths import TEST_AUDIO_DIR, TMP_DEBUG_AUDIO_DIR
from services.asr_service import transcribe_wav
from services.bailian_app_service import BailianAppService, FALLBACK_TEXT
from services.tts_service import synthesize_wav_16k

# 默认参数
DEFAULT_WAV_PATH = TEST_AUDIO_DIR / "sample_ai_upload_20260531_124142_554048.wav"
DEFAULT_OUTPUT_DIR = TMP_DEBUG_AUDIO_DIR
DEFAULT_DEVICE = "debug-server"
DEFAULT_SPOT_ID = "dayanta"
DEFAULT_MODE = "debug_wav_bailian_app"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="从本地 WAV 文件调试 ASR → LLM → TTS 流水线")
    parser.add_argument("--wav", default=str(DEFAULT_WAV_PATH), help="输入 WAV 文件路径")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="调试设备名")
    parser.add_argument("--spot-id", default=DEFAULT_SPOT_ID, help="调试景点 ID")
    parser.add_argument("--mode", default=DEFAULT_MODE, help="调试模式标签")
    return parser.parse_args()


def main() -> int:
    """主函数：运行 WAV 流水线调试。

    流程：
    1. 验证输入 WAV 存在
    2. ASR 语音识别
    3. LLM 问答（百炼 AI）
    4. TTS 语音合成
    5. 保存输出 WAV

    Returns:
        int: 0 成功，1 失败（ASR/TTS），2 文件不存在
    """
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    total_start = time.perf_counter()

    # 验证输入文件
    wav_path = Path(args.wav)
    print(f"[DEBUG] input_wav={wav_path}")
    print(f"[DEBUG] wav_exists={wav_path.exists()}")
    if not wav_path.exists():
        print(f"[ERROR] WAV 文件不存在: {wav_path}", file=sys.stderr)
        return 2
    print(f"[DEBUG] wav_size={wav_path.stat().st_size}")
    print(f"[DEBUG] device={args.device} spot_id={args.spot_id} mode={args.mode}")

    # 第 1 步：ASR 识别
    asr_start = time.perf_counter()
    try:
        asr_text = transcribe_wav(wav_path)
    except Exception:
        logger.exception("ASR 失败，停止在 LLM 之前")
        print(f"[DEBUG-TIME] asr={time.perf_counter() - asr_start:.3f}s error=asr_failed")
        print(f"[DEBUG-TIME] total={time.perf_counter() - total_start:.3f}s")
        return 1
    print(f"[DEBUG] asr_text={asr_text}")
    print(f"[DEBUG-TIME] asr={time.perf_counter() - asr_start:.3f}s")

    print("[DEBUG] llm_provider=bailian_app")

    # 第 2 步：LLM 问答
    llm_start = time.perf_counter()
    try:
        answer_text = BailianAppService().ask(asr_text)
    except Exception:
        logger.exception("LLM 失败，使用降级文本")
        answer_text = FALLBACK_TEXT
        print(f"[DEBUG] llm_fallback_text={answer_text}")
    if answer_text == FALLBACK_TEXT:
        print(f"[DEBUG] llm_fallback_text={answer_text}")
    print(f"[DEBUG] answer_text={answer_text}")
    print(f"[DEBUG-TIME] llm={time.perf_counter() - llm_start:.3f}s")

    # 第 3 步：TTS 合成
    tts_start = time.perf_counter()
    try:
        reply_wav = synthesize_wav_16k(answer_text)
    except Exception:
        logger.exception("TTS 失败")
        print(f"[DEBUG-TIME] tts={time.perf_counter() - tts_start:.3f}s error=tts_failed")
        print(f"[DEBUG-TIME] total={time.perf_counter() - total_start:.3f}s")
        return 1
    print(f"[DEBUG-TIME] tts={time.perf_counter() - tts_start:.3f}s")

    # 保存输出
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_OUTPUT_DIR / f"debug_reply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    output_path.write_bytes(reply_wav)
    print(f"[DEBUG] output_wav={output_path}")
    print(f"[DEBUG] output_size={output_path.stat().st_size}")
    print(f"[DEBUG-TIME] total={time.perf_counter() - total_start:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
