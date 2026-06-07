"""AI 会话取消功能单元测试。

测试 /ai/cancel 接口的各种场景：
- 立即取消应在 result_info 中反映
- 部分上传后取消应拒绝后续上传
- 取消后的 finish 应被忽略
- 文本就绪后取消应保留文本但隐藏音频
- 音频就绪后取消应拒绝 result_chunk
- 取消操作幂等性和未知会话处理
"""

from __future__ import annotations

import os
import struct
import sys
import unittest
import wave
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.walkie_app import create_http_app


def make_wav() -> bytes:
    """生成一段静音 WAV 音频用于测试（16kHz 单声道 16-bit）。"""
    pcm = struct.pack("<" + "h" * 1600, *([0] * 1600))
    out = BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm)
    return out.getvalue()


class AiCancelTest(unittest.TestCase):
    """AI 取消功能测试套件。"""

    def setUp(self) -> None:
        """每个测试用例前的初始化：设置固定回复模式并创建测试客户端。"""
        os.environ["TOUR_MODE"] = "fixed"
        os.environ["TTS_PROVIDER"] = "mock"
        os.environ["ASR_PROVIDER"] = "mock"
        os.environ["AUTO_TTS_BACKGROUND"] = "false"
        root = Path("tmp/debug/test_ai_cancel")
        self.app = create_http_app(root / "wav", root / "jpg", 1, False)
        self.client = TestClient(self.app)
        self.wav = make_wav()

    def start(self) -> str:
        """创建新会话并返回 session_id。"""
        response = self.client.post("/ai/start", json={"device": "test-device"})
        self.assertEqual(response.status_code, 200)
        return response.json()["session"]

    def upload_full(self, session: str) -> None:
        """一次性上传完整 WAV 数据。"""
        response = self.client.post(
            f"/ai/upload?session={session}&index=0&offset=0&total={len(self.wav)}",
            content=self.wav,
        )
        self.assertEqual(response.status_code, 200)

    def test_cancel_immediately_is_reflected_in_result_info(self) -> None:
        """测试：立即取消后 result_info 应反映取消状态。"""
        session = self.start()
        response = self.client.post(f"/ai/cancel?session={session}", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "canceled")

        info = self.client.post(f"/ai/result_info?session={session}", json={}).json()
        self.assertEqual(info["status"], "canceled")
        self.assertFalse(info["audio_ready"])
        self.assertFalse(info["reply_wav_ready"])
        self.assertEqual(info["tts_status"], "canceled")

    def test_cancel_after_partial_upload_rejects_later_upload(self) -> None:
        """测试：部分上传后取消，后续上传应被拒绝（409）。"""
        session = self.start()
        first = self.wav[:100]
        response = self.client.post(
            f"/ai/upload?session={session}&index=0&offset=0&total={len(self.wav)}",
            content=first,
        )
        self.assertEqual(response.status_code, 200)
        self.client.post(f"/ai/cancel?session={session}", json={})

        response = self.client.post(
            f"/ai/upload?session={session}&index=1&offset=100&total={len(self.wav)}",
            content=self.wav[100:200],
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["status"], "canceled")

    def test_finish_after_cancel_is_ignored(self) -> None:
        """测试：取消后的 finish 应被忽略，返回 canceled 状态。"""
        session = self.start()
        self.upload_full(session)
        self.client.post(f"/ai/cancel?session={session}", json={})

        response = self.client.post(f"/ai/finish?session={session}", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "canceled")

    def test_cancel_after_text_ready_keeps_text_but_hides_audio(self) -> None:
        """测试：文本就绪后取消应保留 answer_text 但清除音频状态。"""
        os.environ["AUTO_TTS_BACKGROUND"] = "false"
        session = self.start()
        self.upload_full(session)
        response = self.client.post(f"/ai/finish?session={session}", json={})
        self.assertEqual(response.status_code, 200)

        before_cancel = self.client.post(f"/ai/result_info?session={session}", json={}).json()
        self.assertEqual(before_cancel["status"], "text_ready")
        self.assertTrue(before_cancel["answer_text"])

        self.client.post(f"/ai/cancel?session={session}", json={})
        after_cancel = self.client.post(f"/ai/result_info?session={session}", json={}).json()
        self.assertEqual(after_cancel["status"], "canceled")
        self.assertEqual(after_cancel["answer_text"], before_cancel["answer_text"])
        self.assertFalse(after_cancel["audio_ready"])
        self.assertEqual(after_cancel["tts_status"], "canceled")

    def test_cancel_after_audio_ready_rejects_result_chunk(self) -> None:
        """测试：音频就绪后取消应拒绝 result_chunk 请求（409）。"""
        session = self.start()
        # 手动设置会话为音频就绪状态
        with self.app.state.ai_sessions_lock:
            ai_session = self.app.state.ai_sessions[session]
            ai_session.reply = self.wav
            ai_session.status = "audio_ready"
            ai_session.audio_ready = True
            ai_session.reply_wav_ready = True
            ai_session.reply_wav_size = len(self.wav)
            ai_session.tts_status = "done"

        response = self.client.post(f"/ai/cancel?session={session}", json={})
        self.assertEqual(response.status_code, 200)
        response = self.client.post(f"/ai/result_chunk?session={session}&offset=0&len=64", json={})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["status"], "canceled")

    def test_cancel_is_idempotent_and_unknown_session_returns_not_found(self) -> None:
        """测试：取消操作幂等性，以及未知会话返回 not_found。"""
        session = self.start()
        first = self.client.post(f"/ai/cancel?session={session}", json={})
        second = self.client.post(f"/ai/cancel?session={session}", json={})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["status"], "canceled")

        unknown = self.client.post("/ai/cancel?session=missing", json={})
        self.assertEqual(unknown.status_code, 200)
        self.assertEqual(unknown.json()["status"], "not_found")


if __name__ == "__main__":
    unittest.main()
