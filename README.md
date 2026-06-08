# WKT1 AI Guide Backend

ESP32S3 AI 对讲导游设备的本地 FastAPI 后端。

当前项目已经完成三条核心链路的本地验证：

```text
1. UDP 实时音频对讲
2. 语音问答：WAV -> ASR -> 百炼知识库 -> TTS -> WAV
3. 图片问答：JPG -> 视觉描述 -> 百炼知识库 -> 文本/语音回答
```

设备侧负责录音、拍照、上传 WAV/JPG、拉取回复 WAV 并播放。后端负责设备协议接收、ASR、视觉识别、百炼应用调用、TTS 和运行时文件管理。百炼应用负责文物知识库检索和文本回答。

## 目录结构

```text
core/       项目基础配置：加载 .env、统一路径常量、创建运行目录
server/     生产服务入口：FastAPI app、UDP 服务循环、协议和媒体解析
services/   正式业务能力：ASR、TTS、百炼、视觉、语音问答、拍照问答
knowledge/  文物知识库资料：参考图、候选配置、每件文物 Markdown
tests/      测试脚本和测试数据
tools/      维护和调试工具：清理 tmp、构建知识库、检查参考图、端到端调试
tmp/        运行时临时产物，可清理
```

`tmp/` 只保存运行时文件，长期保留的测试图片、测试音频放到 `tests/data/camera/` 和 `tests/data/audio/`，文物资料放到 `knowledge/`。

## 环境配置

创建 `.env`：

```powershell
copy .env.example .env
```

主要配置：

```text
DASHSCOPE_API_KEY=your_dashscope_api_key
TTS_PROVIDER=dashscope
TTS_MODEL=qwen3-tts-flash
TTS_VOICE=Cherry

ASR_PROVIDER=dashscope
ASR_MODEL=paraformer-realtime-v2

VISION_PROVIDER=dashscope
VISION_MODEL=qwen-vl-plus

BAILIAN_API_KEY=your_bailian_api_key
BAILIAN_APP_ID=your_bailian_app_id
BAILIAN_APP_BASE_URL=https://dashscope.aliyuncs.com
BAILIAN_TIMEOUT=15
AUTO_TTS_BACKGROUND=true
ENABLE_DEBUG_ROUTES=false
```

项目入口会通过 `core/config.py` 自动加载根目录 `.env`。

安装依赖：

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

本机还需要可运行 `ffmpeg`，用于把 TTS 音频转换为设备可播放格式：

```text
16000Hz / 16-bit / mono / PCM / WAV
```

## 启动服务

HTTP 服务可以直接交给 uvicorn 管理，适合部署到服务器、进程管理器或反向代理后面：

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

这个命令只启动 HTTP API：

```text
HTTP: http://<PC_LAN_IP>:8000
```

健康检查：

```text
GET /healthz
GET /readyz
```

如果设备还需要 UDP 实时对讲服务，使用完整设备入口：

```powershell
.\.venv\Scripts\python.exe -m server.walkie_app --host 0.0.0.0 --http-port 8000 --udp-port 9000
```

服务启动后：

```text
HTTP: http://<PC_LAN_IP>:8000
UDP:  9000
```

ESP32S3 固件中的 HTTP/UDP 地址应指向这台服务器的局域网 IP。

生产环境默认不暴露调试接口。确实需要临时调试时再设置：

```text
ENABLE_DEBUG_ROUTES=true
```

开启后才会注册 `/debug/camera_guide/test`。

## 实时对讲

后端保留 UDP WTK1 数据包接收和同设备音频回传能力，用于实时音频对讲测试。

默认端口：

```text
UDP 9000
```

设备侧需要保持 WTK1 UDP 协议格式。后端会记录注册、频道、PTT、音频、心跳等包，并可做本地回声测试。

## 语音问答

设备语音问答流程：

```text
POST /ai/start
POST /ai/upload
POST /ai/finish
POST /ai/result_info
POST /ai/result_chunk
```

后端处理流程：

```text
上传 WAV
-> ASR 转文字
-> 判断是否是图片相关问题
-> 普通问题走百炼知识库
-> TTS 合成回复 WAV
-> 设备分片下载播放
```

本地完整音频闭环测试：

```powershell
python tests\scripts\audio_loop.py --text "大雁塔有什么故事？"
```

使用 mock 百炼回答：

```powershell
python tests\scripts\audio_loop.py --text "大雁塔有什么故事？" --mock-bailian
```

单独测试百炼应用：

```powershell
python tests\scripts\bailian_app.py
```

## 图片问答

图片上传接口：

```text
POST /camera/upload?device=walkie-01
Content-Type: image/jpeg
Body: JPEG bytes
```

图片上传不是单纯保存文件。后端会同步完成初步视觉分析，并返回是否可以进入语音提问。

成功且可提问：

```json
{
  "ok": true,
  "analysis_ok": true,
  "device": "walkie-01",
  "image_id": "camera_upload_...",
  "scene_type": "展柜展品",
  "object_category": "玉器",
  "mode": "category_guide",
  "need_retake": false
}
```

照片不够清楚时：

```json
{
  "ok": true,
  "analysis_ok": false,
  "mode": "retake_request",
  "need_retake": true,
  "answer_text": "这张照片信息不太够。请把展品放在画面中间，靠近一点，避开展柜反光后重拍。"
}
```

客户端建议状态机：

```text
idle
-> 用户拍照
-> uploading_photo
-> 等待 /camera/upload 返回

ok=false
-> 提示上传失败
-> 回到 idle

ok=true 且 analysis_ok=false
-> 播放或显示 answer_text
-> 要求用户重拍
-> 回到 idle

ok=true 且 analysis_ok=true
-> 设置 camera_ready=true
-> 提示“已完成图像分析，可以提问了。”
-> 等待用户按语音键提问
```

用户随后继续走语音问答接口。后端会判断用户是否在问刚才拍的图片，例如：

```text
这是什么
讲讲这个展品
刚才拍的是什么
```

如果是图片相关问题，后端会使用 `/camera/upload` 时缓存的视觉识别结果回答，不会重复识别图片。如果是普通导游问题，则继续走普通语音问答。

固定图片 + 固定问题测试：

```powershell
python tests\scripts\camera_guide.py --image tests\data\camera\test_exhibit.jpg --text "这是什么"
```

这个测试会输出：

```text
vision_result
rewritten_prompt
bailian_answer
timing
```

可以用它检查视觉描述、发给百炼的 prompt 和最终回答是否符合预期。

## 知识库资料

知识库相关文件集中在 `knowledge/`：

```text
knowledge/config/   文物候选配置和视觉索引 JSON
knowledge/refs/     标准参考图片
knowledge/exhibits/ 每件文物一个 Markdown，上传到百炼知识库
```

检查参考图片：

```powershell
python tools\check_museum_refs.py
```

根据标准参考图生成/导出文物 Markdown：

```powershell
python tools\build_exhibit_knowledge.py --overwrite
```

只根据已有视觉索引重新导出 Markdown：

```powershell
python tools\build_exhibit_knowledge.py --export-only
```

## tmp 清理

清理运行时临时产物：

```powershell
python tools\clean_tmp.py
```

清理后保留的运行时目录：

```text
tmp/audio/received/
tmp/audio/replies/
tmp/camera/received/
tmp/camera/preprocess/
tmp/debug/
```

不要把需要长期保留的测试文件放到 `tmp/`。

## 注意事项

- 不要提交 `.env` 或真实 API Key。
- `tmp/` 可清理，只放运行时产物。
- `tests/data/camera/` 放预置测试图片。
- `tests/data/audio/` 放预置测试音频。
- `knowledge/` 放知识库资料和标准参考图片。
- 百炼回答里的具体文物名称必须来自知识库标准名称或别名，不能由后端或视觉模型自造名称。
