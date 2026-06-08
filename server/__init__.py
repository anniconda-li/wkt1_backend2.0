"""HTTP 和 UDP 服务入口模块。

包含：
- walkie_app: FastAPI HTTP 服务 + UDP WTK1 协议服务，提供 AI 语音问答和相机拍照讲解功能
- udp_server: WTK1 UDP 服务循环，负责设备包接收、音频转发和回显测试
- protocol: WTK1 UDP 二进制协议解析和服务端回显包构造
- media: WAV/JPEG 上传数据解析、校验和保存工具
"""
