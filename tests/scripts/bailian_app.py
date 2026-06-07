"""百炼应用手动测试脚本。

简单的命令行工具，用于手动测试百炼 AI 应用的调用。
发送一个问题文本并打印 AI 回复。
"""

from __future__ import annotations

import logging

import core.config  # noqa: F401 - 加载项目 .env 环境变量
from services.bailian_app_service import BailianAppService


def main() -> None:
    """主函数：创建百炼服务并发送测试问题。"""
    logging.basicConfig(level=logging.INFO)
    service = BailianAppService()
    answer = service.ask("大雁塔和西游记有什么关系？")
    print(answer)


if __name__ == "__main__":
    main()
