# tools 目录说明

tools 放维护和调试工具脚本，不承载生产 HTTP/UDP 入口。

```powershell
python tools\build_exhibit_knowledge.py --export-only
```

```powershell
python tools\check_museum_refs.py
```

```powershell
python tools\clean_tmp.py
```

相机导游端到端调试逻辑在 `tools/camera_guide_debug.py`，命令行入口仍放在：

```text
tests/scripts/camera_guide.py
```
