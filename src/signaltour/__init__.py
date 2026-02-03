"""
# signaltour: 面向一维时序振荡数据的信号加载管理、预处理、深入分析与可视化全流程的Python包

---

## 可用的子包
    - `Signal`:
        信号数据读取、生成、封装和预处理等数据管理子包
    - `Analysis`:
        统计分析、谱分析、非平稳时频分析等信号分析子包
    - `Plot`:
        波形图、一维/二维谱图和测试统计图等绘图可视化子包
"""

__version__ = "1.1.0"

from . import Signal, Plot, Analysis  # noqa: I001

__all__ = ["Signal", "Plot", "Analysis"]
