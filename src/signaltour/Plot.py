"""
# Plot: 波形图、一维/二维谱图和测试统计图等绘图可视化子包

---

## 可用的接口

### core
    - class:
        - `BasePlot`: 通用绘图基类, 实现多绘图任务流程框架, 供子类继承并实现具体绘图逻辑
        - `PlotPlugin`: 绘图插件基类, 实现可插拔绘图功能, 供子类继承并实现具体插件逻辑
### PlotPlugin
    - class:
        - `PeakfinderPlugin`: 谱线峰值查找插件，用于查找并标注谱类数据中谱线主瓣对应的坐标
        - `PosNagMaskPlugin`: 谱线正负值掩码插件, 用于对谱类数据中正负值进行不同颜色显示
### LinePlot
    - function:
        - `PlotFunc_waveform`: 信号波形图绘制函数
        - `PlotFunc_spectrum`: 频谱绘制函数
        - `PlotFunc_decResult`: 信号分解结果总览图绘制函数
    - class:
        - `LinePlot`: 波形图、谱图等一维线条图绘制绘图类
### ImagePlot
    - function:
        - `spectrogram_PlotFunc`: 信号时频谱图绘制函数
    - class:
        - `ImagePlot`: 时频谱图、热力图等二维图绘图类
"""

# ruff: noqa: F403
# ruff: noqa: I001

from ._Plot_Module.core import *
from ._Plot_Module.PlotPlugin import *
from ._Plot_Module.LinePlot import *
from ._Plot_Module.ImagePlot import *

if __name__ == "__main__":
    from script.docstring import update_package_docstring

    update_package_docstring(
        __file__, summary="波形图、一维/二维谱图和测试统计图等绘图可视化子包"
    )
