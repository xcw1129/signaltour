"""
# Plot: 波形图、一维/二维谱图和测试统计图等绘图可视化子包

## core: Plot子包核心模块, 实现了绘图任务流程框架与可插拔插件系统的基础类
    - class:
        - `BasePlot`: 通用绘图基类, 实现多绘图任务流程框架, 供子类继承并实现具体绘图逻辑
        - `PlotPlugin`: 绘图插件基类, 实现可插拔绘图功能, 供子类继承并实现具体插件逻辑
## PlotPlugin: 绘图插件模块, 提供谱线峰值查找与正负值掩码等可插拔绘图插件
    - class:
        - `PeakfinderPlugin`: 谱线峰值查找插件，用于查找并标注谱类数据中谱线主瓣对应的坐标
        - `PosNagMaskPlugin`: 谱线正负值掩码插件, 用于对谱类数据中正负值进行不同颜色显示
## LinePlot: 一维线条图模块, 提供波形图、频谱图与分解结果总览图等绘制方法
    - class:
        - `LinePlot`: 波形图、谱图等一维线条图绘制绘图类
    - function:
        - `PlotFunc_waveform`: 信号波形图绘制函数
        - `PlotFunc_spectrum`: 频谱绘制函数
        - `PlotFunc_decResult`: 信号分解结果总览图绘制函数
## ImagePlot: 二维图像模块, 提供时频谱图与热力图等绘制方法
    - class:
        - `ImagePlot`: 时频谱图、热力图等二维图绘图类
    - function:
        - `spectrogram_PlotFunc`: 信号时频谱图绘制函数
"""
# ruff: noqa: F403
# ruff: noqa: I001

from ._Plot_Module._core import *
from ._Plot_Module._PlotPlugin import *
from ._Plot_Module._LinePlot import *
from ._Plot_Module._ImagePlot import *

if __name__ == "__main__":
    from script.docstring import update_package_docstring

    update_package_docstring(__file__, summary="波形图、一维/二维谱图和测试统计图等绘图可视化子包")
