signaltour 信号处理工具集
=====================

signaltour 是一个面向中文科研与工程场景的信号处理工具库，致力于把时间序列、频谱分析和可视化流程用面向对象的方式串联起来。
该项目在保持 NumPy/SciPy 兼容性的基础上，通过统一的 `Signal` 数据结构、可组合的分析器与可扩展的绘图引擎，让你可以在分析、探索与展示之间快速切换。

.. note::
   当前版本：``7.5.4`` · 在持续集成中使用 ``pip install signaltour-xcw`` 即可获取。

优势速览
----------

- **统一信号结构**：`Signal` 类封装数值、时间、采样率等元信息，自带切片、重采样、补零等方法。
- **模块化分析**：`Analysis` 基类打通频谱（`SpectrumAnalysis`）、经验模态（`EMDAnalysis`）与时频（`STFTAnalysis`）等多种算法，所有结果可以直接交给可视化层。
- **链式可视化**：`Plot`、`LinePlot`、`ImagePlot` 等类支持任务队列与插件，轻松绘制波形、频谱、时频图并叠加标记。
- **中文文档与示例**：本套文档以中文为主，介绍安装/入门、API 参考和最佳实践，适合国内开发者阅读与维护。

快速开始
----------

1. 通过 PyPI 安装：

   .. code-block:: bash

       pip install signaltour-xcw

2. 创建信号对象并查看频谱示例：

   .. code-block:: python

       from signaltour.Signal import Signal
       from signaltour.Analysis import SpectrumAnalysis
       from signaltour.Plot import LinePlot

       signal = Signal.from_array([0.0, 1.0, 0.0, -1.0], fs=100)
       spec = SpectrumAnalysis(signal).run()
       LinePlot().spectrum_PlotFunc(spec).show()

3. 阅读下列文档章节，逐步掌握工作流：

   - 使用指南：``usage``
   - API 参考：``api``

文档导航
----------

.. toctree::
   :maxdepth: 2
   :caption: 快速跳转

   usage
   api
   notes

更多资源
--------

- 贡献者可参考 ``notes/`` 中的实验记录与方法思路。
- 通过 ``docs/build/html`` 中的本地站点预览最终效果。
