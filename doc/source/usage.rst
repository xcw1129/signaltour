signaltour 使用指南
=============

本章展示如何在本地环境中安装 signaltour、初始化信号对象、组合分析与可视化任务，并介绍常见的工程级流程。

安装与环境
------------

- 使用国内镜像加速：

  .. code-block:: bash

      pip install -i https://pypi.tuna.tsinghua.edu.cn/simple signaltour-xcw

- 推荐 Python 版本：3.9~3.12，与 NumPy/SciPy/Matplotlib 兼容。
- 如需源代码安装：克隆仓库后执行 ``python -m pip install -e .``。

快速上手示例
--------------

下面的脚本创建一个合成信号，计算功率谱，并渲染频谱图：

.. code-block:: python

    import numpy as np
    from signaltour.Signal import Signal
    from signaltour.Analysis import SpectrumAnalysis
    from signaltour.Plot import LinePlot

    t = np.linspace(0, 1, 400, endpoint=False)
    signal = Signal.from_array(np.sin(2 * np.pi * 20 * t) + 0.4 * np.sin(2 * np.pi * 60 * t), fs=400)

    analysis = SpectrumAnalysis(signal)
    spectrum = analysis.run()

    LinePlot()\
        .spectrum_PlotFunc(spectrum, title="多频成分频谱")\
        .show()

关键工作流
----------

1. **预处理**：使用 ``Signal`` 的 ``resample``/``pad``/``slice`` 函数让数据对齐。采样率一致可避免频谱混叠。
2. **分析**：调用 ``signaltour.Analysis`` 下的分析类（如 ``STFTAnalysis``、``EMDAnalysis``）的 ``run`` 方法，获取结构化结果对象。
3. **绘图**：将分析产出传给 ``LinePlot`` / ``ImagePlot``，利用 ``plot_func`` 系列方法、插件或任务队列定制布局。
4. **导出**：可通过 Matplotlib API 进一步保存图像或导出信号，确保与 Matplotlib 工具链兼容。


