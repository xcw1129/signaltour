# signaltour

一个用于一维时间序列振荡数据分析的 Python 信号处理库。

## 概述

signaltour 为信号加载、预处理、分析和可视化提供了面向对象的工作流程。它通过模块化、可扩展的架构，为处理一维时域信号提供了全面的工具集。

## 特性

- **信号管理**：加载、生成和操作时间序列数据，自动处理元数据（采样频率、持续时间、单位）
- **频谱分析**：基于 FFT 的频域分析，支持可配置的窗函数和缩放方式
- **模态分解**：实现 EMD（经验模态分解）和 VMD（变分模态分解）算法
- **时频分析**：STFT、小波变换等时频表示方法
- **数字滤波**：FIR、IIR 和中值滤波，支持常用滤波器设计（Butterworth、Chebyshev 等）
- **灵活可视化**：基于任务队列的绘图系统，支持多子图布局和可扩展插件
- **数据导入导出**：支持常见文件格式，提供文件夹和数据集管理功能

## 安装

### 从 PyPI 安装

```bash
pip install signaltour
```

### 从源码安装

```bash
git clone https://github.com/xcw1129/signaltour.git
cd signaltour
pip install -e .
```

## 依赖要求

- Python ≥ 3.11
- numpy ≥ 2.0.0
- scipy ≥ 1.14.0
- matplotlib ≥ 3.9.0
- pandas ≥ 2.2.2
- anytree ≥ 2.13.0
- pyarrow ≥ 22.0.0

## 快速入门

```python
from signaltour import Signal, Analysis, Plot

# 生成仿真信号
sig = Signal.periodic(
    fs=1000,
    T=1.0,
    CosParams=((10, 1.0, 0), (50, 0.5, 0)),  # (频率, 幅值, 相位)
    noise=0.1
)

# 执行频谱分析
analyzer = Analysis.SpectrumAnalysis(sig, isPlot=True)
spectrum = analyzer.ft()  # 傅里叶变换

# 创建自定义可视化
plot = Plot.LinePlot()
plot.waveform(sig)
plot.show()
```

## 架构设计

signaltour 采用三层模块化设计：

### Signal 模块（信号模块）
提供信号表示、文件读写、仿真生成、采样处理和滤波的核心数据结构。

```python
from signaltour import Signal

# 生成周期信号
sig = Signal.periodic(
    fs=1000,
    T=2.0,
    CosParams=((50, 1.0, 0),),  # (频率, 幅值, 相位)
    noise=0.1
)

# 生成冲击信号
sig_impulse = Signal.impulse(
    fs=4000,
    T=1.0,
    ImpParams=(1400, 20, 0.05, 20, 0.01),  # (中心频率, 出现频率, 滑移百分比, 幅值, 衰减时间)
)

# 应用滤波
filtered = Signal.filtIIR(sig, cutoff=100, order=4, btype='low', ftype='butter')

# 重采样
resampled = Signal.resample(sig, new_fs=500)
```

### Analysis 模块（分析模块）
为信号分析算法提供标准化框架，支持可选的可视化输出。

```python
from signaltour import Signal, Analysis

sig = Signal.periodic(fs=1000, T=1.0, CosParams=((50, 1.0, 0),))

# 频域分析
analyzer = Analysis.SpectrumAnalysis(sig, isPlot=False)
spectrum = analyzer.ft(window='汉宁窗')  # 傅里叶变换
psd = analyzer.psd(window='汉宁窗')  # 功率谱密度

# 模态分解
emd_analysis = Analysis.EMDAnalysis(sig, isPlot=True)
imfs = emd_analysis.emd(decNum=5)  # 经验模态分解

vmd_analysis = Analysis.VMDAnalysis(sig, isPlot=True)
modes = vmd_analysis.vmd(K=3)  # 变分模态分解

# 时频分析
stft_analysis = Analysis.STFTAnalysis(sig, isPlot=True)
t_axis, f_axis, tfr = stft_analysis.stft(segNum=256)  # 短时傅里叶变换
```

### Plot 模块（绘图模块）
基于任务队列的绘图引擎，提供链式 API 和插件支持。

```python
from signaltour import Signal, Analysis, Plot

sig1 = Signal.periodic(fs=1000, T=1.0, CosParams=((50, 1.0, 0),))
sig2 = Signal.periodic(fs=1000, T=1.0, CosParams=((100, 0.5, 0),))

# 多信号波形图
plot = Plot.LinePlot(ncols=2, title="信号对比")
plot.waveform(sig1)  # 第一个子图
plot.waveform(sig2)  # 第二个子图
plot.show()

# 频谱可视化
analyzer = Analysis.SpectrumAnalysis(sig1)
spectrum = analyzer.ft()

plot = Plot.LinePlot()
plot.spectrum(spectrum)
plot.show()

# 二维时频谱图
stft_analysis = Analysis.STFTAnalysis(sig1)
t_axis, f_axis, tfr = stft_analysis.stft(segNum=256)

plot = Plot.ImagePlot()
plot.spectrogram(time=t_axis, freq=f_axis, matrix=tfr)
plot.show()

# 使用插件
plot = Plot.LinePlot()
plot.spectrum(spectrum)
plot.add_plugin_to_task(Plot.PeakfinderPlugin(threshold=0.8))  # 峰值查找插件
plot.show()
```

## 许可证

Apache License 2.0

详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 作者：Xiong Chengwen
- 邮箱：xiongcw1129@gmail.com
- GitHub：[https://github.com/xcw1129/signaltour](https://github.com/xcw1129/signaltour)
