"""
# signaltour: 面向一维时序振荡数据的信号加载管理、预处理、深入分析与可视化全流程的Python包

---

# Signal: 信号数据读取、生成、封装和预处理等数据管理子包

## core: Signal子包核心模块, 实现了坐标轴、序列与信号数据封装的基础类与通用方法
    - class:
        - `Axis`: 通用坐标轴类, 用于生成和管理一维顺序均匀采样坐标轴数据
        - `Series`: 通用序列数据类, 用于保存和管理以一维序列数据及其坐标轴
        - `t_Axis`: 时间坐标轴类
        - `f_Axis`: 频率坐标轴类
        - `Signal`: 一维时域信号类, 实现采样信息与数据的绑定, 支持混合运算并内置常用信号数据交互方法
        - `Spectra`: 一维频谱数据类, 实现采样信息与数据的绑定, 支持混合运算并内置常用频谱数据交互方法
## SignalRead: 数据读取模块, 提供数据文件批量管理、文件夹预览与数据集扫描加载等方法
    - class:
        - `Files`: 数据文件批量管理类, 支持单一目录下指定类型数据文件的快速筛选与批量加载
        - `Folder`: 数据文件夹管理类, 支持快速预览和批量检索、筛选和加载数据文件
        - `Dataset`: 数据集扫描与管理类, 支持自动识别层级结构并发现、加载数据文件, 支持嵌套键索引
    - function:
        - `set_logging_level`: 设置当前模块的日志显示级别
## SignalSimulate: 信号仿真模块, 提供准周期信号、冲击信号与调制信号等含噪仿真信号的生成方法
    - function:
        - `periodic`: 生成仿真含噪准周期信号
        - `impulse`: 生成仿真冲击序列和噪声冲击复合信号
        - `modulation`: 生成仿真含噪调制信号
## SignalSample: 信号采样模块, 提供重采样、边界延拓与滑窗分段等方法
    - function:
        - `resample`: 截取信号任意时间段并重采样, 支持下采样与上采样
        - `pad`: 对信号对象进行边界延拓处理, 支持镜像延拓和零填充方式
        - `slice`: 对信号进行滑窗跳步分段, 首尾段自动延拓
## SignalFilter: 信号滤波模块, 提供FIR/IIR滤波器与中值滤波等去噪方法
    - function:
        - `filtFIR`: 基于有限冲击响应滤波器对信号进行各种类型滤波
        - `filtIIR`: 基于无限冲击响应滤波器对信号进行各种类型滤波
        - `filtMedian`: 基于中值滤波器对信号进行去噪处理

---

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

---

# Analysis: 统计分析、谱分析、非平稳时频分析等信号分析子包

## core: Analysis子包核心模块, 实现了信号分析处理方法的基础类与通用函数
    - class:
        - `BaseAnalysis`: 通用信号分析处理方法基类
## StatsTrendAnalysis: 时域统计分析模块, 提供时域统计趋势等方法
    - class:
        - `StatsTrendAnalysis`: 信号时域统计分析类
## SpectrumAnalysis: 平稳信号谱分析模块, 提供多种基于DFT的频谱分析方法
    - class:
        - `Spectrum`: 平稳信号频谱分析方法类
        - `Hilbert`: 单成分调制信号希尔伯特分析方法类
    - function:
        - `get_window`: 生成指定窗函数的整周期采样序列
        - `find_spectralines`: 检测谱数据中的谱线类局部峰值
        - `convolveCycle`: 计算两个序列数据的循环卷积, 该卷积方式满足DFT的卷积定理
        - `convolve`: 计算两个序列数据的线性卷积
## TimeFreqAnalysis: 非平稳信号时频分析模块, 提供多种时频谱图计算方法
    - class:
        - `STFTAnalysis`: 短时傅里叶变换 (Short-Time Fourier Transform, STFT) 分析类
        - `WVDAnalysis`: 魏格纳威利分布(Wigner-Ville Distribution, WVD) 分析类
## WaveletAnalysis:小波分析模块, 提供连续小波、离散小波等多种小波多分辨率分析方法
    - class:
        - `CWTAnalysis`: 连续小波变换 (Continuous Wavelet Transform, CWT) 分析类
        - `DWTAnalysis`: 离散小波变换 (Discrete Wavelet Transform, DWT) 分析类
## ModeAnalysis: 非平稳多分量信号模态分解模块, 提供多种分解算法(如EMD, VMD)的实现与辅助函数
    - class:
        - `EMDAnalysis`: 经验模态分解(EMD), 对输入的一维信号执行分解, 提供 IMF 提取、筛选过程可视化与结果绘制等功能。
        - `VMDAnalysis`: 变分模态分解(VMD)类, 通过频域交替优化将信号分解为若干具有有限带宽的本征模态
    - function:
        - `siftProcess_PlotFunc`: 绘制单次筛选过程的辅助图像
        - `updateProcess_PlotFunc`: 绘制 VMD 迭代更新过程的辅助图像
        - `search_localExtrema`: 搜索序列中的局部极大与极小值索引, 并基于阈值剔除弱极值点
        - `get_spectraCenter`: 计算频谱的功率加权中心频率
        - `get_Trend`: 提取信号的趋势模态
"""

__version__ = "1.2.0"

from ._Signal import *  # noqa: F403, I001
from ._Plot import *  # noqa: F403
from ._Analysis import *  # noqa: F403

if __name__ == "__main__":
    from script.docstring import update_init_docstring

    update_init_docstring(
        __file__,
        first_line="# signaltour: 面向一维时序振荡数据的信号加载管理、预处理、深入分析与可视化全流程的Python包",
    )
