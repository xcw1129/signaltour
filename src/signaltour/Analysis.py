"""
# Analysis: 统计趋势分析、谱分析、特征提取和分解等信号处理模块

---

## 可用的接口

### core: Analysis子包核心模块, 提供各种信号分析处理方法的基类接口
    - class:
        - `BaseAnalysis`: 通用信号分析处理方法类
### StatsTrendAnalysis: 时域统计分析模块, 提供时域统计趋势等方法
    - class:
        - `StatsTrendAnalysis`: 信号时域统计分析方法
### SpectrumAnalysis: 平稳信号谱分析模块, 提供多种基于DFT的频谱分析方法
    - function:
        - `window`: 生成各类窗函数的整周期采样序列
        - `find_spectralines`: 对序列数据进行谱线类局部峰值检测
    - class:
        - `SpectrumAnalysis`: 平稳信号频谱分析方法
### TimeFreqAnalysis: 非平稳信号时频分析模块, 提供多种时频谱图计算方法
    - class:
        - `STFTAnalysis`: 短时傅里叶变换 (Short-Time Fourier Transform, STFT) 分析类
        - `WVDAnalysis`: 魏格纳威利分布(Wigner-Ville Distribution, WVD) 分析类
### WaveletAnalysis:小波分析模块, 提供连续小波、离散小波等多种小波多分辨率分析方法
    - class:
        - `CWTAnalysis`: 连续小波变换 (Continuous Wavelet Transform, CWT) 分析类
### ModeAnalysis: 非平稳多分量信号模态分解模块, 提供多种分解算法(如EMD, VMD)的实现与辅助函数
    - function:
        - `siftProcess_PlotFunc`: 绘制单次筛选过程的辅助图像
        - `decResult_PlotFunc`: 绘制 EMD/VMD 分解结果的辅助图像
        - `updateProcess_PlotFunc`: 绘制 VMD 迭代更新过程的辅助图像
        - `search_localExtrema`: 搜索序列中的局部极大与极小值索引, 并基于阈值剔除弱极值点
        - `get_spectraCenter`: 计算频谱的功率加权中心频率
        - `get_Trend`: 提取信号的趋势模态
    - class:
        - `EMDAnalysis`: 经验模态分解(EMD), 对输入的一维信号执行分解, 提供 IMF 提取、筛选过程可视化与结果绘制等功能。
        - `VMDAnalysis`: 变分模态分解(VMD), 通过频域交替优化将信号分解为若干具有有限带宽的本征模态。
"""
# ruff: noqa: F403
# ruff: noqa: I001

from ._Analysis_Module.core import *
from ._Analysis_Module.StatsTrendAnalysis import *
from ._Analysis_Module.SpectrumAnalysis import *
from ._Analysis_Module.TimeFreqAnalysis import *
from ._Analysis_Module.WaveletAnalysis import *
from ._Analysis_Module.ModeAnalysis import *

if __name__ == "__main__":
    from script.autogenerate_module_doc import generate_aggregate_docstring

    generate_aggregate_docstring(__file__, summary="统计趋势分析、谱分析、特征提取和分解等信号处理模块")
