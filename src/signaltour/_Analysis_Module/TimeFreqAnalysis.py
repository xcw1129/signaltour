"""
# TimeFreqAnalysis: 非平稳信号时频分析模块, 提供多种时频谱图计算方法

---

## 可用的接口

    - class:
        - `STFTAnalysis`: 短时傅里叶变换 (Short-Time Fourier Transform, STFT) 分析类
        - `WVDAnalysis`: 魏格纳威利分布(Wigner-Ville Distribution, WVD) 分析类
"""

__all__ = ["STFTAnalysis", "WVDAnalysis"]

from .._Assist_Module.Dependencies import Optional, Tuple, fft, np, signal
from .._Plot_Module.ImagePlot import spectrogram_PlotFunc
from .._Signal_Module.core import Signal
from .._Signal_Module.SignalSample import slice
from .core import BaseAnalysis
from .SpectrumAnalysis import window


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class STFTAnalysis(BaseAnalysis):
    """
    短时傅里叶变换 (Short-Time Fourier Transform, STFT) 分析类

    用于分析非平稳频率变化信号的线性时频特性, 提供灵活的时频分辨率控制和多种窗函数选择

    Attributes
    ----------
    Sig : Signal
        待分析信号
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        自定义绘图参数

    Methods
    -------
    - get_segNum(df: Optional[float] = None, dt: Optional[float] = None)
            -> int
        根据期望的频率分辨率df或时间分辨率dt计算合适的分段数

    - windowSegments(Sig: Signal, segNum: int, overlap: float = 0.5, padTimes: int = 0, winType: str = "汉宁窗")
            -> Tuple[np.ndarray, np.ndarray]
        将信号切片分段并加窗

    - stft(df: Optional[float] = None, dt: Optional[float] = None,
            segNum: Optional[int] = None, winType: str = "汉宁窗")
            -> tuple[np.ndarray, np.ndarray, np.ndarray]
        计算信号的短时傅里叶变换时频谱
    """

    def get_segNum(
        self,
        df: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> int:
        """
        根据期望的频率分辨率df或时间分辨率dt计算合适的分段数

        Parameters
        ----------
        df : float, optional
            期望的频率分辨率, 单位Hz
        dt : float, optional
            期望的时间分辨率, 单位s

        Returns
        -------
        int
            推荐分段数
        """
        segNum = None  # 默认值
        if df is not None or dt is not None:
            # df_seg/df=(segNum-1)/2
            if df is not None:
                df /= 3  # 考虑到谱泄露, 适当提高频率分辨率要求
                segNum_max = int((df / self.Sig.f_axis.df) * 2 + 1)
                segNum = segNum_max
            # dt_seg/dt=2N/(segNum-1)
            if dt is not None:
                segNum_min = int((2 * len(self.Sig)) / (dt * self.Sig.t_axis.fs) + 1)
                segNum = segNum_min
            if df is not None and dt is not None:
                if segNum_min > segNum_max:
                    raise ValueError(f"df<={df}Hz 和 dt<={dt}s 无法同时满足")
                segNum = (segNum_min + segNum_max) // 2
        return segNum

    @staticmethod
    def windowSegments(
        Sig: Signal, segNum: int, overlap: float = 0.5, padTimes: int = 0, winType: str = "汉宁窗"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将信号切片分段并加窗

        Parameters
        ----------
        segNum : int
            分段数
        overlap : float, default: 0.5
            分段重叠比例, 取值范围[0, 1)
        padTimes : int, default: 0
            分段零填充延拓倍数
        winType : str, default: "汉宁窗"
            窗函数类型, 支持: "矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"

        Returns
        -------
        (np.ndarray, np.ndarray)
            分段数据矩阵, 分段时间轴
        """
        # 分段
        segments, time = slice(Sig - np.mean(Sig), segNum=segNum, projection=True, overlap=overlap)
        nperseg = segments.shape[1]
        # 延拓
        segments = np.pad(
            segments,
            ((0, 0), (padTimes * nperseg // 2, padTimes * nperseg // 2)),
            mode="constant",
        )
        # 加窗
        win = window(num=nperseg, type=winType, padding=padTimes * nperseg // 2)
        segments = segments * win
        return segments, time

    @BaseAnalysis._plot(spectrogram_PlotFunc)
    def stft(
        self,
        df: Optional[float] = None,
        dt: Optional[float] = None,
        segNum: Optional[int] = None,
        winType: str = "汉宁窗",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算信号的短时傅里叶变换时频谱

        一般设置df或dt来控制时频分辨率从而自动计算segNum, 特殊情况直接指定segNum

        默认分段重叠比例为50%, 且3倍延拓, 以提高时频谱的时频位移不变性

        Parameters
        ----------
        df : float, optional
            期望的频率分辨率, 单位Hz
        dt : float, optional
            期望的时间分辨率, 单位s
        segNum : int, optional
            分段长度, 影响时频分辨率
        winType : str, default: "汉宁窗"
            窗函数类型, 支持: "矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            时间轴, 频率轴和STFT复数谱矩阵
        """
        # 根据要求的时频分辨率自动确定分段数
        auto_segNum = self.get_segNum(df=df, dt=dt)
        if auto_segNum is None:
            if segNum is None:
                raise ValueError("若未指定df或dt, 则必须指定segNum")
        else:
            segNum = auto_segNum
        # 信号分段切片与加窗
        segments, time = STFTAnalysis.windowSegments(
            Sig=self.Sig, segNum=segNum, overlap=0.5, padTimes=3, winType=winType
        )
        # 单独计算所有分段的FFT, 此时分段存在线性相移项
        Sf = fft.fft(segments, axis=1)
        freq = np.linspace(0, self.Sig.t_axis.fs, Sf.shape[1], endpoint=False)  # 生成频率轴
        if self.isPlot:
            return time, freq[: Sf.shape[1] // 2], 2 * np.abs(Sf[:, : Sf.shape[1] // 2])
        return time, freq, Sf


class WVDAnalysis(BaseAnalysis):
    """
    魏格纳威利分布(Wigner-Ville Distribution, WVD) 分析类

    用于计算信号的时频能量概率分布, 具有高时频分辨率但存在交叉项

    Attributes
    ----------
    Sig : Signal
        待分析信号
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        自定义绘图参数

    Methods
    -------
    - wvd(dt: Optional[float] = None)
            -> tuple[np.ndarray, np.ndarray, np.ndarray]
        计算信号的WVD, 表示其时频能量概率分布
    """

    @BaseAnalysis._plot(spectrogram_PlotFunc)
    def wvd(self, dt: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算信号的WVD, 表示其时频能量概率分布

        Parameters
        ----------
        dt : float, optional
            期望的时间分辨率, 单位s. 若指定则对信号进行跳步自相关以加快计算速度

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            时间轴, 频率轴, WVD实数谱矩阵
        """
        # 根据要求的时间分辨率计算nhop
        nhop = 1
        if dt is not None:
            nhop = max(1, int(dt * self.Sig.t_axis.fs))
            nhop = min(nhop, len(self.Sig) // 10)  # 限制最大跳步
        # 准备WVD参数
        N = len(self.Sig)
        analytic = signal.hilbert(self.Sig.data - np.mean(self.Sig.data))  # 求解析信号
        analytic_left = np.pad(analytic, (N, 0), "constant")  # 左侧补零
        analytic_right = np.pad(analytic, (0, N), "constant")  # 右侧补零
        # 组合时延数据矩阵
        N_t_stride = (N - 1) // nhop + 1
        # 0维为时延轴, 1维为时间轴
        Wf = np.zeros((N, N_t_stride), dtype=complex)  # N=8192占用1G内存, 并降低到1/nhop
        for n_tau in range(N):
            # 左右各移动tau/2, 共移动tau
            seg = analytic_right[n_tau : n_tau + N] * np.conj(analytic_left[N - n_tau : N - n_tau + N])
            Wf[n_tau, :] = seg[::nhop]  # 跨步长采样
        # 沿时延轴FFT
        Wf = (fft.fft(Wf, axis=0).real / N).T  # 对0维(时延轴)做FFT, 转置后0维为时间轴, 1维为频率轴
        time = self.Sig.t_axis()[::nhop]
        freq = np.arange(N) * (self.Sig.f_axis.df / 2)
        return time, freq, Wf
