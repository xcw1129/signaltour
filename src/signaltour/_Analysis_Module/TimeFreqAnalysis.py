"""
# TimeFreqAnalysis: 非平稳信号时频分析模块, 提供多种时频谱图计算方法

---

## 可用的接口

    - class:
        - `STFTAnalysis`: 短时傅里叶变换 (Short-Time Fourier Transform, STFT) 分析类
        - `WVDAnalysis`: 魏格纳威利分布(Wigner-Ville Distribution, WVD) 分析类
"""

__all__ = ["STFTAnalysis", "WVDAnalysis", "CWTAnalysis"]

from .._Assist_Module.Dependencies import Optional, fft, np, signal
from .._Plot_Module.ImagePlot import spectrogram_PlotFunc
from .._Signal_Module.core import Signal
from .._Signal_Module.SignalSample import slice
from .core import BaseAnalysis
from .SpectrumAnalysis import window
from .WaveletAnalysis import CWTAnalysis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class STFTAnalysis(BaseAnalysis):
    """
    短时傅里叶变换 (Short-Time Fourier Transform, STFT) 分析类

    用于分析非平稳频率变化信号的线性时频特性, 提供灵活的时频分辨率控制和多种窗函数选择

    Methods
    -------
    - get_segNum(df: Optional[float] = None, dt: Optional[float] = None)
            -> int
        根据期望的频率分辨率df或时间分辨率dt计算合适的分段数
    - get_segMatrix(Sig: Signal, segNum: int,
            padTimes: int = 0, winType: str = "汉宁窗")
            -> tuple[np.ndarray, np.ndarray]
        将信号切片分段并加窗, 返回分段数据矩阵和段时间轴
    - stft(df: Optional[float] = None, dt: Optional[float] = None,
            segNum: int = 101, WinType: str = "汉宁窗")
            -> tuple[np.ndarray, np.ndarray, np.ndarray]
        计算信号的STFT时频谱, 返回时间轴、频率轴和谱矩阵
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
    def get_segMatrix(
        Sig: Signal, segNum: int, padTimes: int = 0, winType: str = "汉宁窗"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        将信号切片分段并加窗, 返回分段数据矩阵和段时间轴

        Parameters
        ----------
        segNum : int
            分段长度, 影响时频分辨率
        padTimes : int, default: 0
            信号分段切片零填充延拓倍数
        winType : str, default: "汉宁窗"
            窗函数类型, 支持: "矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"

        Returns
        -------
        (np.ndarray, np.ndarray)
            分段数据矩阵和每段时间轴
        """
        Mat, time = slice(Sig - np.mean(Sig), segNum=segNum, projection=True)  # 默认50%切片重叠, 提高时频分布时移不变性
        nperseg = Mat.shape[1]
        Mat = np.pad(
            Mat,
            ((0, 0), (padTimes * nperseg // 2, padTimes * nperseg // 2)),
            mode="constant",
        )  # 延拓提高时频分布的频移不变性
        # 加窗优化能量泄露效应
        win = window(num=nperseg, type=winType, padding=padTimes * nperseg // 2)
        Mat = Mat * win
        return Mat, time

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
        type : str, default: "幅值谱"
            谱类型, 支持: "幅值谱", "功率谱"

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            时间轴、频率轴和STFT复数谱矩阵

        Notes
        -----
        一般设置df或dt来控制时频分辨率从而自动计算segNum, 特殊情况直接指定segNum
        """
        # 根据要求的时频分辨率自动确定分段数
        auto_segNum = self.get_segNum(df=df, dt=dt)
        if auto_segNum is None:
            if segNum is None:
                raise ValueError("若未指定df或dt, 则必须指定segNum")
        else:
            segNum = auto_segNum
        # 信号分段切片与加窗
        Mat, time = STFTAnalysis.get_segMatrix(Sig=self.Sig, segNum=segNum, padTimes=3, winType=winType)
        # 计算每帧FFT
        S = fft.fft(Mat, axis=1)
        # 生成频率轴
        freq = np.linspace(0, self.Sig.t_axis.fs, S.shape[1], endpoint=False)
        if self.isPlot:
            return time, freq[: S.shape[1] // 2], 2 * np.abs(S[:, : S.shape[1] // 2])
        return time, freq, S


class WVDAnalysis(BaseAnalysis):
    """
    魏格纳威利分布(Wigner-Ville Distribution, WVD) 分析类

    用于计算信号的Cohen类双线性时频分布, 具有高时频分辨率但存在交叉项

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
            期望的时间分辨率, 单位s
            若指定则对信号进行跳步自相关以加快计算速度

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            时间轴、频率轴和WVD实数谱矩阵
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
        W = np.zeros((N, N_t_stride), dtype=complex)  # N=8192占用1G内存, 并降低到1/nhop
        for n_tau in range(N):
            # 左右各移动tau/2, 共移动tau
            seg = analytic_right[n_tau : n_tau + N] * np.conj(analytic_left[N - n_tau : N - n_tau + N])
            W[n_tau, :] = seg[::nhop]  # 跨步长采样
        # 沿时延轴FFT
        W = (fft.fft(W, axis=0).real / N).T  # 对0维(时延轴)做FFT, 转置后0维为时间轴, 1维为频率轴
        time = self.Sig.t_axis()[::nhop]
        freq = np.arange(N) * (self.Sig.f_axis.df / 2)
        return time, freq, W
