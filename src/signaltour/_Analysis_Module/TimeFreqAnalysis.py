"""
# TimeFreqAnalysis: 非平稳信号时频分析模块, 提供多种时频谱图计算方法

---

## 可用的接口

    - class:
        - `STFTAnalysis`: 短时傅里叶变换 (Short-Time Fourier Transform, STFT) 分析类
        - `WVDAnalysis`: 魏格纳威利分布(Wigner-Ville Distribution, WVD) 分析类
        - `CWTAnalysis`: 连续小波变换 (Continuous Wavelet Transform, CWT) 分析类
"""

__all__ = ["STFTAnalysis", "WVDAnalysis", "CWTAnalysis"]

from .._Assist_Module.Dependencies import Literal, Optional, fft, linalg, np, signal
from .._Plot_Module.ImagePlot import spectrogram_PlotFunc
from .._Plot_Module.LinePlot import LinePlot
from .._Signal_Module.core import Signal, t_Axis
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


class CWTAnalysis(BaseAnalysis):
    """
    连续小波变换 (Continuous Wavelet Transform, CWT) 分析类

    用于分析非平稳尺度变化信号的时间-尺度特性, 提供多种尺度伸缩方案和小波函数选择

    Methods
    -------
    - get_scale(b: float = 2, j: int = 10, v: int = 1)
            -> np.ndarray
        生成对数分布离散尺度轴, b^(-j)<s<=1
    - get_wavelet(type: str, param: dict, scale: np.ndarray,
            N: int, normalized: str = "能量", isPlot: bool = False)
            -> np.ndarray
        生成指定参数基小波在不同尺度下的采样序列
    - cwt(flow: Optional[float] = None, fhigh: Optional[float] = None,
            nperoctave: int = 10, wavelet: str = "Morlet")
            -> tuple[np.ndarray, np.ndarray, np.ndarray]
        计算信号的连续小波变换时频谱
    """

    @staticmethod
    def get_scale(b: float = 2, j: int = 10, v: int = 1) -> np.ndarray:
        """生成对数分布离散尺度轴, b^(-j)<s<=1"""
        if b <= 1:
            raise ValueError("b必须大于1")
        # s=1, b^(-1/v), b^(-2/v),.., b^(-1),..., b^(-2),..., b^(-j)
        scale = 1 / np.logspace(0, j, v * j, endpoint=False, base=b)
        return scale

    @staticmethod
    def get_wavelet(
        type: str,
        param: dict = {},
        scale: Optional[np.ndarray] = None,
        N: int = 1024,
        normalized: str = "能量",
        includeScaling: bool = False,
        isPlot: bool = False,
    ) -> np.ndarray | None:
        """
        生成指定参数基小波在不同尺度下的采样序列

        函数内部自动设置各基小波的剩余参数, 使得基小波支撑集恰好为[-0.5, 0.5]

        即当对基小波进行[-0.5,0.5]区间采样时, scale=1下基小波原子取得最窄频域支撑集, 且保持频响特性

        Parameters
        ----------
        type : str
            基小波类型, 支持:
            - "Morlet": Morlet小波, 参数: fc(中心频率, 默认5), fb(带宽, 默认5)
            - "MexicanHat": Mexican Hat (Ricker)小波, 参数: fb(带宽, 默认5)
            - "DOG": DOG (Derivative of Gaussian)小波, 参数: fb(带宽, 默认5), order(阶数, 默认2)
            - "B-Spline": B样条小波, 参数: fc(中心频率, 默认5), fb(带宽, 默认5), p(阶数, 默认2)
            - "shannon": Shannon小波, 参数: fc(中心频率, 默认5), fb(带宽, 默认5)
            - "harmonic": 谐波小波, 参数: fc(中心频率, 默认5), fb(带宽, 默认5)
        param : dict, optional
            基小波函数参数字典, 详见type参数说明. 实小波通常没有fc参数
        scale : np.ndarray, optional
            离散尺度序列, 必须满足0<scale<=1. 推荐使用CWTAnalysis.get_scale()生成
        N : int, default: 1024
            离散采样点数, 一般取待变换信号长度
        normalized : str, default: "能量"
            归一化类型, 支持: "能量", "幅值", "无"
        includeScaling : bool, default: False
            是否生成scale=1下的尺度函数采样序列
        isPlot : bool, default: False
            是否绘制选定基小波采样后时域波形和频谱

        Returns
        -------
        np.ndarray or None
            基小波不同尺度采样序列, shape=(len(scale), N). 如果includeScaling=True, 则shape=(len(scale)+1, N)
        """
        match type:
            case "Morlet":

                def wavelet_func(t, fc=5, fb=5):
                    b = 1 / fb
                    atom = np.exp(-0.5 * (t / b) ** 2) * np.exp(1j * 2 * np.pi * fc * t)
                    return atom

                def scaling_func(t, fc=5, fb=5):
                    b = 1 / fb
                    return np.exp(-0.5 * (t / b) ** 2)

            case "MexicanHat":

                def wavelet_func(t, fb=5):
                    fb *= 1.7
                    atom = (1 - (t * fb) ** 2) * np.exp(-0.5 * (t * fb) ** 2)
                    return atom

                def scaling_func(t, fb=5):
                    fb *= 1.7
                    return np.exp(-0.5 * (t * fb) ** 2)

            case "DOG":

                def wavelet_func(t, fb=5, order=2):
                    fb *= 1.7
                    b = 1 / fb
                    guassian = np.exp(-0.5 * (t / b) ** 2)
                    for _ in range(order):
                        guassian = -1 * np.gradient(guassian, axis=1)
                    return guassian

                def scaling_func(t, fb=5, order=2):
                    fb *= 1.7
                    b = 1 / fb
                    return np.exp(-0.5 * (t / b) ** 2)

            case "B-Spline":

                def wavelet_func(t, fc=5, fb=5, p=2):
                    atom = np.sinc(fb * t / p) ** p * np.exp(1j * 2 * np.pi * fc * t)
                    return atom

                def scaling_func(t, fc=5, fb=5, p=2):
                    return np.sinc(fb * t / p) ** p

            case "shannon":

                def wavelet_func(t, fc=5, fb=5):
                    atom = np.sinc(fb * t) * np.exp(1j * 2 * np.pi * fc * t)
                    return atom

                def scaling_func(t, fc=5, fb=5):
                    return np.sinc(fb * t)

            case "harmonic":

                def wavelet_func(t, fc=5, fb=5):
                    X_f = np.zeros_like(t, dtype=complex)
                    for i in range(len(t)):
                        m = int((fc - fb / 2) * (t[i][-1] - t[i][0]))
                        n = int((fc + fb / 2) * (t[i][-1] - t[i][0])) + 1
                        X_f[i, m:n] = (1 + 0j) / (n - m)
                    atom = fft.ifft(X_f, axis=1)
                    atom = np.fft.fftshift(x=atom, axes=1)  # 时域居中
                    return atom

                def scaling_func(t, fc=5, fb=5):
                    X_f = np.zeros_like(t, dtype=complex)
                    for i in range(len(t)):
                        n = int((fc - fb / 2) * (t[i][-1] - t[i][0])) + 1
                        X_f[i, :n] = (1 + 0j) / (n if n > 0 else 1)
                    atom = fft.ifft(X_f, axis=1)
                    atom = np.fft.fftshift(x=atom, axes=1)
                    return atom

            case _:
                raise ValueError(f"type={type}: 不支持的小波函数类型")
        # ------------------------------------------------------------------------#
        # 标准采样时间轴, 此时N即为基小波离散采样频率
        time = np.linspace(0, 1, N, endpoint=False) - 0.5
        if isPlot:
            wavelet = wavelet_func(time, **param)
            Sig_wavelet = Signal(
                t_Axis(N=len(wavelet), T=1, t0=-0.5),
                wavelet,
                label=type + "小波",
            )
            plot = LinePlot(title=type + "小波", ncols=2, figsize=(14, 6), ylabel="幅值")
            if Sig_wavelet.data.dtype == complex:
                plot.waveform(
                    [
                        np.real(Sig_wavelet).set_label("实部"),
                        np.imag(Sig_wavelet).set_label("虚部"),
                        np.abs(Sig_wavelet).set_label("包络"),
                    ],
                    title="时域波形",
                )
            else:
                plot.waveform(Sig_wavelet, title="时域波形")
            plot.spectrum(np.abs(Sig_wavelet.to_Spectra().halfCut()), title="频谱", xlim=(0, 100))
            plot.show()
            return None
        # 生成不同尺度下的时间轴, 并带入小波解析函数得采样序列
        if scale is None:
            scale = np.array([1.0])
        time_Diffscale = time / scale.reshape(-1, 1)
        wavelet_Diffscale = wavelet_func(time_Diffscale, **param)  # 时间伸长, 则波形收缩, 频率升高
        if includeScaling:
            scale = np.hstack([1, scale])
            scaling = scaling_func(time / 1, **param)  # scale=1时的尺度函数
            wavelet_Diffscale = np.vstack([scaling, wavelet_Diffscale])
        # 归一化
        if normalized == "能量":  # 缩放前后能量一致
            wavelet_Diffscale /= linalg.norm(wavelet_Diffscale, axis=1, keepdims=True)
        elif normalized == "幅值":  # 缩放前后频谱峰值一致
            wavelet_Diffscale /= scale.reshape(-1, 1)
        else:  # 缩放前后时域峰值一致
            pass
        return wavelet_Diffscale

    @BaseAnalysis._plot(spectrogram_PlotFunc)
    def cwt(
        self,
        flow: Optional[float] = None,
        fhigh: Optional[float] = None,
        nperoctave: int = 10,
        wavelet: str = "Morlet",
        param: dict = {},
        includeScaling: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算信号的连续小波变换时频谱

        Parameters
        ----------
        flow : float, optional
            最小分析频率, 单位Hz
            默认值为信号频率轴分辨率的10倍
        fhigh : float, optional
            最大分析频率, 单位Hz
            默认值为信号采样频率的50%(奈奎斯特频率)
        nperoctave : int, default: 10
            每倍频程的离散尺度数, 控制尺度轴分辨率
        wavelet : str, default: "Morlet"
            小波函数类型, 支持:
            - "Morlet": Morlet小波
            - "MexicanHat": Mexican Hat (Ricker)小波
            - "DOG": DOG (Derivative of Gaussian)小波
            - "B-Spline": B样条小波
            - "shannon": Shannon小波
            - "harmonic": 谐波小波
        param : dict, optional
            基小波函数参数字典, 详见get_wavelet()方法说明
        includeScaling : bool, default: True
            时频谱中是否包含尺度函数分量

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            时间轴, 频率轴, CWT谱矩阵
        """
        # 生成离散尺度轴
        flow = 10 * self.Sig.f_axis.df if flow is None else flow
        fhigh = self.Sig.t_axis.fs / 2 if fhigh is None else fhigh
        ratio = fhigh / flow
        j = int(np.log2(ratio)) + 1
        scale = CWTAnalysis.get_scale(b=2, j=j, v=nperoctave)  # s<=1
        # 生成基小波的离散尺度采样序列
        param.update({"fc": flow / self.Sig.f_axis.df})  # 归一化频率
        wavelets = CWTAnalysis.get_wavelet(
            type=wavelet,
            param=param,
            scale=scale,
            N=len(self.Sig),
            normalized="幅值",
            includeScaling=includeScaling,
        )
        freq = flow / scale
        if includeScaling:
            freq = np.hstack([0, freq])

        time: np.ndarray = self.Sig.t_axis()
        # 去除中心频率超出fn的尺度
        validIdx = np.where(freq <= self.Sig.t_axis.fs / 2)[0]
        wavelets, freq = wavelets[validIdx, :], freq[validIdx]
        # ------------------------------------------------------------------------#
        # 滤波法计算CWT谱矩阵: 兼容复小波和实小波
        # 1. 预计算信号的FFT
        data_fft = fft.fft(self.Sig._data)
        # 2. 预计算所有小波核的FFT (同时计算, 利用axis=1向量化)
        wavelets_fft = fft.fft(np.conj(np.flip(wavelets, axis=1)), axis=1)
        # 3. 频域相乘 (利用广播: (N,) * (M, N) -> (M, N))
        Wf_fft = data_fft * wavelets_fft
        # 4. 逆变换回时域
        Wf = fft.ifft(Wf_fft, axis=1)
        # 5. 调整相位/对齐 (FFT卷积是循环卷积, 需要对应 mode='same' 进行移位)
        Wf = fft.fftshift(Wf, axes=1).T  # 转置后 0维为时间, 1维为尺度
        if self.isPlot:
            return time, freq, np.abs(Wf) if np.iscomplexobj(Wf) else Wf
        return time, freq, Wf
