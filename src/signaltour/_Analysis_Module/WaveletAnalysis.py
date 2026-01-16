"""
# WaveletAnalysis:小波分析模块, 提供连续小波、离散小波等多种小波多分辨率分析方法

---

## 可用的接口

    - class:
        - `CWTAnalysis`: 连续小波变换 (Continuous Wavelet Transform, CWT) 分析类
"""

__all__ = ["CWTAnalysis"]

from .._Assist_Module.Dependencies import Optional, fft, linalg, np
from .._Plot_Module.ImagePlot import spectrogram_PlotFunc
from .._Plot_Module.LinePlot import LinePlot
from .._Signal_Module.core import Signal, t_Axis
from .core import BaseAnalysis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
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
            # 复小波一般使用窗函数法设计, 其中核函数即为尺度函数
            # 实小波一般无中心频率参数, 适合分析信号奇异性
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

            case "Shannon":

                def wavelet_func(t, fc=5, fb=5):
                    atom = np.sinc(fb * t) * np.exp(1j * 2 * np.pi * fc * t)
                    return atom

                def scaling_func(t, fc=5, fb=5):
                    return np.sinc(fb * t)

            case "Harmonic":

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
            默认值为信号频率分辨率的10倍
        fhigh : float, optional
            最大分析频率, 单位Hz
            默认值为信号采样频率的50%(奈奎斯特频率)
        nperoctave : int, default: 10
            每倍频程的离散尺度数, 控制时频谱频移不变性和平滑度
        wavelet : str, default: "Morlet"
            基小波类型, 支持:
            - "Morlet": Morlet小波
            - "MexicanHat": Mexican Hat (Ricker)小波
            - "DOG": DOG (Derivative of Gaussian)小波
            - "B-Spline": B样条小波
            - "shannon": Shannon小波
            - "harmonic": 谐波小波
        param : dict, optional
            基小波函数参数, 详见get_wavelet()方法说明
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
            normalized="幅值",  # 保持与STFT时频谱一致的幅值尺度
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
