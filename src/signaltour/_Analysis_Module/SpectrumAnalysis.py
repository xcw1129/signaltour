"""
# SpectrumAnalysis: 平稳信号谱分析模块, 提供多种基于DFT的频谱分析方法

---

## 可用的接口

    - function:
        - `get_window`: 生成各类窗函数的整周期采样序列
        - `find_spectralines`: 对序列数据进行谱线类局部峰值检测
    - class:
        - `Spectrum`: 平稳信号频谱分析方法
"""

__all__ = [
    "get_window",
    "find_spectralines",
    "convolveCycle",
    "convolve",
    "Spectrum",
    "Hilbert",
]

from .._Assist_Module.Dependencies import Callable, Optional, fft, linalg, np, signal
from .._Plot_Module.LinePlot import PlotFunc_spectrum
from .._Signal_Module.core import Signal, Spectra, f_Axis
from .core import BaseAnalysis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
def get_window(
    num: int,
    type: str = "汉宁窗",
    winParam: Optional[float] = None,
    symmetric: bool = False,
    padding: Optional[int] = None,
    func: Optional[Callable] = None,
) -> np.ndarray:
    """
    生成指定窗函数的整周期采样序列

    Parameters
    ----------
    num : int
        采样点数
    type : str, default: "汉宁窗"
        窗函数类型, 可选:
        "矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗",
        "三角窗", "高斯窗", "凯泽窗", "平顶窗", "余弦窗", "自定义窗"
    winParam : float, optional
        窗函数参数, 仅对部分窗函数有效
    symmetric : bool, default: False
        是否生成对称窗. 对于DFT计算加窗, 建议设置为False
    padding : int, optional
        零填充点数
    func : Callable, optional
        自定义窗函数

    Returns
    -------
    np.ndarray
        窗函数采样序列

    See Also
    --------
    - scipy.signal.get_window

    Notes
    -----
    当 type='高斯窗' 时, 需通过 winParam 指定标准差参数 sigma

    当 type='凯泽窗' 时, 需通过 winParam 指定形状参数 beta

    当 type='自定义窗' 时, 需通过 func 指定窗函数, func 输入变量 t 范围为 [0, 1)
    """
    window_name = {
        "矩形窗": "boxcar",
        "汉宁窗": "hann",
        "海明窗": "hamming",
        "巴特利特窗": "bartlett",
        "布莱克曼窗": "blackman",
        "三角窗": "triang",
        "高斯窗": "gaussian",
        "凯泽窗": "kaiser",
        "平顶窗": "flattop",
        "余弦窗": "cosine",
    }
    # --------------------------------------------------------------------------------#
    # 生成窗采样序列
    # 对称窗: t= n/(num-1), n=0,1/(num-1),2/(num-1),.., 1
    # 非对称窗: t=n/num=0,1/num,2/num,..(num//2)/num,..,(num-1)/num
    if type == "自定义窗":
        if not isinstance(func, Callable):
            raise ValueError(f"`func`错误. `type`='自定义窗'时, 需通过`func`指定窗函数. 当前`func`={func}")
        n = np.arange(num)  # n=0,1,2,3,...,N-1
        if symmetric:
            data = func(n / (num - 1))  # t= 0,...,1
        else:
            data = func(n / num)  # t= 0,...,(N-1)/N
    elif type not in window_name.keys():
        raise ValueError(f"`type`不支持. 可选类型: {list(window_name.keys())}. 当前`type`={type}")
    else:
        if type in ["高斯窗", "凯泽窗"] and winParam is not None:
            window = (window_name[type], winParam)
        else:
            window = window_name[type]
        data = signal.get_window(window=window, Nx=num, fftbins=not symmetric)
    # 当num为偶数且非对称窗时, 可取到t=0.5位置
    # 当num为奇数且对称窗时, 可取到t=0.5位置
    # --------------------------------------------------------------------------------#
    # 进行双边零填充
    if padding is not None:
        data = np.pad(data, padding, mode="constant")
    return data


def find_spectralines(
    data: np.ndarray,
    threshold: float = 0.8,
    distance: float = 0.01,
) -> np.ndarray:
    """
    检测谱数据中的谱线类局部峰值

    Parameters
    ----------
    data : np.ndarray
        谱数据
    threshold : float, default: 0.8
        邻域稀疏度阈值, 输入范围: (1/sqrt(d*2+1), 1)
    distance : float, default: 0.01
        峰值最小间距. 若<1则表示数据总长度的比例, 若>1则表示数据点数

    Returns
    -------
    np.ndarray
        峰值索引数组

    Notes
    -----
    方法使用signal.find_peaks函数初步筛选局部峰值点, 然后结合谱线邻域稀疏度判据进行二次筛选
    """

    def sparsity(x: np.ndarray) -> float:
        # : L1范数 / (sqrt(N) * L2范数)
        # : 1. 尺度不变; 2. 长度相关; 3. 范围[1/sqrt(N), 1]
        if np.linalg.norm(x, 2) == 0:
            return 1.0  # 全零序列视为最不稀疏
        return (np.linalg.norm(x, 1)) / (np.sqrt(len(x)) * np.linalg.norm(x, 2))

    # 初筛所有局部峰值
    if distance < 1:
        distance = int(len(data) * distance)
    else:
        distance = int(distance)
    peaks_idx, _ = signal.find_peaks(data, distance=distance + 1)  # +1确保峰值间距至少为distance
    # 二次筛选谱线类峰值
    line_peaks_idx = []
    for idx in peaks_idx:
        # 取出峰值邻域数据段
        seg = data[max(0, idx - distance) : min(len(data), idx + distance + 1)]
        # 计算稀疏度指标
        seg_sparse = sparsity(seg)
        # 邻域稀疏的峰值判定为谱线
        if seg_sparse < threshold:
            line_peaks_idx.append(idx)
    line_peaks_idx = np.array(line_peaks_idx)
    return line_peaks_idx


def convolveCycle(x: np.ndarray, y: np.ndarray, method="fft") -> np.ndarray:
    """
    计算两个序列数据的循环卷积, 该卷积方式满足DFT的卷积定理

    两个序列长度必须相等

    method='direct'时输入序列我的长度推荐不超过16384, 避免计算过程占用2GB以上内存

    Parameters
    ----------
    x : np.ndarray
        序列1
    y : np.ndarray
        序列2
    method : str, default: "fft"
        卷积计算方式, 可选: "direct", "fft"

    Returns
    -------
    np.ndarray
        循环卷积结果
    """
    if len(x) != len(y):
        raise ValueError(f"输入序列长度错误. 循环卷积要求输入序列长度相等. 当前len(x)={len(x)}, len(y)={len(y)}")
    if method == "fft":
        # 通过频域乘计算循环卷积
        X_f = Spectrum.dft(x)
        Y_f = Spectrum.dft(y)
        Z_f = X_f * Y_f
        # 还原时域得卷积结果
        z_n = Spectrum.idft(Z_f).real
        return z_n
    elif method == "direct":
        # 直接计算循环卷积
        Y_pad_trans = linalg.circulant(np.conj(y))  # 循环卷积矩阵, circulant自动转置
        z = np.dot(Y_pad_trans, x)  # 计算循环卷积的一个周期
        return z
    else:
        raise ValueError(f"卷积计算方式`method`不支持. 可选方式: 'fft', 'direct'. 当前`method`={method}")


def _convolve(x: np.ndarray, y: np.ndarray, method="fft") -> np.ndarray:
    """
    计算两个序列数据的线性卷积, 输出长度默认为 len(x)+len(y)-1, 即"full"模式

    该方法仅用于演示序列数据的线性卷积计算过程, 实际计算使用 convolve

    method='direct'时输入参数x和y的长度总和推荐不超过16384, 避免计算过程占用2GB以上内存

    Parameters
    ----------
    x : np.ndarray
        序列1
    y : np.ndarray
        序列2
    method : str, default: "fft"
        卷积计算方式, 可选: "direct", "fft"

    Returns
    -------
    np.ndarray
        线性卷积结果
    """
    N = len(x) + len(y) - 1  # 卷积结果长度
    # 延拓数据以通过循环卷积计算线性卷积
    x_pad = np.pad(x, (0, N - len(x)), mode="constant")
    y_pad = np.pad(y, (0, N - len(y)), mode="constant")
    # 执行循环卷积
    z = convolveCycle(x_pad, y_pad, method=method)
    return z


def convolve(x: np.ndarray, y: np.ndarray, mode: str = "full") -> np.ndarray:
    """
    计算两个序列数据的线性卷积

    该方法根据输入序列长度自动选择重叠相加法, 以实现快速计算

    Parameters
    ----------
    x : np.ndarray
        序列1
    y : np.ndarray
        序列2
    mode : str, default: "full"
        卷积结果长度模式, 可选: "full", "same", "valid"

    Returns
    -------
    np.ndarray
        线性卷积结果

    See Also
    --------
    - scipy.signal.convolve
    - scipy.signal.oaconvolve
    """
    if len(x) // len(y) >= 10:  # 当一个序列远长于另一个序列时(例如FIR滤波), 使用重叠相加法进行快速卷积
        z = signal.oaconvolve(x, y, mode=mode)
    else:
        z = signal.convolve(x, y, mode=mode)  # 自动选择最快的卷积方法: direct or fft
    return z


# --------------------------------------------------------------------------------------------#
class Spectrum(BaseAnalysis):
    """
    平稳信号频谱分析方法类

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
    - dft(data: np.ndarray) -> np.ndarray
        计算序列数据的离散傅里叶变换

    - idft(data: np.ndarray) -> np.ndarray
        计算序列数据的逆离散傅里叶变换

    - ft(symmetric: bool = False) -> Spectra
        计算能量信号在0~N/2*Δf范围内傅里叶变换的离散近似

    - cft(winType: str = "汉宁窗", padTimes: int = 3) -> Spectra
        计算功率信号在0~N/2*Δf范围内傅里叶级数系数的离散近似

    - psd(averageTimes: int = 10, type: str = "功率") -> Spectra
        估计带噪声功率信号在0~N/2*Δf范围内的功率分布

    - psdDiff(Sig_ref: Signal, averageTimes: int = 10, mode: str = "absolute") -> Spectra
        计算与参考信号的差分功率谱

    - enveSpectra() -> Spectra
        计算希尔伯特包络幅值谱
    """

    def __init__(
        self,
        Sig: Signal,
        isLinked: bool = True,
        isPlot: bool = False,
        **kwargs,
    ):
        """
        平稳信号频谱分析方法类

        Parameters
        ----------
        Sig : Signal
            待分析信号
        isLinked : bool, default: True
            是否链接信号原始数据
        isPlot : bool, default: False
            是否绘制分析结果图
        """
        self.Sig = Sig if isLinked else Sig.copy()
        self.isPlot = isPlot
        self.plot_kwargs = kwargs
        self.plot_kwargs.setdefault("isFindPeaks", True)  # 默认启用谱线峰值标记

    @staticmethod
    def _dft(data: np.ndarray) -> np.ndarray:
        """
        计算序列数据的离散傅里叶变换

        该方法仅用于演示DFT计算过程, 实际计算使用 Spectrum.dft

        输入参数data的长度推荐不超过16384, 避免计算过程占用4GB以上内存

        Returns
        -------
        np.ndarray
            DFT结果
        """
        N = len(data)
        n = np.arange(N)  # 时间序列索引
        k = n.reshape((N, 1))  # 频率序列索引
        # 构造DFT矩阵
        W = np.exp(-2j * np.pi * k * n / N)
        # 计算DFT: X(k)=Σx(n)*e^(-2πknj/N)
        X_k = np.dot(W, data)
        return X_k

    @staticmethod
    def _idft(data: np.ndarray) -> np.ndarray:
        """
        计算序列数据的逆离散傅里叶变换

        该方法仅用于演示IDFT计算过程, 实际计算使用 Spectrum.idft

        输入参数data的长度推荐不超过16384, 避免计算过程占用4GB以上内存

        Returns
        -------
        np.ndarray
            IDFT结果
        """
        N = len(data)
        n = np.arange(N)  # 时间序列索引
        k = n.reshape((N, 1))  # 频率序列索引
        # 构造IDFT矩阵
        W_inv = np.exp(2j * np.pi * k * n / N)
        # 计算IDFT: x(n)=(1/N)*ΣX(k)*e^(2πknj/N)
        x_n = (1 / N) * np.dot(W_inv, data)
        return x_n

    @staticmethod
    def dft(data: np.ndarray) -> np.ndarray:
        """计算序列数据的离散傅里叶变换"""
        X_k = fft.fft(data)
        return X_k

    @staticmethod
    def idft(data: np.ndarray) -> np.ndarray:
        """计算序列数据的逆离散傅里叶变换"""
        x_n = fft.ifft(data)
        return x_n

    @BaseAnalysis._plot(PlotFunc_spectrum)
    def ft(self, symmetric: bool = False) -> Spectra:
        """
        计算能量信号在0~N/2*Δf范围内傅里叶变换的离散近似

        Parameters
        ----------
        symmetric : bool, default: False
            是否生成零频率中心的对称频谱

        Returns
        -------
        Spectra
            傅里叶变换谱
        """
        # 计算傅里叶变换: FT=DFT*Δt
        X_f = Spectrum.dft(self.Sig.data) * self.Sig.t_axis.dt
        # 构造频谱对象
        Spc = Spectra(
            axis=self.Sig.f_axis,
            data=X_f,
            name="幅值密度",
            unit=self.Sig.unit + "/Hz",
            label=self.Sig.label,
        )
        Spc.data = np.abs(Spc.data)
        if symmetric:
            Spc.data = fft.fftshift(Spc.data)
            freq = fft.fftshift(fft.fftfreq(len(Spc), d=self.Sig.t_axis.dt))
            Spc.f_axis.f0, Spc.f_axis.df = freq[0], freq[1] - freq[0]
        return Spc

    @BaseAnalysis._plot(PlotFunc_spectrum)
    def cft(self, winType: str = "汉宁窗", padTimes: int = 3) -> Spectra:
        """
        计算功率信号在0~N/2*Δf范围内傅里叶级数系数的离散近似

        Parameters
        ----------
        winType : str, default: "汉宁窗"
            加窗类型，可选："矩形窗", "汉宁窗", "海明窗", "巴特利特窗", "布莱克曼窗", "自定义窗"
        padTimes : int, default: 3
            零填充延拓倍数, 信号长度在计算DFT前将延长为原来的 (1+padTimes) 倍, 以增强频谱频移不变性

        Returns
        -------
        Spectra
            傅里叶级数系数谱
        """
        win = get_window(num=len(self.Sig), type=winType, padding=padTimes * len(self.Sig) // 2)
        scale = 1 / np.mean(win)  # 幅值补偿因子
        # 计算傅里叶级数系数: CFT=DFT/N
        data_pad = np.pad(self.Sig.data, padTimes * len(self.Sig) // 2, mode="constant")
        X_k = Spectrum.dft(data_pad * win) / len(data_pad)
        X_k = X_k * scale  # 幅值补偿
        # 构造频谱对象
        Spc = Spectra(
            axis=f_Axis(len(X_k), df=self.Sig.f_axis.df / (1 + padTimes)),
            data=X_k,
            name="幅值",
            unit=self.Sig.unit,
            label=self.Sig.label,
        )
        Spc.data = np.abs(Spc.data)
        Spc.halfCut()
        return Spc

    @BaseAnalysis._plot(PlotFunc_spectrum)
    def psd(self, averageTimes: int = 10, type: str = "功率") -> Spectra:
        """
        估计带噪声功率信号在0~N/2*Δf范围内的功率分布

        功率谱中的峰值高度是信号震荡成分均方根幅值的估计值

        功率谱中的平坦部分平均是白噪声功率的估计值

        Parameters
        ----------
        averageTimes : int, default: 10
            功率谱平均次数. 平均次数越多, 频谱估计越稳定, 但谱分辨率越低. 推荐值范围: 5~20
        type : str, default: "功率"
            功率谱类型, 可选: '功率', '功率密度'

        Returns
        -------
        Spectra
            功率谱估计结果

        See Also
        --------
        - scipy.signal.welch
        """
        nperseg = max(64, len(self.Sig) // averageTimes)  # 每段长度
        # 计算功率谱
        freq, P_k = signal.welch(
            self.Sig.data,
            fs=self.Sig.t_axis.fs,
            window="boxcar",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=4 * nperseg,  # 增加频率分辨率, 缓解频谱频移问题
            return_onesided=False,
            scaling="spectrum" if type == "功率" else "density",
            average=("mean" if averageTimes < 30 else "median"),  # 平均段数多时容易受异常值影响, 改用中值平均
        )
        # 构造频谱对象
        Spc = Spectra(
            axis=f_Axis(len(P_k), df=freq[1] - freq[0], f0=freq[0]),
            data=P_k,
            name="功率" if type == "功率" else "功率密度",
            unit=self.Sig.unit + ("^2" if type == "功率" else "^2/Hz"),
            label=self.Sig.label,
        )
        Spc.halfCut()
        return Spc

    @BaseAnalysis._plot(PlotFunc_spectrum)
    def psdDiff(
        self,
        Sig_ref: Signal,
        averageTimes: int = 10,
        mode: str = "absolute",
    ) -> Spectra:
        """
        计算与参考信号的差分功率谱

        Parameters
        ----------
        Sig_ref : Signal
            参考信号
        averageTimes : int, default: 10
            功率谱平均次数. 平均次数越多, 频谱估计越稳定, 但谱分辨率越低. 推荐值范围: 5~20
        mode : Literal["absolute", "relative", "log"], default: "absolute"
            计算模式:
            - "absolute": 绝对差值 (Spc2 - Spc1)
            - "relative": 相对变化率 (Spc2 - Spc1) / Spc1
            - "log": 对数差分 (dB), 10 * log10(Spc2 / Spc1)
        """
        # 计算两个信号的功率谱
        Spc1 = Spectrum(self.Sig).psd(averageTimes=averageTimes)
        Spc2 = Spectrum(Sig_ref).psd(averageTimes=averageTimes)

        if mode == "absolute":
            Spc_diff = Spc2 - Spc1
            Spc_diff.name = "绝对差分功率"
        elif mode == "relative":
            Spc_diff = (Spc2 - Spc1) / (Spc1 + 1e-12)  # 避免除零
            Spc_diff.name = "功率变化率"
            Spc_diff.unit = "%"
        elif mode == "log":
            # 计算 dB 差值: 10 * log10(P2/P1)
            diff_data = 10 * np.log10(Spc2.data / (Spc1.data + 1e-12) + 1e-12)
            Spc_diff = Spc2.template(diff_data)
            Spc_diff.name = "对数差分功率"
            Spc_diff.unit = "dB"
        else:
            raise ValueError(f"计算模式`mode`不支持. 可选模式: 'absolute', 'relative', 'log'. 当前`mode`={mode}")

        return Spc_diff


# --------------------------------------------------------------------------------------------#
class Hilbert(BaseAnalysis):
    """
    单成分调制信号希尔伯特分析方法类

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
    - analytic() -> Signal
        计算解析信号

    - amplitude() -> Signal
        计算包络幅值

    - phase() -> Signal
        计算瞬时相位

    - frequency() -> Signal
        计算瞬时频率

    - envelopeSpectrum() -> Spectra
        计算包络幅值谱

    """

    def __init__(
        self,
        Sig: Signal,
        isLinked: bool = True,
        isPlot: bool = False,
        **kwargs,
    ):
        """
        单成分调制信号希尔伯特分析方法类

        Parameters
        ----------
        Sig : Signal
            待分析信号
        isLinked : bool, default: True
            是否链接信号原始数据
        isPlot : bool, default: False
            是否绘制分析结果图
        """
        self.Sig: Signal = Sig if isLinked else Sig.copy()
        self.isPlot: bool = isPlot
        self.plot_kwargs: dict = kwargs
        self.plot_kwargs.setdefault("isFindPeaks", True)  # 默认启用谱线峰值标记

    def analytic(self) -> Signal:
        """计算解析信号"""
        analytic: np.ndarray = signal.hilbert(self.Sig.data)
        Sig_analytic: Signal = self.Sig.template(analytic)
        Sig_analytic.name = "解析幅值"
        return Sig_analytic

    def amplitude(self) -> Signal:
        """计算包络幅值"""
        Sig_analytic: Signal = self.analytic()
        amplitude: np.ndarray = np.abs(Sig_analytic.data)
        Sig_amplitude: Signal = self.Sig.template(amplitude)
        Sig_amplitude.name = "包络幅值"
        return Sig_amplitude

    def phase(self) -> Signal:
        """计算瞬时相位"""
        Sig_analytic: Signal = self.analytic()
        phase: np.ndarray = np.angle(Sig_analytic.data)
        Sig_phase: Signal = self.Sig.template(phase)
        Sig_phase.name, Sig_phase.unit = "瞬时相位", "rad"
        return Sig_phase

    def frequency(self) -> Signal:
        """计算瞬时频率"""
        Sig_phase: Signal = self.phase()
        # 计算相位的一阶差分并除以采样间隔得到瞬时频率
        insFreq = np.gradient(Sig_phase.data, self.Sig.t_axis.dt) / (2 * np.pi)
        Sig_freq = self.Sig.template(insFreq)
        Sig_freq.name, Sig_freq.unit = "瞬时频率", "Hz"
        return Sig_freq

    def envelopeSpectrum(self) -> Spectra:
        """计算包络幅值谱"""
        Sig_amplitude: Signal = self.amplitude()
        Spc_envelope: Spectra = Spectrum(Sig_amplitude).cft(padTimes=3)
        Spc_envelope.name = "包络幅值"
        return Spc_envelope
