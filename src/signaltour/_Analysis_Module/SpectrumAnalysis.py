"""
# SpectrumAnalysis: 平稳信号谱分析模块, 提供多种基于DFT的频谱分析方法

---

## 可用的接口

    - function:
        - `window`: 生成各类窗函数的整周期采样序列
        - `find_spectralines`: 对序列数据进行谱线类局部峰值检测
    - class:
        - `SpectrumAnalysis`: 平稳信号频谱分析方法
"""

__all__ = [
    "window",
    "find_spectralines",
    "SpectrumAnalysis",
]

from .._Assist_Module.Dependencies import Callable, Optional, fft, linalg, np, signal
from .._Plot_Module.LinePlot import spectrum_PlotFunc
from .._Signal_Module.core import Signal, Spectra, f_Axis
from .core import BaseAnalysis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
def window(
    num: int,
    type: str = "汉宁窗",
    winParam: Optional[float] = None,
    symmetric: bool = False,
    padding: Optional[int] = None,
    func: Optional[Callable] = None,
) -> np.ndarray:
    """
    生成各类窗函数的整周期采样序列

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
    - scipy.signal.get_window : 各种窗函数序列生成

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
    if type not in window_name.keys():
        raise ValueError(f"type={type}: 不支持的窗函数类型")
    elif type == "自定义窗":
        n = np.arange(num)  # n=0,1,2,3,...,N-1
        if symmetric:
            data = func(n / (num - 1))
        else:
            data = func(n / num)
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
        data = np.pad(data, padding, mode="constant")  # 双边各填充padding点, 共延长2*padding点
    return data


def find_spectralines(
    data: np.ndarray,
    threshold: float = 0.8,
    distance: float = 0.01,
) -> np.ndarray:
    """
    对序列数据进行谱线类局部峰值检测

    Parameters
    ----------
    data : np.ndarray
        一维序列数据, 元素为非负实数
    threshold : float, default: 0.8
        邻域稀疏度阈值, 输入范围: (1/sqrt(d*2+1), 1)
    distance : float, default: 0.01
        峰值最小间距, 若<1则表示数据总长度的比例, 若>1则表示数据点数

    Returns
    -------
    np.ndarray
        满足谱线特征的峰值索引数组

    Notes
    -----
    方法使用signal.find_peaks函数初步筛选局部峰值点, 然后结合谱线邻域稀疏度判据进行二次筛选。
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
    lines_idx, _ = signal.find_peaks(data, distance=distance + 1)  # +1确保峰值间距至少为distance
    # 二次筛选谱线类峰值
    valid_lines_idx = []
    for idx in lines_idx:
        # 取出峰值邻域数据段
        seg = data[max(0, idx - distance) : min(len(data), idx + distance + 1)]
        if max(seg) != data[idx]:
            continue  # 非峰值点跳过
        # 计算稀疏度指标
        seg_s = sparsity(seg)
        # 邻域稀疏的峰值判定为谱线
        if seg_s < threshold:
            valid_lines_idx.append(idx)
    valid_lines_idx = np.array(valid_lines_idx)
    return valid_lines_idx


# --------------------------------------------------------------------------------------------#
class SpectrumAnalysis(BaseAnalysis):
    """
    平稳信号频谱分析方法

    Attributes
    ----------
    Sig : Signal
        待分析信号
    isPacket : bool
        是否对结果进行类型封装
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        自定义绘图参数

    Methods
    -------
    - cycleconvolve : 计算两个序列数据的循环卷积
    - convolve : 计算两个序列数据的线性卷积
    - dft : 计算序列数据的离散傅里叶变换
    - idft : 计算序列数据的逆离散傅里叶变换
    - ft : 计算能量信号在0~N/2*Δf范围傅里叶变换的离散近似
    - cft : 计算功率信号在0~N/2*Δf范围傅里叶级数系数的离散近似
    - psd : 估计带噪声功率信号在0~N/2*Δf范围的功率谱分布
    - enveSpectra : 计算信号的希尔伯特包络幅值谱
    """

    # ----------------------------------------------------------------------------------------#
    # 离散卷积相关方法
    @staticmethod
    def cycleconvolve(x: np.ndarray, y: np.ndarray, method="fft") -> np.ndarray:
        """
        计算两个序列数据的循环卷积, 该卷积方式满足DFT的卷积定理

        两个序列长度必须相等

        method='direct'时输入参数的长度推荐不超过16384, 避免计算过程占用2GB以上内存

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
            raise ValueError("循环卷积要求输入序列长度相等")
        if method == "fft":
            # 通过频域乘计算循环卷积
            X_f = SpectrumAnalysis.dft(x)
            Y_f = SpectrumAnalysis.dft(y)
            Z_f = X_f * Y_f
            # 还原时域得卷积结果
            z_n = SpectrumAnalysis.idft(Z_f).real
            return z_n
        elif method == "direct":
            # 直接计算循环卷积
            Y_pad_trans = linalg.circulant(np.conj(y))  # 循环卷积矩阵, circulant自动转置
            z = np.dot(Y_pad_trans, x)  # 计算循环卷积的一个周期
            return z
        else:
            raise ValueError(f"method={method}: 不支持的卷积计算方法:")

    @staticmethod
    def _convolve(x: np.ndarray, y: np.ndarray, method="fft") -> np.ndarray:
        """
        计算两个序列数据的线性卷积, 输出长度默认为 len(x)+len(y)-1, 即"full"模式

        该方法仅用于演示序列数据的线性卷积计算过程, 实际计算请使用 SpectrumAnalysis.convolve

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
        # 延拓数据以使循环卷积与线性卷积等价
        x_pad = np.pad(x, (0, N - len(x)), mode="constant")
        y_pad = np.pad(y, (0, N - len(y)), mode="constant")
        # 执行循环卷积
        z = SpectrumAnalysis.cycleconvolve(x_pad, y_pad, method=method)
        return z

    @staticmethod
    def convolve(x: np.ndarray, y: np.ndarray, mode: str = "full") -> np.ndarray:
        """计算两个序列数据的线性卷积"""
        if len(x) // len(y) >= 10:  # 当一个序列远长于另一个序列时(例如FIR滤波), 使用重叠相加法进行快速卷积
            z = signal.oaconvolve(x, y, mode=mode)
        else:
            z = signal.convolve(x, y, mode=mode)  # 自动选择最快的卷积方法: direct or fft
        return z

    # ----------------------------------------------------------------------------------------#
    # 傅里叶变换相关方法
    @staticmethod
    def _dft(data: np.ndarray) -> np.ndarray:
        """
        计算序列数据的离散傅里叶变换

        该方法仅用于演示DFT计算过程, 实际计算请使用 SpectrumAnalysis.dft

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

        该方法仅用于演示IDFT计算过程, 实际计算请使用 SpectrumAnalysis.idft

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

    @BaseAnalysis._plot(spectrum_PlotFunc)
    def ft(self, symmetric: bool = False) -> Spectra:
        """计算能量信号在0~N/2*Δf范围傅里叶变换的离散近似"""
        # 计算傅里叶变换: FT=DFT*Δt
        # FT结果幅值随序列fs变化而变化, 因为信号被伸缩, 总能量分布变化
        X_f = SpectrumAnalysis.dft(self.Sig.data) * self.Sig.t_axis.dt
        # 构造频谱对象
        Spc = Spectra(
            axis=self.Sig.f_axis,
            data=X_f,
            name="密度",
            unit=self.Sig.unit + "/Hz",
            label=self.Sig.label,
        )
        if symmetric:
            Spc.data = fft.fftshift(Spc.data)
            freq = fft.fftshift(fft.fftfreq(len(Spc), d=self.Sig.t_axis.dt))
            Spc.f_axis.f0, Spc.f_axis.df = freq[0], freq[1] - freq[0]
        if self.isPlot:
            return np.abs(Spc)
        return Spc

    @BaseAnalysis._plot(spectrum_PlotFunc)
    def cft(self, winType: str = "汉宁窗", padTimes: int = 3) -> Spectra:
        """
        计算功率信号在0~N/2*Δf范围傅里叶级数系数的离散近似

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
        win = window(num=len(self.Sig), type=winType, padding=padTimes * len(self.Sig) // 2)
        scale = 1 / np.mean(win)  # 幅值补偿因子
        # 计算傅里叶级数系数: CFT=DFT/N
        # CFT结果幅值不随序列fs变化而变化, 因为信号总功率分布不变
        # CFT结果幅值随序列padding而变化, 因为信号被稀释, 总功率分布变化
        data_pad = np.pad(self.Sig.data, padTimes * len(self.Sig) // 2, mode="constant")
        X_k = SpectrumAnalysis.dft(data_pad * win) / len(data_pad)
        X_k = X_k * scale  # 幅值补偿
        # 构造频谱对象
        Spc = Spectra(
            axis=f_Axis(len(X_k), df=self.Sig.f_axis.df / (1 + padTimes)),
            data=X_k,
            name="系数",
            unit=self.Sig.unit,
            label=self.Sig.label,
        )
        if self.isPlot:
            return np.abs(Spc.halfCut())  # 单边截断
        return Spc

    # ----------------------------------------------------------------------------------------#
    # 噪声信号谱估计方法
    @BaseAnalysis._plot(spectrum_PlotFunc)
    def psd(self, averageTimes: int = 10, type: str = "功率") -> Spectra:
        """
        估计带噪声功率信号在0~N/2*Δf范围的功率分布

        功率谱中的峰值高度是信号震荡成分均方根幅值的估计值

        功率谱中的平坦部分平均是白噪声功率的估计值
        """
        nperseg = max(64, len(self.Sig) // averageTimes)  # 每段长度
        # 计算功率谱
        freq, P_k = signal.welch(
            self.Sig._data,
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
        if self.isPlot:
            return Spc.halfCut()  # 单边截断
        return Spc

    # ----------------------------------------------------------------------------------------#
    # 其它平稳谱分析方法
    @BaseAnalysis._plot(spectrum_PlotFunc)
    def enveSpectra(self) -> Spectra:
        """计算信号的希尔伯特包络幅值谱"""
        # 计算包络幅值
        analytic = signal.hilbert(self.Sig)
        envelope = np.abs(analytic)
        # 计算幅值谱
        Spc_Amp = SpectrumAnalysis(self.Sig.template(envelope)).cft(padTimes=3)
        Spc_Amp = np.abs(Spc_Amp.halfCut())
        Spc_Amp.name = "包络幅值"
        return Spc_Amp

    @BaseAnalysis._plot(spectrum_PlotFunc)
    def DiffSpectra(
        self,
        Sig_ref: Signal,
        averageTimes: int = 10,
        mode: str = "absolute",
    ) -> Spectra:
        """
        计算信号与输入参考信号的差分功率谱

        Parameters
        ----------
        Sig_ref : Signal
            参考信号
        averageTimes : int, default: 10
            计算功率谱时的平均次数
        mode : str, default: "absolute"
            计算模式:
            - "absolute": 绝对差值 (Spc2 - Spc1)
            - "relative": 相对变化率 (Spc2 - Spc1) / Spc1
            - "log": 对数差分 (dB), 10 * log10(Spc2 / Spc1)
        """
        if self.Sig.t_axis != Sig_ref.t_axis:
            raise ValueError("计算差分谱的两个信号的时间轴必须一致")

        # 计算两个信号的功率谱
        Spc1 = SpectrumAnalysis(self.Sig).psd(averageTimes=averageTimes).halfCut()
        Spc2 = SpectrumAnalysis(Sig_ref).psd(averageTimes=averageTimes).halfCut()

        if mode == "absolute":
            Spc_diff = Spc2 - Spc1
            Spc_diff.name = "绝对差分功率"
        elif mode == "relative":
            Spc_diff = (Spc2 - Spc1) / (Spc1 + 1e-12)  # 避免除零
            Spc_diff.name = "功率变化率"
            Spc_diff.unit = "%"
        elif mode == "log":
            # 计算 dB 差值: 20 * log10(P2/P1)
            diff_data = 20 * np.log10(Spc2.data / (Spc1.data + 1e-12) + 1e-12)
            Spc_diff = Spc2.template(diff_data)
            Spc_diff.name = "对数差分功率"
            Spc_diff.unit = "dB"
        else:
            raise ValueError(f"不支持的模式: {mode}")

        return Spc_diff
