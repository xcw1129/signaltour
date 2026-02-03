"""
# StatsTrendAnalysis: 时域统计分析模块, 提供时域统计趋势等方法

---

## 可用的接口

    - class:
        - `StatsTrendAnalysis`: 信号时域统计分析方法
"""

__all__ = ["StatsTrendAnalysis"]

from .._Assist_Module.Dependencies import Optional, Tuple, np, stats
from .._Plot_Module.LinePlot import PlotFunc_waveform
from .._Signal_Module.core import Series, t_Axis
from .._Signal_Module.SignalSample import slice
from .core import BaseAnalysis


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class StatsTrendAnalysis(BaseAnalysis):
    """
    信号时域统计分析类

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
    - evaluate() -> dict
        计算信号整段的各项时域统计指标
    - trend(type: str, segNum: Optional[int] = None, tperseg: Optional[float] = None)
            -> Series
        滑窗法计算信号指定统计值的时域趋势
    - distribution(bins: int = 100, density: bool = True) -> Tuple[np.ndarray, np.ndarray]
        计算信号的幅值概率分布(直方图)
    """

    _stats_func_dict = {
        "均值": np.mean,
        "标准差": np.std,
        "有效值": lambda x: np.sqrt(np.mean(np.square(x))),
        "偏度": stats.skew,
        "峭度": lambda x: stats.kurtosis(x, fisher=False),
        "峰值": lambda x: np.max(np.abs(x)),
        "峰峰值": np.ptp,
        "方根幅值": lambda x: np.square(np.mean(np.sqrt(np.abs(x)))),
        "平均幅值": lambda x: np.mean(np.abs(x)),
        "波形因子": lambda x: np.sqrt(np.mean(np.square(x))) / np.mean(np.abs(x)),
        "峰值因子": lambda x: np.max(np.abs(x)) / np.sqrt(np.mean(np.square(x))),
        "脉冲因子": lambda x: np.max(np.abs(x)) / np.mean(np.abs(x)),
        "裕度因子": lambda x: np.max(np.abs(x)) / np.square(np.mean(np.sqrt(np.abs(x)))),
    }

    def evaluate(self) -> dict:
        """
        计算信号整段的各项时域统计指标

        Returns
        -------
        dict
            包含各项统计指标名称和值的字典
        """
        return {name: func(self.Sig.data) for name, func in self._stats_func_dict.items()}

    def distribution(self, bins: int = 100, density: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算信号的幅值概率分布(直方图)

        Parameters
        ----------
        bins : int, default: 100
            直方图分柱数
        density : bool, default: True
            是否返回概率密度(True)还是计数(False)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (概率密度/计数, 幅值分界点)
        """
        return np.histogram(self.Sig.data, bins=bins, density=density)

    @BaseAnalysis._plot(PlotFunc_waveform)
    def trend(self, type: str, segNum: Optional[int] = None, tperseg: Optional[float] = None) -> Series:
        """
        滑窗法计算信号指定统计值的时域趋势

        Parameters
        ----------
        type : str
            统计值类型, 支持: "均值", "标准差", "有效值", "偏度", "峭度", "峰值", "峰峰值",
            "方根幅值", "平均幅值", "波形因子", "峰值因子", "脉冲因子", "裕度因子"
        segNum : int, optional
            分段数
        tperseg : float, optional
            分段时长

        Returns
        -------
        Series
            统计值时域趋势序列
        """
        # 信号滑动切片
        seg_data_list, seg_time = slice(
            self.Sig, segNum=segNum, tperseg=tperseg, pad_mode="edge"
        )  # 默认50%重叠, 边界常数填充
        # 计算所有分段统计值
        if type not in self._stats_func_dict:
            raise ValueError(f"type={type}: 不支持的统计值类型")
        stat_values = np.apply_along_axis(self._stats_func_dict[type], axis=1, arr=seg_data_list)
        dt = seg_time[1] - seg_time[0] if len(seg_time) > 1 else self.Sig.t_axis.dt
        Srs_stats = Series(
            t_Axis(len(stat_values), t0=seg_time[0], dt=dt),
            stat_values,
            label=f"{self.Sig.label}滑动{type}",
        )  # 统计值量纲不确定, 不设置name和unit属性
        return Srs_stats
