"""
# StatsTrendAnalysis: 时域统计分析模块, 提供时域统计趋势等方法

---

## 可用的接口

    - class:
        - `StatsTrendAnalysis`: 信号时域统计分析方法
"""

__all__ = ["StatsTrendAnalysis"]

from .._Assist_Module.Dependencies import Optional, np, stats
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
    信号时域统计分析方法

    Attributes
    ----------
    Sig : Signal
        待分析信号
    isPlot : bool
        是否绘制分析结果图
    isInputCheck : bool
        是否进行输入值检查
    plot_kwargs : dict
        自定义绘图参数

    Methods
    -------
    - trend(type: str, segNum: Optional[int] = None, tperseg: Optional[float] = None)
            -> Series
        滑窗法计算信号指定统计值的时域趋势
    """

    @BaseAnalysis._plot(PlotFunc_waveform)
    def trend(self, type: str, segNum: Optional[int] = None, tperseg: Optional[float] = None) -> Series:
        """
        滑窗法计算信号指定统计值的时域趋势

        Parameters
        ----------
        type : str
            统计值类型, 支持: "均值", "标准差", "有效值", "偏度", "峭度", "峰峰值"
        segNum : int, optional
            分段数
        tperseg : float, optional
            分段时长

        Returns
        -------
        Series
            统计值时域趋势序列
        """
        seg_data_list, seg_time = slice(self.Sig, segNum=segNum, tperseg=tperseg, pad_mode="edge")
        stats_func_dict = {
            "均值": np.mean,
            "标准差": np.std,
            "有效值": lambda x: np.sqrt(np.mean(np.square(x))) / np.sqrt(np.mean(np.square(x))),
            "偏度": stats.skew,
            "峭度": stats.kurtosis,
            "峰峰值": np.ptp,
        }
        if type not in stats_func_dict:
            raise ValueError(f"type={type}: 不支持的统计值类型")
        stat_values = np.apply_along_axis(stats_func_dict[type], axis=1, arr=seg_data_list)
        Srs_stats = Series(
            t_Axis(len(stat_values), t0=seg_time[0], dt=seg_time[1] - seg_time[0]),
            stat_values,
            label=f"{self.Sig.label}滑动{type}",
        )
        return Srs_stats
