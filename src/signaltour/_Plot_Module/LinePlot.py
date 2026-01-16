"""
# LinePlot

---

## 可用的接口

    - function:
        - `waveform_PlotFunc`: 信号波形图绘制函数
        - `spectrum_PlotFunc`: 频谱绘制函数
    - class:
        - `LinePlot`: 波形图、谱图等一维线条图绘制绘图类
"""

__all__ = [
    "LinePlot",
    "waveform_PlotFunc",
    "spectrum_PlotFunc",
]

from .._Assist_Module.Dependencies import List, Optional, Self, np
from .._Signal_Module.core import Series, Signal, Spectra
from .core import BasePlot
from .PlotPlugin import PeakfinderPlugin, PosNagMaskPlugin

# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#


class LinePlot(BasePlot):
    """
    波形图、谱图等一维线条图绘制绘图类

    Methods
    -------
    waveform(Srs: Series | List[Series], **kwargs) -> LinePlot
        注册一个时域波形图的绘制任务
    spectrum(Spc: Spectra, **kwargs) -> LinePlot
        注册一个谱图的绘制任务
    """

    def __init__(
        self,
        title: str = "",
        scheme: str = "LinePlot1",
        autoRestore: bool = True,
        ncols: int = 1,
        figsize: Optional[tuple] = None,
        **kwargs,
    ):
        """
        波形图、谱图等一维线条图绘制绘图类

        Parameters
        ----------
        title : str, default: ""
            总图标题
        scheme : str, default: "default"
            绘图风格配置方案
        autoRestore : bool, default: True
            是否自动恢复用户原始rcParams配置
        ncols : int, default: 1
            多图绘制时的子图列数
        figsize : Optional
            所有子图共享的图形大小
        """
        super().__init__(
            title=title,
            scheme=scheme,
            autoRestore=autoRestore,
            ncols=ncols,
            figsize=figsize,
            **kwargs,
        )

    def waveform(self, Srs: Series | List[Series], **kwargs) -> "LinePlot":
        """
        注册一个时域波形图的绘制任务

        Parameters
        ----------
        Srs : Series or List[Series]
            需要绘制的序列数据，支持单个 Series 对象或 Series 对象列表输入
        **kwargs :
            该子图特定的绘图参数（如 title, xlabel, ylim 等），会覆盖初始化时的全局设置

        Returns
        -------
        LinePlot
            返回绘图对象本身，以支持链式调用

        Raises
        ------
        ValueError
            输入数据不是 Series 对象或 Series 对象列表
        """

        # ------------------------------------------------------------------------------------#
        # 时域波形绘制函数: 通过任务队列传递到绘图引擎
        def _draw_waveform(ax, data, kwargs):
            SrsList = data.get("SrsList")
            for Srs in SrsList:
                kwargs_plot = kwargs.get("plot", {})
                ax.plot(
                    Srs._axis(),
                    Srs.data,
                    label=Srs.label,
                    **kwargs_plot.get(Srs.label, {}),
                )
            if len(SrsList) > 1:
                ax.legend(loc="best")

        # ------------------------------------------------------------------------------------#
        # 波形图绘制个性化设置
        if not isinstance(Srs, list):
            SrsList = [Srs]
        else:
            SrsList = Srs
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "xlabel": SrsList[0]._axis.label,
            "xlim": SrsList[0]._axis.lim,
            "ylabel": (f"{SrsList[0].name}" + f"[{SrsList[0].unit}]" if SrsList[0].unit else ""),
            "title": f"{SrsList[0].label}波形图",
        }
        task_kwargs.update(self.kwargs)
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": {"SrsList": SrsList},
            "kwargs": task_kwargs,
            "function": _draw_waveform,
            "plugins": [],
        }
        self.tasks.append(task)
        return self

    def spectrum(self, Spc: Spectra, isFindPeaks: bool = False, isPosNeg: bool = False, **kwargs) -> "LinePlot":
        """
        注册一个谱图的绘制任务

        Parameters
        ----------
        Spc : Spectra
            需要绘制的谱对象
        isFindPeaks: bool, default: False
            是否需要在谱中标注峰值谱线
        isPosNeg: bool, default: False
            是否需要对谱线正负值进行不同颜色显示

        Returns
        -------
        LinePlot
            返回绘图对象本身，以支持链式调用
        """

        # ------------------------------------------------------------------------------------#
        # 谱图绘制函数: 通过任务队列传递到绘图引擎
        def _draw_spectrum(ax, data, kwargs):
            Spc = data["Spc"]
            kwargs_plot = kwargs.get("plot", {})
            f, d = Spc.f_axis(), Spc.data.real  # 取实部舍去计算误差
            ax.plot(f, d, **kwargs_plot.get(Spc.label, {}))

        # ------------------------------------------------------------------------------------#
        # 频谱绘制个性化设置
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "xlabel": Spc.f_axis.label,
            "xlim": Spc.f_axis.lim,
            "ylabel": f"{Spc.name}" + f"[{Spc.unit}]" if Spc.unit else "",
            "title": f"{Spc.label}{Spc.name}谱",
        }
        task_kwargs.update(self.kwargs)
        task_kwargs.update(kwargs)
        task_plugins = []
        if isFindPeaks:
            task_plugins.append(
                PeakfinderPlugin(
                    threshold=task_kwargs.get("plugin_threshold", 0.85),
                    distance=task_kwargs.get("plugin_distance", 0.01),
                )
            )
        if isPosNeg:
            task_plugins.append(
                PosNagMaskPlugin(
                    sensitivity=task_kwargs.get("plugin_sensitivity", 0.05),
                )
            )
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": {"Spc": Spc},
            "kwargs": task_kwargs,
            "function": _draw_spectrum,
            "plugins": task_plugins,
        }
        self.tasks.append(task)
        return self

    def trendsCompare(
        self,
        conditions: list,
        trends: List[Series],
        errors: Optional[Series] = None,
        **kwargs,
    ) -> "LinePlot":
        """注册一个多趋势线条对比图的绘制任务"""

        # ------------------------------------------------------------------------------------#
        # 趋势线条绘制函数: 通过任务队列传递到绘图引擎
        def _draw_trendsCompare(ax, data, kwargs):
            conditions, trends, errors = (
                data["conditions"],
                data["trends"],
                data["errors"],
            )
            kwargs_plot = kwargs.get("plot", {})
            marker_list = ["o", "s", "^", "D", "*"]
            for idx, trend in enumerate(trends):
                err = errors[idx] if errors is not None else None
                ax.errorbar(
                    conditions,
                    trend._data,
                    yerr=err._data if err is not None else None,
                    capsize=5,
                    ecolor="gray",
                    label=trend.label,
                    marker=marker_list[idx % len(marker_list)],
                    **kwargs_plot.get(trend.label, {}),
                )
            ax.legend(loc="best")

        # ------------------------------------------------------------------------------------#
        # 趋势线条绘制个性化设置

        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "xlabel": trends[0]._axis.label,
            "ylabel": (f"{trends[0].name}" + f"[{trends[0].unit}]" if trends[0].unit else ""),
            "title": trends[0].name + "趋势对比图",
        }
        task_kwargs.update(self.kwargs)
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": {"conditions": conditions, "trends": trends, "errors": errors},
            "kwargs": task_kwargs,
            "function": _draw_trendsCompare,
            "plugins": [],
        }
        self.tasks.append(task)
        return self

    def orbit(self, Srs_X: Series, Srs_Y: Series, **kwargs) -> Self:
        """注册一个平面轨迹图的绘制任务"""

        # ------------------------------------------------------------------------#
        # 绘图函数
        def _draw_orbit(ax, data, kwargs):
            Srs_X, Srs_Y = data["Srs_X"], data["Srs_Y"]
            kwargs_plot = kwargs.get("plot", {})
            ax.plot(
                Srs_X._data,
                Srs_Y._data,
                linestyle="-",
                marker=".",
                markersize=2,
                **kwargs_plot,
            )
            ax.set_aspect("equal")
            ax.grid(False)

        # ------------------------------------------------------------------------#
        # 绘图个性化设置
        if Srs_X._axis != Srs_Y._axis:
            raise ValueError("用于绘制轨迹图的两个序列对象必须具有相同的坐标轴！")
        # ------------------------------------------------------------------------#
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "figsize": (8, 6),
            "title": "平面轨迹图",
            "xlabel": f"{Srs_X.name}[{Srs_X.unit}]",
            "ylabel": f"{Srs_Y.name}[{Srs_Y.unit}]",
        }
        task_kwargs.update(self.kwargs)
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": {"Srs_X": Srs_X, "Srs_Y": Srs_Y},
            "kwargs": task_kwargs,
            "function": _draw_orbit,
            "plugins": [],
        }
        self.tasks.append(task)
        return self


# --------------------------------------------------------------------------------------------#
# LinePlot类绘图方法函数形式调用接口
def waveform_PlotFunc(Srs: Series, **kwargs) -> tuple:
    """信号波形图绘制函数"""
    fig, ax = LinePlot().waveform(Srs, **kwargs).show(pattern="return")
    fig.show()
    return fig, ax


def spectrum_PlotFunc(Spc: Spectra, **kwargs) -> tuple:
    """频谱绘制函数"""
    fig, ax = LinePlot().spectrum(Spc, **kwargs).show(pattern="return")
    fig.show()
    return fig, ax


def decResult_PlotFunc(
    SigList_deco: List[Signal],
    **kwargs,
) -> tuple:
    """信号分解结果总览图绘制函数"""
    # 合成原始信号
    data = np.sum(SigList_deco, axis=0)
    Sig = SigList_deco[0].template(data).set_label("原始信号")
    Spc = (np.abs(Sig.to_Spectra()) / len(Sig)).halfCut()
    # 准备绘图参数
    # 设置总图标题
    title = kwargs.pop("title", "分解结果")
    # 设置xlim
    xlim_waveform_allax = kwargs.pop("xlim_waveform", None)
    xlim_spectrum_allax = kwargs.pop("xlim_spectrum", None)
    # 设置ylim
    ylim_waveform_allax = (
        np.min(Sig) - 0.1 * np.ptp(Sig),
        np.max(Sig) + 0.1 * np.ptp(Sig),
    )  # 设置与原始信号相同的ylim
    ampRange = np.max(Spc) - np.min(Spc)
    ylim_spectrum_allax = (np.min(Spc) - 0.1 * ampRange, np.max(Spc) + 0.1 * ampRange)  # 设置频谱y轴范围为110%
    # --------------------------------------------------------------------------------#
    # 绘制分解结果
    plot = LinePlot(title=title, ncols=2, **kwargs)
    # 绘制原始信号时域波形与频谱
    plot.waveform(Sig, title="原始信号时域波形", xlim=xlim_waveform_allax, ylim=ylim_waveform_allax)
    plot.waveform(Spc, title="原始信号幅值谱", xlim=xlim_spectrum_allax, ylim=ylim_spectrum_allax)
    # 绘制各分解成分时域波形与频谱
    for Sig_deco in SigList_deco:
        # 绘制时域波形
        plot.waveform(Sig_deco, title=f"{Sig_deco.label}时域波形", xlim=xlim_waveform_allax, ylim=ylim_waveform_allax)
        # 绘制频谱
        Spc_deco = (np.abs(Sig_deco.to_Spectra()) / len(Sig_deco)).halfCut()
        plot.waveform(Spc_deco, title=f"{Sig_deco.label}幅值谱", xlim=xlim_spectrum_allax, ylim=ylim_spectrum_allax)
    fig, ax = plot.show(pattern="return")
    fig.show()
    return fig, ax
