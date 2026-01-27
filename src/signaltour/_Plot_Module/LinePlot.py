"""
# LinePlot

---

## 可用的接口

    - function:
        - `PlotFunc_waveform`: 信号波形图绘制函数
        - `PlotFunc_spectrum`: 频谱绘制函数
        - `PlotFunc_decResult`: 信号分解结果总览图绘制函数
    - class:
        - `LinePlot`: 波形图、谱图等一维线条图绘制绘图类
"""

__all__ = [
    "LinePlot",
    "PlotFunc_waveform",
    "PlotFunc_spectrum",
    "PlotFunc_decResult",
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

    Attributes
    ----------
    figure : plt.Figure
        当前绘图流程操作的 Figure 对象
    axs : np.ndarray
        当前绘图流程操作的 Axes 对象二维数组
    title : str
        总图标题
    scheme : str
        绘图风格配置方案
    autoRestore : bool
        是否自动恢复用户原始rcParams配置
    ncols : int
        多图绘制时的子图列数
    figsize : tuple
        所有子图共享的图形大小
    kwargs : dict
        全局绘图参数, 一般初始化后不再修改
    tasks : deque
        绘图任务队列, 存储所有待绘制图形相关信息
    last_task : dict
        最新添加的绘图任务

    Methods
    -------
    - init_Plot_rcParams(scheme: str = "default")
            -> None
        设置绘图风格配置方案, 并保存原始配置用于恢复

    - restore_User_rcParams()
            -> None
        恢复原始绘图风格配置方案

    - set_params_to_task(**kwargs)
            -> Self
        为最新添加的绘图任务设置专属参数

    - add_plugin_to_task(plugin: PlotPlugin)
            -> Self
        为最新添加的绘图任务添加一个插件

    - show(pattern: str = "plot", filename="Plot.png", save_format="png")
            -> tuple
        执行所有已注册的绘图任务并显示/返回/保存最终图形

    - canvas()
            -> tuple
        生成当前绘图对象的空白画布

    - waveform(Srs: Series | List[Series], **kwargs)
            -> Self
        信号时域波形图绘制函数

    - spectrum(Spc: Spectra, **kwargs)
            -> Self
        频谱绘制函数

    - trendsCompare(conditions: list, trends: List[Series], errors: Optional[Series] = None, **kwargs)
            -> Self
        趋势线条对比图绘制函数

    - orbit(Srs_X: Series, Srs_Y: Series, **kwargs)
            -> Self
        平面轨迹图绘制函数
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
        波形图、谱图等一维线条图绘制方法类

        Parameters
        ----------
        title : str, default: ""
            总图标题
        scheme : str, default: "LinePlot1"
            绘图风格配置方案
        autoRestore : bool, default: True
            是否自动恢复用户原始rcParams配置
        ncols : int, default: 1
            多图绘制时的子图列数
        figsize : Optional[tuple]
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

    def waveform(self, Sig: Signal | List[Signal], **kwargs) -> Self:
        """
        信号时域波形图绘制函数

        Parameters
        ----------
        Sig : Signal or List[Signal]
            待绘制信号，支持单个 Signal 对象或 Signal 对象列表输入
        """

        # ------------------------------------------------------------------------------------#
        # 绘制函数: 通过任务队列传递
        def _draw_waveform(ax, data, kwargs):
            SigList = data.get("SigList")
            for Sig in SigList:
                kwargs_plot = kwargs.get("plot") or {}
                ax.plot(
                    Sig._axis(),
                    Sig.data,
                    label=Sig.label,
                    **kwargs_plot.get(Sig.label) or {},
                )
            if len(SigList) > 1:
                ax.legend(loc="best")

        # ------------------------------------------------------------------------------------#
        # 绘制设置
        if not isinstance(Sig, list):
            SigList = [Sig]
        else:
            SigList = Sig
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "xlabel": SigList[0]._axis.label,
            "xlim": SigList[0]._axis.lim,
            "ylabel": (f"{SigList[0].name}" + f"[{SigList[0].unit}]" if SigList[0].unit else ""),
            "title": f"{SigList[0].label}波形图",
        }  # 方法默认设置
        task_kwargs.update(self.kwargs)  # 全局kwargs
        task_kwargs.update(kwargs)  # 用户传入kwargs
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": {"SigList": SigList},
            "kwargs": task_kwargs,
            "function": _draw_waveform,
            "plugins": [],
        }
        self.tasks.append(task)
        return self

    def spectrum(self, Spc: Spectra, isFindPeaks: bool = False, isPosNeg: bool = False, **kwargs) -> Self:
        """
        频谱绘制函数

        Parameters
        ----------
        Spc : Spectra
            待绘制频谱
        isFindPeaks: bool, default: False
            是否在谱中标注峰值谱线
        isPosNeg: bool, default: False
            是否对谱线正负值进行不同颜色显示
        """

        # ------------------------------------------------------------------------------------#
        # 绘制函数: 通过任务队列传递
        def _draw_spectrum(ax, data, kwargs):
            Spc = data["Spc"]
            kwargs_plot = kwargs.get("plot") or {}
            f, d = Spc.f_axis(), Spc.data.real  # 取实部舍去计算误差
            ax.plot(f, d, **kwargs_plot.get(Spc.label) or {})

        # ------------------------------------------------------------------------------------#
        # 绘制设置
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "xlabel": Spc.f_axis.label,
            "xlim": Spc.f_axis.lim,
            "ylabel": f"{Spc.name}" + f"[{Spc.unit}]" if Spc.unit else "",
            "title": f"{Spc.label}{Spc.name}谱",
        }  # 方法默认设置
        task_kwargs.update(self.kwargs)  # 全局kwargs
        task_kwargs.update(kwargs)  # 用户传入kwargs
        task_plugins = []
        if isFindPeaks:
            task_plugins.append(
                PeakfinderPlugin(
                    threshold=task_kwargs.get("plugin_threshold") or 0.85,
                    distance=task_kwargs.get("plugin_distance") or 0.01,
                )
            )
        if isPosNeg:
            task_plugins.append(
                PosNagMaskPlugin(
                    sensitivity=task_kwargs.get("plugin_sensitivity") or 0.05,
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
    ) -> Self:
        """趋势线条对比图绘制函数"""

        # ------------------------------------------------------------------------------------#
        # 绘制函数: 通过任务队列传递
        def _draw_trendsCompare(ax, data, kwargs):
            conditions, trends, errors = (
                data["conditions"],
                data["trends"],
                data["errors"],
            )
            kwargs_plot = kwargs.get("plot") or {}
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
                    **kwargs_plot.get(trend.label) or {},
                )
            ax.legend(loc="best")

        # ------------------------------------------------------------------------------------#
        # 绘制设置
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "xlabel": trends[0]._axis.label,
            "ylabel": (f"{trends[0].name}" + f"[{trends[0].unit}]" if trends[0].unit else ""),
            "title": trends[0].name + "趋势对比图",
        }  # 方法默认设置
        task_kwargs.update(self.kwargs)  # 全局kwargs
        task_kwargs.update(kwargs)  # 用户传入kwargs
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
        """平面轨迹图绘制函数"""

        # ------------------------------------------------------------------------#
        # 绘制函数: 通过任务队列传递
        def _draw_orbit(ax, data, kwargs):
            Srs_X, Srs_Y = data["Srs_X"], data["Srs_Y"]
            kwargs_plot = kwargs.get("plot") or {}
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
        # 绘制设置
        if Srs_X._axis != Srs_Y._axis:
            raise ValueError("用于绘制轨迹图的两个序列对象必须具有相同的坐标轴！")
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "figsize": (8, 6),
            "title": "平面轨迹图",
            "xlabel": f"{Srs_X.name}[{Srs_X.unit}]",
            "ylabel": f"{Srs_Y.name}[{Srs_Y.unit}]",
        }  # 方法默认设置
        task_kwargs.update(self.kwargs)  # 全局kwargs
        task_kwargs.update(kwargs)  # 用户传入kwargs
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
# 基于 LinePlot 类的高级绘图函数
def PlotFunc_waveform(Srs: Series, **kwargs) -> tuple:
    """信号波形图绘制函数"""
    fig, ax = LinePlot().waveform(Srs, **kwargs).show(pattern="return")
    fig.show()
    return fig, ax


def PlotFunc_spectrum(Spc: Spectra, **kwargs) -> tuple:
    """频谱绘制函数"""
    fig, ax = LinePlot().spectrum(Spc, **kwargs).show(pattern="return")
    fig.show()
    return fig, ax


def PlotFunc_decResult(
    SigList_deco: List[Signal],
    reco: bool = False,
    spectrum: bool = True,
    **kwargs,
) -> tuple:
    """信号分解结果总览图绘制函数"""
    # 合成原始信号
    if reco:
        data = np.sum(SigList_deco, axis=0)
        Sig = SigList_deco[0].template(data).set_label("原始信号")
        Spc = np.abs(Sig.to_Spectra().halfCut())
    else:
        Sig = SigList_deco[0]
        Spc = np.abs(Sig.to_Spectra().halfCut())
        SigList_deco = SigList_deco[1:]
    # 准备绘图参数
    # 设置总图标题
    title = kwargs.pop("title", "分解结果")
    # 设置xlim
    xlim_waveform_allax = kwargs.pop("xlim_waveform", None)
    xlim_spectrum_allax = kwargs.pop("xlim_spectrum", None)
    # 设置ylim
    if reco:
        ylim_waveform_allax = (
            np.min(Sig) - 0.1 * np.ptp(Sig),
            np.max(Sig) + 0.1 * np.ptp(Sig),
        )  # 设置与原始信号相同的ylim
        ylim_spectrum_allax = (
            np.min(Spc) - 0.1 * np.ptp(Spc),
            np.max(Spc) + 0.1 * np.ptp(Spc),
        )  # 设置频谱y轴范围为110%
    else:
        ylim_waveform_allax = kwargs.pop("ylim_waveform", None)
        ylim_spectrum_allax = kwargs.pop("ylim_spectrum", None)
    # --------------------------------------------------------------------------------#
    # 绘制分解结果
    plot = LinePlot(title=title, ncols=2 if spectrum else 1, **kwargs)
    # 绘制原始信号时域波形与频谱
    plot.waveform(Sig, xlim=xlim_waveform_allax, ylim=ylim_waveform_allax)
    if spectrum:
        plot.spectrum(Spc, xlim=xlim_spectrum_allax, ylim=ylim_spectrum_allax)
    # 绘制各分解成分时域波形与频谱
    for Sig_deco in SigList_deco:
        # 绘制时域波形
        plot.waveform(Sig_deco, xlim=xlim_waveform_allax, ylim=ylim_waveform_allax)
        # 绘制频谱
        if spectrum:
            Spc_deco = np.abs(Sig_deco.to_Spectra().halfCut())
            plot.spectrum(Spc_deco, xlim=xlim_spectrum_allax, ylim=ylim_spectrum_allax)
    fig, axs = plot.show(pattern="return")
    fig.show()
    return fig, axs
