"""
# core

---

## 可用的接口

    - class:
        - `BasePlot`: 通用绘图类, 实现多绘图任务流程框架, 供子类继承并实现具体绘图逻辑
        - `PlotPlugin`: 绘图插件类，提供扩展绘图功能的接口
"""

__all__ = ["BasePlot", "PlotPlugin"]

from .._Assist_Module.Dependencies import (
    List,
    Optional,
    Self,
    cycler,
    deque,
    font_manager,
    np,
    os,
    plt,
    shallowcopy,
    ticker,
)


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class PlotPlugin:
    """
    绘图插件类，提供扩展绘图功能的接口

    Methods
    -------
    _apply(ax: plt.Axes, data: any) -> None
        将插件应用于指定分图
    """

    def _apply(self, ax: plt.Axes, Data):
        """
        将插件应用于指定分图。

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            插件将作用于的子图坐标轴对象。
        data : any
            与该子图关联的数据。

        Raises
        ------
        NotImplementedError
            子类必须实现 _apply 方法。
        """
        raise NotImplementedError("子类必须实现_apply方法")


# --------------------------------------------------------------------------------------------#
class BasePlot:
    """
    通用绘图类, 实现多绘图任务流程框架, 供子类继承并实现具体绘图逻辑

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        当前绘图流程操作的 Figure 对象
    axes : numpy.ndarray
        当前绘图流程操作的 Axes 对象数组
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
        全局绘图参数，一般初始化后不再修改
    tasks : collections.deque
        绘图任务队列，存储所有待绘制的任务
    last_task : dict
        最新添加的绘图任务

    Methods
    -------
    init_Plot_rcParams(scheme: str = "default") -> None
        设置绘图风格配置方案，并保存原始配置用于恢复
    restore_User_rcParams() -> None
        恢复原始绘图风格配置方案
    set_params_to_task(**kwargs) -> "Plot"
        为最新添加的绘图任务设置专属参数
    add_plugin_to_task(plugin: PlotPlugin) -> "Plot"
        为最新添加的绘图任务添加一个插件
    show(pattern: str = "plot", filename="Plot.png", save_format="png") -> tuple
        执行所有已注册的绘图任务并显示/返回/保存最终图形
    canvas() -> tuple
        生成当前绘图对象的空白画布
    """

    def __init__(
        self,
        title: str = "",
        scheme: str = "default",
        autoRestore: bool = True,
        ncols: int = 1,
        figsize: Optional[tuple] = None,
        **kwargs,
    ):
        """
        通用绘图类, 实现多绘图任务流程框架, 供子类继承并实现具体绘图逻辑

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
        figsize : tuple, optional
            所有子图共享的图形大小
        """
        self.figure: plt.Figure = None  # 当前绘图流程操作的 Figure 对象
        self.axs: List[plt.Axes] = None  # 当前绘图流程操作的 Axes 对象数组
        self.title = title  # 总图标题
        self.scheme = scheme  # 绘图风格配置方案
        self.autoRestore = autoRestore  # 是否自动恢复用户原始rcParams配置
        self.ncols = ncols  # 多图绘制时的子图列数
        self.figsize = figsize  # 所有子图共享的图形大小
        self.kwargs = kwargs  # 全局绘图参数, 一般初始化后不再修改
        self.tasks = deque()  # 绘图任务队列, 存储所有待绘制的任务

    @property
    def last_task(self) -> dict:
        """最新添加的绘图任务"""
        if not self.tasks:
            raise RuntimeError("请先添加一个绘图任务, 再访问其参数")
        return self.tasks[-1]

    # --------------------------------------------------------------------------------#
    # matplotlib 基础配置管理方法

    IS_ALREADY_CONFIGURED = False  # 标记是否已初始化绘图配置

    SAVED_USER_RCPARAMS = None  # 类变量，保存用户的原始 matplotlib 配置

    RCPARAMS_SCHEME_DEFAULT = {  # 绘图配置方案-默认
        "axes.grid": True,  # 显示网格
        "axes.labelsize": 18,  # 轴标签字体大小
        "axes.unicode_minus": False,  # 负号正常显示
        "axes.titlesize": 20,  # 标题字体大小
        "figure.dpi": 100,  # 显示分辨率
        "figure.figsize": (9, 4),  # 默认图形大小
        "font.family": ["Times New Roman + SimSun", "serif"],  # 支持中英文
        "font.serif": ["SimSun"],  # 备用字体
        "font.size": 18,  # 设置全局字体大小
        "grid.linestyle": "--",  # 网格线为虚线
        "legend.fontsize": 16,  # 图例字体大小
        "mathtext.fontset": "custom",  # 公式字体设置
        "mathtext.rm": "Times New Roman",  # 数学公式字体 - 正常
        "mathtext.it": "Times New Roman:italic",  # 数学公式字体 - 斜体
        "mathtext.bf": "Times New Roman:bold",  # 数学公式字体 - 粗体
        "savefig.dpi": 600,  # 保存分辨率
        "xtick.labelsize": 16,  # x轴刻度标签字体大小
        "xtick.direction": "in",  # x轴刻度线朝内
        "ytick.labelsize": 16,  # y轴刻度标签字体大小
        "ytick.direction": "in",  # y轴刻度线朝内
    }

    RCPARAMS_SCHEME_LinePlot1 = {  # 绘图配置方案-线条图风格1
        "axes.grid.axis": "y",  # 只显示y轴网格
        "axes.prop_cycle": cycler(
            color=[
                "#1f77b4",  # 蓝
                "#ff7f0e",  # 橙
                "#2ca02c",  # 绿
                "#d62728",  # 红
                "#a77ece",  # 紫
                "#8c564b",  # 棕
                "#520e8e",  # 粉
                "#7f7f7f",  # 灰
                "#bcbd22",  # 橄榄
                "#17becf",  # 青
            ]
        ),  # 设置颜色循环
        "grid.linestyle": (0, (8, 6)),  # 网格线为实虚比8:6的点划线
    }

    RCPARAMS_SCHEME_LinePlot2 = {  # 绘图配置方案-线条图风格2
        "axes.grid": False,  # 不显示网格
        "axes.prop_cycle": cycler(
            color=[
                "#1f77b4",  # 蓝
                "#ff7f0e",  # 橙
                "#2ca02c",  # 绿
                "#d62728",  # 红
                "#a77ece",  # 紫
                "#8c564b",  # 棕
                "#520e8e",  # 粉
                "#7f7f7f",  # 灰
                "#bcbd22",  # 橄榄
                "#17becf",  # 青
            ]
        ),  # 设置颜色循环
    }

    RCPARAMS_SCHEME_LinePlot3 = {  # 绘图配置方案-线条图风格3
        "axes.grid.axis": "both",  # 同时显示x, y轴网格
        "axes.prop_cycle": cycler(
            color=[
                "#e41a1c",  # 鲜红
                "#1227e6",  # 鲜蓝
                "#23cf10",  # 鲜绿
                "#5d0cd0",  # 橙色
                "#df6800",  # 紫色
            ]
        ),  # 设置颜色循环
        "figure.figsize": (7, 5),  # 默认图形大小
        "grid.alpha": 1,  # 网格线透明度
        "grid.color": "#000000",  # 网格线颜色为黑色
        "grid.linestyle": (0, (3, 3)),  # 网格线为较密集的点
        "lines.linewidth": 1.5,  # 线条宽度
        "lines.markersize": 7,  # 标记点大小
    }

    RCPARAMS_SCHEME_ImagePlot1 = {  # 绘图配置方案-热力图风格1
        "axes.grid": False,  # 不显示网格
        "figure.figsize": (8, 6),  # 默认图形大小
    }

    @staticmethod
    def init_Plot_rcParams(scheme: str = "default") -> None:
        """
        设置绘图风格配置方案，并保存原始配置用于恢复

        Parameters
        ----------
        scheme : str, default: "default"
            配置方案名称，可选:
            - "default": 默认配置方案
            - "LinePlot1": 线条图风格1
            - "LinePlot2": 线条图风格2
            - "LinePlot3": 线条图风格3
            - "ImagePlot1": 热力图风格1
        """
        if BasePlot.IS_ALREADY_CONFIGURED:
            return  # 已初始化过，无需重复操作
        BasePlot.IS_ALREADY_CONFIGURED = True
        BasePlot.SAVED_USER_RCPARAMS = shallowcopy(plt.rcParams)  # 保存外部原始配置
        # 尝试加载自定义字体
        try:
            font_path = os.path.join(os.path.dirname(__file__), "..", "_Assist_Module", "times+simsun.ttf")
            font_path = os.path.abspath(font_path)
            font_manager.fontManager.addfont(font_path)
        except Exception:
            print("Times New Roman + SimSun: 自定义字体加载失败，使用系统默认字体。")
        # 选择库自定义配置方案
        rcParams = BasePlot.RCPARAMS_SCHEME_DEFAULT.copy()
        if scheme == "LinePlot1":
            rcParams.update(BasePlot.RCPARAMS_SCHEME_LinePlot1)
        elif scheme == "LinePlot2":
            rcParams.update(BasePlot.RCPARAMS_SCHEME_LinePlot2)
        elif scheme == "LinePlot3":
            rcParams.update(BasePlot.RCPARAMS_SCHEME_LinePlot3)
        elif scheme == "ImagePlot1":
            rcParams.update(BasePlot.RCPARAMS_SCHEME_ImagePlot1)
        elif scheme != "default":
            raise ValueError(f"scheme={scheme}:未知的配置方案")
        # 更新 matplotlib 全局配置
        plt.rcParams.update(rcParams)

    @staticmethod
    def restore_User_rcParams() -> None:
        """恢复原始绘图风格配置方案"""
        if BasePlot.IS_ALREADY_CONFIGURED and BasePlot.SAVED_USER_RCPARAMS is not None:
            plt.rcParams.update(BasePlot.SAVED_USER_RCPARAMS)
            BasePlot.SAVED_USER_RCPARAMS = None
            BasePlot.IS_ALREADY_CONFIGURED = False

    # --------------------------------------------------------------------------------#
    # 内部绘图基本框架方法
    def _setup_figure(self, num_tasks):
        """根据任务数量设置图形和子图"""
        ncols = self.ncols
        nrows = (num_tasks + ncols - 1) // ncols
        # 根据子图数量调整图形大小
        if self.figsize is not None:
            figsize = (self.figsize[0] * ncols, self.figsize[1] * nrows)
        else:
            cur_figsize = plt.rcParams.get("figure.figsize")
            figsize = (cur_figsize[0] * ncols, cur_figsize[1] * nrows)
        # 创建图形和子图
        self.figure, self.axs = plt.subplots(nrows, ncols, figsize=figsize)
        # 统一将 axes 转为 1 维数组，方便迭代
        if isinstance(self.axs, (list, tuple)):
            self.axs = np.array(self.axs).flatten().tolist()
        else:
            self.axs = [self.axs]
        # 设置总图标题
        self.figure.suptitle(self.title)

    # --------------------------------------------------------------------------------#
    # 子图级图形元素设置方法
    def _setup_title(self, ax, task_kwargs):
        """设置标题"""
        title = task_kwargs.get("title")
        if title:
            ax.set_title(title)

    def _setup_x_axis(self, ax, task_kwargs):
        """设置X轴"""
        # 设置X轴标签
        xlabel = task_kwargs.get("xlabel")
        ax.set_xlabel(xlabel)
        # 检测用户是否指定为字符串刻度
        cur_xticklabels = ax.get_xticklabels()
        xticks_IS_STR = False
        labeltexts = [label.get_text() for label in cur_xticklabels]
        if any(text != "" and not text.replace(".", "", 1).replace("-", "", 1).isdigit() for text in labeltexts):
            xticks_IS_STR = True
        # 设置X轴范围
        ax.margins(x=0)  # 设置X轴出血边边距为0
        cur_xlim = ax.get_xlim()
        xlim = task_kwargs.get("xlim", cur_xlim)
        if xticks_IS_STR:
            ax.set_xlim(-0.5, len(labeltexts) - 0.5)  # 字符串刻度左右各留0.5个单位的边距
        else:
            ax.set_xlim(xlim[0], xlim[1])  # 数值刻度则占满整个x轴显示范围
        # 仅对数值刻度进行后续设置
        if not xticks_IS_STR:
            # 设置X轴刻度
            xticks = task_kwargs.get("xticks")
            if xticks is not None:
                ax.set_xticks(xticks)
            else:
                # 设置11个均匀分布的刻度
                xticks = np.linspace(xlim[0], xlim[1], 11, endpoint=True)
                ax.set_xticks(xticks)
            # 设置X轴刻度标签格式
            sf = ticker.ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-2, 3))  # 显示x位有效数字, 则为(-x+1, x)
            if xticks[0] != 0 and (xticks[1] - xticks[0]) / abs(xticks[0]) <= 1e-3:  # 1e-x
                sf.set_useOffset(xticks[0])  # 启用偏移量显示, 避免大数值+小范围导致显示位数过多
            ax.xaxis.set_major_formatter(sf)  # mpl会自动调节显示位数防止刻度标签重叠

    def _setup_y_axis(self, ax, task_kwargs):
        """设置Y轴"""
        # 设置Y轴标签
        ylabel = task_kwargs.get("ylabel")
        ax.set_ylabel(ylabel)
        # 设置Y轴刻度格式
        yscale = task_kwargs.get("yscale", "linear")
        ax.set_yscale(yscale)
        # 设置Y轴范围
        ax.margins(y=0)  # 设置为0, 以获取数据范围
        cur_ylim = ax.get_ylim()
        if yscale == "log":
            cur_ylim = (max(cur_ylim[0], 1e-8), max(cur_ylim[1], 1e-8))
        true_ylim = task_kwargs.get("ylim", cur_ylim)
        if "ylim" in task_kwargs:
            # 如果用户指定了 ylim，则直接使用，不添加出血边
            ylim = true_ylim
        else:
            ymargin = task_kwargs.get("ymargin", 0.1)
            if yscale == "log":
                # 对数坐标下，按对数比例设置出血边
                log_min = np.log10(true_ylim[0])
                log_max = np.log10(true_ylim[1])
                log_range = log_max - log_min
                ylim = (
                    10 ** (log_min - log_range * ymargin),
                    10 ** (log_max + log_range * ymargin),
                )
            else:
                ylim = (
                    true_ylim[0] - (true_ylim[1] - true_ylim[0]) * ymargin,
                    true_ylim[1] + (true_ylim[1] - true_ylim[0]) * ymargin,
                )  # 人工设置y轴出血边
        ax.set_ylim(ylim[0], ylim[1])
        # 设置Y轴刻度
        yticks = task_kwargs.get("yticks")
        if yscale == "log":  # 对数刻度下强制自动刻度
            # 主刻度：每10倍一个主刻度
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
            # 次刻度：每10倍区间内插9个小刻度（2~9倍）
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        else:
            if yticks is not None:
                ax.set_yticks(yticks)
            else:
                ybins = task_kwargs.get("ybins", 5)
                # 设置指定数量的均匀分布刻度
                yticks = np.linspace(
                    true_ylim[0],
                    true_ylim[1],
                    ybins,
                    endpoint=True,
                )
                ax.set_yticks(yticks)
            # 设置Y轴刻度标签格式
            sf = ticker.ScalarFormatter(useMathText=True)
            sf.set_powerlimits((-3, 4))
            if yticks[0] != 0 and (yticks[1] - yticks[0]) / abs(yticks[0]) <= 1e-4:
                sf.set_useOffset(yticks[0])
            ax.yaxis.set_major_formatter(sf)

    def _save_figure(self, filename, save_format):
        """保存图形"""
        if self.figure is not None:
            if save_format != filename.split(".")[-1]:
                filename = f"{filename.split('.')[0]}.{save_format}"
            self.figure.savefig(filename)
        else:
            raise ValueError("图形未创建，无法保存")

    # --------------------------------------------------------------------------------#
    # 绘图个性化修改外部接口方法
    # 默认修改的绘图任务为最新添加的任务, 保持调用时的可读性
    def set_params_to_task(self, **kwargs) -> "BasePlot":
        """为最新添加的绘图任务设置额外绘图参数"""
        self.last_task["kwargs"].update(kwargs)
        return self

    def add_plugin_to_task(self, plugin: PlotPlugin) -> "BasePlot":
        """为最新添加的绘图任务设置额外插件"""
        if not isinstance(plugin, PlotPlugin):
            raise TypeError("插件必须是 PlotPlugin 的实例。")
        self.last_task["plugins"].append(plugin)
        return self

    # --------------------------------------------------------------------------------#
    # 子类绘图任务注册接口函数实现示例
    def plot(self, Data, **kwargs) -> Self:
        """注册一个平面轨迹图的绘制任务"""

        # ------------------------------------------------------------------------#
        # 绘图函数
        def _draw_plot(ax, data, kwargs):
            Data = data["Data"]  # noqa: F841
            pass

        # ------------------------------------------------------------------------#
        # 绘图个性化设置

        # ------------------------------------------------------------------------#
        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {}
        task_kwargs.update(self.kwargs)
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------#
        # 注册绘图任务
        task = {
            "data": {"Data": Data},
            "kwargs": task_kwargs,
            "function": _draw_plot,
            "plugins": [],
        }
        self.tasks.append(task)
        return self

    # --------------------------------------------------------------------------------#
    # 执行绘图任务总控方法
    def show(self, pattern: str = "plot", filename="Plot.png", save_format="png") -> tuple:
        """
        执行所有已注册的绘图任务并显示/返回/保存最终图形

        Parameters
        ----------
        pattern : str, default: "plot"
            执行模式, 可选:
            - "plot": 显示图形窗口
            - "return": 返回 (figure, axes) 元组
            - "save": 保存图形到文件
        filename : str, default: "Plot.png"
            保存图形的文件名, 仅在 pattern="save" 时有效
        save_format : str, default: "png"
            保存图形的格式, 仅在 pattern="save" 时有效

        Returns
        -------
        tuple or None
            当 pattern="return"返回 (figure, axes) 元组, 其它返回 None

            (figure, axes) 元组, 分别为Figure 和 Axes 对象.
            其中 Axes 为 1 维 numpy 数组，便于索引和迭代
        """
        # 检查是否有绘图任务
        num_tasks = len(self.tasks)
        if num_tasks == 0:
            return ()
        # 设置绘图风格配置方案
        self.init_Plot_rcParams(self.scheme)
        # ------------------------------------------------------------------------#
        # 以分图形式执行所有绘图任务
        self._setup_figure(num_tasks)  # 创建图形和子图
        # 依次在对应子图上执行每个绘图任务
        for i, task_ax in enumerate(self.axs):
            # 隐藏多余子图
            if i >= num_tasks:
                task_ax.set_visible(False)
                continue
            # 获取当前任务的信息
            # 按 FIFO 原则从队列左端弹出任务
            task = self.tasks.popleft()
            task_data = task["data"]
            task_kwargs = task["kwargs"]
            task_function = task["function"]
            task_plugins = task["plugins"]
            # ----------------------------------------------------------------#
            # 执行当前绘图任务
            try:
                task_function(task_ax, task_data, task_kwargs)  # 先执行数据相关绘图任务，便于后续图形元素设置
                self._setup_title(task_ax, task_kwargs)
                self._setup_x_axis(task_ax, task_kwargs)
                self._setup_y_axis(task_ax, task_kwargs)
                for plugin in task_plugins:
                    plugin._apply(task_ax, task_data)
            except Exception as e:
                print(f"绘制第{i + 1}个子图失败: {e}")
        # ------------------------------------------------------------------------#
        # 恢复用户原始 matplotlib 配置
        if self.autoRestore:
            self.restore_User_rcParams()
        # 总图调整设置
        self.figure.tight_layout()
        if pattern == "plot":
            self.figure.show()
        elif pattern == "return":
            return self.figure, self.axs
        elif pattern == "save":
            self._save_figure(filename, save_format)
            plt.close(self.figure)
        else:
            raise ValueError(f"未知的模式: {pattern}")
        return ()

    # --------------------------------------------------------------------------------#
    # 其它绘图辅助方法
    def canvas(self) -> tuple:
        """返回当前绘图流程的空白画布"""
        # 转移当前绘图任务
        existing_tasks = shallowcopy(self.tasks)
        self.tasks.clear()
        # 生成空白画布
        self.plot(0)
        fig, ax = self.show(pattern="return")
        # 恢复之前的绘图任务
        self.tasks = existing_tasks
        return fig, ax
