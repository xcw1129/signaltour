"""
# core: Analysis子包核心模块, 提供各种信号分析处理方法的基类接口

---

## 可用的接口

    - class:
        - `BaseAnalysis`: 通用信号分析处理方法类
"""

__all__ = ["BaseAnalysis"]

from .._Assist_Module.Dependencies import (
    Callable,
    Tuple,
)
from .._Signal_Module.core import Signal

# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#


class BaseAnalysis:
    """
    通用信号分析处理方法类

    定义了一般信号处理算法必需初始化方法、常用属性和各种装饰器

    Attributes
    ----------
    Sig : Signal
        待分析信号
    isPlot : bool
        是否绘制分析结果图
    plot_kwargs : dict
        自定义绘图参数
    """

    def __init__(
        self,
        Sig: Signal,
        isLinked: bool = True,
        isPlot: bool = False,
        **kwargs,
    ):
        """
        初始化分析方法

        Parameters
        ----------
        Sig : Signal
            待分析信号
        isLinked : bool, default: True
            是否链接原始待分析信号
        isPlot : bool, default: False
            是否绘制分析结果图
        """
        self.Sig = Sig if isLinked else Sig.copy()
        self.isPlot = isPlot
        self.plot_kwargs = kwargs

    @staticmethod
    def _plot(plot_func: Callable) -> Callable:
        """
        Analysis类专用绘图装饰器, 对方法运行结果进行绘图

        Parameters
        ----------
        plot_func : callable
            执行绘图操作的函数, 需与被装饰方法的返回值格式兼容

        Returns
        -------
        callable
            装饰器函数

        Notes
        -----
        若 `isPlot`=False, 则直接返回被装饰方法的结果

        若 `isPlot`=True, 则调用 `plot_func` 进行绘图, 并不返回结果
        """

        def decorator(func):
            def wrapper(self, *args, **kwargs):
                plot_args = func(self, *args, **kwargs)
                if not self.isPlot:
                    return plot_args  # 不进行绘图, 直接返回结果
                # 需确保被装饰函数返回值格式与plot_func输入参数格式一致
                if isinstance(plot_args, Tuple):
                    plot_func(*plot_args, **self.plot_kwargs)
                else:
                    plot_func(plot_args, **self.plot_kwargs)
                return None  # 绘图则不返回结果

            return wrapper

        return decorator
