"""
# core: Analysis子包核心模块, 实现了信号分析处理方法的基础类与通用函数

---

## 可用的接口

    - class:
        - `BaseAnalysis`: 通用信号分析处理方法基类
"""

__all__ = ["BaseAnalysis"]

from .._Assist_Module.Dependencies import (
    Callable,
    Optional,
    ParamSpec,
    TypeVar,
    wraps,
)
from .._Signal_Module.core import Signal

_P = ParamSpec("_P")
_R = TypeVar("_R")

# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#


class BaseAnalysis:
    """
    通用信号分析处理方法基类

    定义了一般信号处理算法必需初始化方法, 常用属性和各种装饰器

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
            是否链接信号原始数据
        isPlot : bool, default: False
            是否绘制分析结果图
        """
        self.Sig = Sig if isLinked else Sig.copy()
        self.isPlot = isPlot
        self.plot_kwargs = kwargs

    @staticmethod
    def _plot(PlotFunc: Callable) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """
        Analysis类专用绘图装饰器, 接收方法分析处理结果进行绘图

        该装饰器通过Analysis.isPlot属性控制是否执行绘图操作

        Parameters
        ----------
        PlotFunc : callable
            执行绘图操作的函数, 需与被装饰函数的返回值格式兼容

        Returns
        -------
        callable
            装饰器函数
        """

        def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
            @wraps(func)
            def wrapper(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
                plot_args = func(self, *args, **kwargs)
                if not self.isPlot:
                    return plot_args
                # 返回值格式与PlotFunc输入格式需一致
                if isinstance(plot_args, tuple):
                    PlotFunc(*plot_args, **self.plot_kwargs)
                else:
                    PlotFunc(plot_args, **self.plot_kwargs)
                return plot_args

            return wrapper  # ty:ignore[invalid-return-type]

        return decorator
