"""
# ImagePlot

---

## 可用的接口

    - function:
        - `spectrogram_PlotFunc`: 信号时频谱图绘制函数
    - class:
        - `ImagePlot`: 时频谱图、热力图等二维图绘图类
"""

__all__ = ["ImagePlot", "spectrogram_PlotFunc"]

from .._Assist_Module.Dependencies import Optional, np
from .core import BasePlot


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class ImagePlot(BasePlot):
    """
    时频谱图、热力图等二维图绘图类

    Methods
    -------
    spectrogram(time: np.ndarray, freq: np.ndarray, matrix: np.ndarray, continuous: bool = True, **kwargs) -> ImagePlot
        注册一个时频谱图的绘制任务
    """

    def __init__(
        self,
        scheme: str = "ImagePlot1",
        autoRestore: bool = True,
        ncols: int = 1,
        figsize: Optional[tuple] = None,
        **kwargs,
    ):
        """
        时频谱图、热力图等二维图绘图类

        Parameters
        ----------
        scheme : str, default: "default"
            绘图风格配置方案
        autoRestore : bool, default: True
            是否自动恢复用户原始rcParams配置
        ncols : int, default: 1
            多图绘制时的子图列数
        figsize : tuple, optional
            所有子图共享的图形大小
        """
        super().__init__(
            scheme=scheme,
            autoRestore=autoRestore,
            ncols=ncols,
            figsize=figsize,
            **kwargs,
        )

    def spectrogram(
        self,
        time: np.ndarray,
        freq: np.ndarray,
        matrix: np.ndarray,
        continuous: bool = True,
        **kwargs,
    ) -> "ImagePlot":
        """
        注册一个时频谱图的绘制任务

        Parameters
        ----------
        time : np.ndarray
            时间轴数据
        freq : np.ndarray
            频率轴数据
        matrix : np.ndarray
            时频矩阵数据
        continuous : bool, default: True
            是否采用连续插值显示

        Returns
        -------
        ImagePlot
            返回绘图对象本身，以支持链式调用
        """

        # ------------------------------------------------------------------------------------#
        # 时频谱图绘制函数: 通过任务队列传递到绘图引擎
        def _draw_spectrogram_imshow(ax, data, kwargs):
            Matrix = data.get("Matrix")
            kwargs_imshow = kwargs.get("imshow", {})
            ax.imshow(
                Matrix.T,  # 时间行转置为列, 符合时频图习惯
                **kwargs_imshow,
            )

        def _draw_spectrogram_pcolormesh(ax, data, kwargs):
            Matrix = data.get("Matrix")
            axis1, axis2 = data.get("axis1"), data.get("axis2")
            kwargs_pcolormesh = kwargs.get("pcolormesh", {})
            T, F = np.meshgrid(axis1, axis2)
            ax.pcolormesh(
                T,
                F,
                Matrix.T,  # 时间行转置为列, 符合时频图习惯
                **kwargs_pcolormesh,
            )

        # ------------------------------------------------------------------------------------#
        # 时频谱图绘制个性化设置

        # 绘图任务kwargs优先级: 用户传入kwargs > 全局kwargs > 方法默认设置
        task_kwargs = {
            "title": "时频图",
            "xlabel": "时间[s]",
            "ylabel": "频率[Hz]",
            "ymargin": 0,
        }
        task_kwargs.update(self.kwargs)
        task_kwargs.update(kwargs)
        # ------------------------------------------------------------------------------------#
        # 注册绘图任务
        # 根据时间轴和频率轴是否均匀分布选择不同绘图函数
        if np.allclose(np.diff(freq), freq[1] - freq[0]) and np.allclose(np.diff(time), time[1] - time[0]):
            # 均匀网格数据, 可用imshow绘制
            task_function = _draw_spectrogram_imshow
            task_data = {"Matrix": matrix}
            task_kwargs["imshow"] = {
                "cmap": ("viridis" if np.all(matrix >= 0) else "seismic"),  # 根据数据分布选择合适colormap
                "aspect": "auto",
                "interpolation": "bilinear" if continuous else "none",
                "extent": [  # 默认坐标轴为线性均匀分布
                    time[0],
                    time[-1],
                    freq[0],
                    freq[-1],
                ],
                "origin": "lower",
            }
        else:
            # 非均匀网格数据, 需用pcolormesh绘制
            task_function = _draw_spectrogram_pcolormesh
            task_data = {"Matrix": matrix, "axis1": time, "axis2": freq}
            task_kwargs["pcolormesh"] = {
                "cmap": ("viridis" if np.all(matrix >= 0) else "seismic"),  # 根据数据分布选择合适colormap
                "shading": "auto",
            }
        task = {
            "data": task_data,
            "kwargs": task_kwargs,
            "function": task_function,
            "plugins": [],
        }
        self.tasks.append(task)
        return self


def spectrogram_PlotFunc(
    times: np.ndarray,
    freqs: np.ndarray,
    matrix: np.ndarray,
    **kwargs,
) -> tuple:
    """
    信号时频谱图绘制函数

    Parameters
    ----------
    times : np.ndarray
        时间轴数据
    freqs : np.ndarray
        频率轴数据
    matrix : np.ndarray
        时频矩阵数据

    Returns
    -------
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    """
    continuous = kwargs.pop("continuous", True)
    fig, ax = ImagePlot().spectrogram(times, freqs, matrix, continuous=continuous, **kwargs).show(pattern="return")
    fig.show()
    return fig, ax
