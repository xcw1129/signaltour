"""
# PlotPlugin

---

## 可用的接口

    - class:
        - `PeakfinderPlugin`: 谱线峰值查找插件，用于查找并标注谱类数据中谱线主瓣对应的坐标
        - `PosNagMaskPlugin`: 谱线正负值掩码插件, 用于对谱类数据中正负值进行不同颜色显示
"""

__all__ = ["PeakfinderPlugin", "PosNagMaskPlugin"]

from .._Assist_Module.Dependencies import Dict, plt
from .core import PlotPlugin

# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#


class PeakfinderPlugin(PlotPlugin):
    """谱线峰值查找插件，用于查找并标注谱类数据中谱线主瓣对应的坐标"""

    def __init__(self, threshold: float = 0.85, distance: float = 0.01):
        """
        谱线峰值查找插件，用于查找并标注谱类数据中谱线主瓣对应的坐标

        Parameters
        ----------
        threshold : float, default: 0.85
            谱线峰值的稀释度阈值, 输入范围: (1/sqrt(d*2+1), 1)
        distance : float, default: 0.01
            峰值最小间距, 若<1则表示数据总长度的比例, 若>1则表示数据点数
        """
        self.distance = distance
        self.threshold = threshold

    def _apply(self, ax: plt.Axes, data: Dict):
        """在指定的子图上查找并标注峰值"""
        # 插件作用于单个ax
        Spc = data.get("Spc")
        if Spc is None:
            return  # 插件仅适用于谱图
        # 寻找峰值
        from .._Analysis_Module.SpectrumAnalysis import find_spectralines

        peak_idx = find_spectralines(Spc.data, distance=self.distance, threshold=self.threshold)
        if peak_idx.size == 0:
            return  # 未找到峰值
        # 标注峰值
        peak_idx = peak_idx[Spc.data[peak_idx] > max(Spc.data) * 1e-3]  # 仅标注显著峰值
        if peak_idx.size > 0:
            peak_idx = peak_idx.astype(int)
            peak_height = Spc.data[peak_idx]
            peak_axis = Spc._axis()[peak_idx]
            ax.plot(peak_axis, peak_height, "o", color="red", markersize=5)
            for x, y in zip(peak_axis, peak_height):
                ax.annotate(
                    f"({x:.2f}, {y:.2f})",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color="red",
                    size=14,
                )


class PosNagMaskPlugin(PlotPlugin):
    """谱线正负值掩码插件, 用于对谱类数据中正负值进行不同颜色显示"""

    def __init__(self, sensitivity: float = 0.05):
        """谱线正负值掩码插件, 用于对谱类数据中正负值进行不同颜色显示"""
        self.sensitivity = sensitivity

    def _apply(self, ax: plt.Axes, data: Dict):
        """在指定的子图上标记正负值"""
        # 插件作用于单个ax
        Spc = data.get("Spc")
        if Spc is None:
            return  # 插件仅适用于谱图
        f, d = Spc.f_axis(), Spc.data.real  # 取实部舍去计算误差
        threshold = self.sensitivity * max(abs(d))
        pos_mask = d > threshold
        neg_mask = d < -threshold
        # 正值点
        ax.scatter(f[pos_mask], d[pos_mask], c="red", s=25, zorder=3, marker="^")
        # 负值点
        ax.scatter(f[neg_mask], d[neg_mask], c="green", s=25, zorder=3, marker="v")
        # 添加水平零值线
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7, zorder=2)
