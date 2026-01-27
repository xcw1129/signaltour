"""
# core

---

## 可用的接口

    - class:
        - `Axis`: 通用坐标轴类, 用于生成和管理一维顺序均匀采样坐标轴数据
        - `Series`: 通用一维序列数据类, 用于保存和处理坐标轴和对应序列数据
        - `t_Axis`: 时间坐标轴类
        - `f_Axis`: 频率坐标轴类
        - `Signal`: 一维时域信号类, 带有时间采样信息
        - `Spectra`: 一维频谱类, 带有频率采样信息
"""

__all__ = ["Axis", "Series", "t_Axis", "f_Axis", "Signal", "Spectra"]

from .._Assist_Module.Dependencies import NDArrayOperatorsMixin, Optional, Self, Tuple, deepcopy, np, re


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class Axis:
    """
    通用坐标轴类, 用于生成和管理一维顺序均匀采样坐标轴数据

    `Axis`及其子类均通过维护核心参数(N, _dx, _x0)来动态生成坐标轴数据, 避免冗余存储和不同采样参数间的冲突

    Attributes
    ----------
    N : int
        坐标轴数据点数
    _dx : float
        坐标轴采样间隔
    _x0 : float
        坐标轴起始点
    name : str
        坐标轴数据名称
    unit : str
        坐标轴数据单位
    data : np.ndarray
        坐标轴数据数组
    lim : tuple
        坐标轴数据范围 (min, max)
    L : float
        坐标轴分布长度
    label : str
        坐标轴标签 "name[unit]"

    Methods
    -------
    copy() -> Self
        返回自身深拷贝
    """

    def __init__(self, N: int, dx: float, x0: float = 0.0, name: str = "", unit: str = ""):
        """
        通用坐标轴类, 用于生成和管理一维顺序均匀采样坐标轴数据

        Parameters
        ----------
        N : int
            坐标轴数据点数
        dx : float
            坐标轴采样间隔
        x0 : float, default: 0.0
            坐标轴起始点
        name : str, optional
            坐标轴数据名称
        unit : str, optional
            坐标轴数据单位
        """
        # Axis类核心维护参数
        self.N: int = N
        self._dx: float = dx
        self._x0: float = x0
        self.name: str = name
        self.unit: str = unit  # 推荐使用标准单位或领域内通用单位

    # --------------------------------------------------------------------------------#
    # 动态可读属性
    @property
    def data(self) -> np.ndarray:
        """返回坐标轴数据数组"""
        return self._x0 + np.arange(self.N) * self._dx  # x=[x0,x0+dx,x0+2dx,...,x0+(N-1)dx]

    @property
    def lim(self) -> tuple:
        """返回坐标轴数据范围 (min, max)"""
        return (self._x0, self._x0 + self._dx * self.N)  # (x0, x0+N*dx)

    @property
    def L(self) -> float:
        """返回坐标轴分布长度"""
        return self.lim[1] - self.lim[0]  # 坐标轴分布长度

    @property
    def label(self) -> str:
        """返回坐标轴标签"""
        return f"{self.name}[{self.unit}]" if self.unit else self.name

    # --------------------------------------------------------------------------------#
    # 数组特性支持
    def __len__(self):
        return self.N

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        # : 计算理论索引
        index = (item - self._x0) / self._dx
        idx_round = round(index)
        # 检查是否接近整数且在范围内
        return abs(index - idx_round) < 1e-9 and 0 <= idx_round < self.N

    def _to_real_index(self, key):
        """将索引键中的物理坐标转换为逻辑索引"""
        if isinstance(key, slice):
            return slice(self._to_real_index(key.start), self._to_real_index(key.stop), key.step)
        # 仅对字符串类型进行物理坐标解析
        if isinstance(key, str):
            pattern = r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)" + re.escape(self.unit) + r"$"
            match = re.fullmatch(pattern, key)
            if not match:
                raise IndexError(f"slice={key}: 物理坐标解析失败")
            val = float(match.group(1))
            # 转换为逻辑索引
            return int(np.ceil((val - self._x0) / self._dx - 1e-9))  # 非整除索引右移, 确保切片在范围内
        # int, None 等其他类型原样返回，由 numpy 处理逻辑索引
        return key

    def __getitem__(self, index):
        # 1. 统一转换物理/逻辑索引为纯逻辑索引
        real_idx = self._to_real_index(index)
        # 2. 处理顺序间隔索引以保持类型
        if isinstance(real_idx, slice):
            start, stop, step = real_idx.indices(self.N)
            if step > 0:
                new_axis = self.copy()
                # 调整核心参数
                new_axis.N = len(range(start, stop, step))
                new_axis._dx = self._dx * step
                new_axis._x0 = self._x0 + start * self._dx
                return new_axis
        # 3. 其它情况直接返回array
        return self.data[real_idx]

    # --------------------------------------------------------------------------------#
    # Python操作兼容
    def __call__(self):
        """返回坐标轴数据"""
        return self.data  # Axis()返回.data属性，方便直接调用

    def __eq__(self, other) -> bool:
        if isinstance(other, Axis):
            return bool(
                self.N == other.N
                and np.isclose(self._dx, other._dx)
                and np.isclose(self._x0, other._x0)
                and self.unit == other.unit
            )
        return False  # 与非Axis类型比较均返回False

    def __str__(self):
        """面向运行时"""
        return f"{type(self).__name__}({self.name}={self.data}[{self.unit}])"

    def __repr__(self):
        """面向开发时"""
        return (
            f"{type(self).__name__}(N={self.N}, dx={self._dx}, x0={self._x0}, name='{self.name}', unit='{self.unit}')"
        )

    # --------------------------------------------------------------------------------#
    # numpy兼容
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if copy is False:
            raise ValueError("copy=False: Axis类不支持返回数据视图")
        return self.data.astype(dtype, copy=False)

    # --------------------------------------------------------------------------------#
    # 外部用户方法
    def copy(self) -> Self:
        """
        返回自身深拷贝

        See Also
        --------
        `numpy.ndarray.copy` : 返回数组的副本
        """
        return deepcopy(self)


# --------------------------------------------------------------------------------------------#
class Series(NDArrayOperatorsMixin):
    """
    通用一维序列数据类, 用于保存和处理坐标轴和对应序列数据

    Series类及其子类实例均支持各种算术和比较符操作, 以及NumPy函数输入, 并在结果符合条件时返回同类实例

    Attributes
    ----------
    axis : Axis
        坐标轴
    data : np.ndarray
        序列数据数组
    name : str
        序列数据名称
    unit : str
        序列数据单位
    label : str
        序列数据标签

    Methods
    -------
    set_label(label: str) -> Self
        修改序列数据标签并返回自身, 便于链式调用
    copy() -> Self
        返回自身深拷贝
    plot(**kwargs) -> Tuple
        绘制序列数据的波形图
    template(data: Optional[np.ndarray] = None) -> Self
        使用自身采样参数生成新实例

    See Also
    --------
    `numpy.lib.mixins.NDArrayOperatorsMixin` : 通过使用`__array_ufunc__`, 定义所有运算符特殊方法的混入类
    """

    def __init__(
        self,
        axis: Axis,
        data: Optional[np.ndarray] = None,
        name: str = "",
        unit: str = "",
        label: str = "",
    ):
        """
        通用一维序列数据类接口, 用于保存和处理坐标轴和对应序列数据

        Parameters
        ----------
        axis : Axis
            坐标轴
        data : ArrayLike, optional
            一维序列数据数组，长度需与axis一致
        name : str, optional
            序列数据名称
        unit : str, optional
            序列数据单位
        label : str, optional
            序列标签
        """
        # Series类核心维护参数
        self._axis: Axis = axis.copy()  # _axis优先级高于_data
        self.name: str = name
        self.unit: str = unit
        self.label: str = label
        # 初始化数据
        if data is not None:
            if self._check_data(data) is False:
                raise ValueError(f"data={data}: 输入序列数据数组非法")
            if not self.OWNDATA:
                # 与源数据共享内存, 但不共享元数据比如shape等
                self._data: np.ndarray = np.asarray(data, copy=False).view()
            else:
                self._data: np.ndarray = np.array(data, copy=True)
        else:
            self._data: np.ndarray = np.zeros(len(axis))

    OWNDATA: bool = False  # noqa: F821

    # --------------------------------------------------------------------------------#
    # 动态可读属性
    @property
    def axis(self) -> Axis:
        """坐标轴"""
        return self._axis  # 支持内容修改

    @axis.setter
    def axis(self, value: Axis):
        self._axis = value  # 支持整体替换

    # 序列数据动态属性, 隔离用户与源数据, 防止误操作导致序列数据异常
    @property
    def data(self) -> np.ndarray:
        """序列数据数组"""
        arr: np.ndarray = self._data.view()  # 返回源数据视图, 避免内存拷贝消耗
        arr.flags.writeable = False  # 防止用户意外修改._data属性
        return arr

    @data.setter
    def data(self, value: np.ndarray):
        # 支持用户整体替换数据, 但输入数据需合法
        if self._check_data(value) is False:
            raise ValueError(f"value={value}: 输入序列数据数组非法")
        if not self.OWNDATA:
            self._data = np.asarray(value, copy=False).view()
        else:
            self._data = np.array(value, copy=True)

    # --------------------------------------------------------------------------------#
    # 数据检查和转换
    def _check_data(self, data):
        arr = np.asarray(data)
        if arr.ndim != 1 or len(arr) != len(self._axis):
            return False
        if np.any(np.isnan(arr)):
            return False
        return True

    # --------------------------------------------------------------------------------#
    # Python操作兼容
    def __str__(self) -> str:
        """面向运行时"""
        return f"{type(self).__name__}[{self.label}]({self.name}={self._data}[{self.unit}], {self._axis})"

    def __repr__(self) -> str:
        """面向开发时"""
        return f"{type(self).__name__}(axis={repr(self._axis)}, data={repr(self._data)}, name='{self.name}', unit='{self.unit}', label='{self.label}')"  # noqa: E501

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> bool:
        if isinstance(other, Series):
            return self._axis == other._axis and np.allclose(self._data, other._data) and self.unit == other.unit
        return False  # 与非Series类型比较均返回False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    # --------------------------------------------------------------------------------#
    # 数组特性支持
    def __getitem__(self, index):
        # 1. 统一转换物理/逻辑索引为纯逻辑索引
        real_idx = self._axis._to_real_index(index)
        # 2. 对坐标轴进行索引/切片
        new_axis = self._axis[index]
        if isinstance(new_axis, Axis):
            # 返回同类实例
            new_Srs = self.template()
            new_Srs._axis = new_axis
            new_Srs._data = self._data[real_idx]  # 与array的切片视图机制一致
            return new_Srs
        else:
            # 其它情况直接返回array
            return self._data[real_idx]

    def __setitem__(self, index, value):
        # 支持用户通过索引部分修改数据, 长度保持不变
        self._data[index] = value

    # --------------------------------------------------------------------------------#
    # numpy兼容

    # 普通接口函数兼容
    def __array_function__(self, func, types, args, kwargs):
        # 将输入中的Series实例转为array以便函数处理
        args = [x._data if isinstance(x, Series) else x for x in args]
        # 执行NumPy的函数操作
        res = func(*args, **kwargs)
        # 检查结果，保持返回类型一致
        if isinstance(res, np.ndarray) and res.shape == self._data.shape and np.issubdtype(res.dtype, np.number):
            new_Srs = self.template(res)
            return new_Srs
        else:
            return res

    # 底层运算函数兼容
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # 支持out参数：将out中的Series实例替换为其_data属性便于in-place修改
        out = kwargs.get("out", None)
        if out is not None:
            new_out = []
            for o in out:
                if isinstance(o, Series):
                    new_out.append(o._data)  # 输入._data原始变量
                else:
                    new_out.append(o)
            kwargs = dict(kwargs)
            kwargs["out"] = tuple(new_out)
        # 将输入中的Series实例转为array以便ufunc处理
        args = [x._data if isinstance(x, Series) else x for x in inputs]
        # ------------------------------------------------------------------------#
        # 根据method调用相应的ufunc方法
        # 处理就地操作（如add.at等，不支持）
        if method == "at":
            return NotImplemented
        # 处理非逐元素操作（如add.reduce等，极少使用）
        if method == "reduce" or method == "reduceat" or method == "outer":
            res = getattr(ufunc, method)(*args, **kwargs)
            return res
        # 处理逐元素运算（如abs、multiply等，常用）
        elif method == "__call__" or method == "accumulate":
            res = getattr(ufunc, method)(*args, **kwargs)
            # 如果指定了out参数, 则直接返回out
            if out is not None:
                return out if len(out) > 1 else out[0]
            # 检查结果，保持返回类型一致
            if isinstance(res, np.ndarray) and res.shape == self._data.shape and np.issubdtype(res.dtype, np.number):
                new_Srs = self.template(res)
                return new_Srs
            else:
                return res
        else:
            return NotImplemented

    # 底层数组接口兼容
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.asarray(self._data, dtype=dtype, copy=copy)

    # --------------------------------------------------------------------------------#
    # 外部用户方法
    def set_label(self, label: str) -> Self:
        """修改序列数据标签并返回自身"""
        self.label = label
        return self

    def copy(self) -> Self:
        """
        返回自身深拷贝

        See Also
        --------
        `numpy.ndarray.copy` : 返回数组的副本
        """
        return deepcopy(self)

    def plot(self, **kwargs) -> Tuple:
        """绘制序列数据的波形图"""
        from .._Plot_Module.LinePlot import PlotFunc_waveform

        fig, ax = PlotFunc_waveform(self, **kwargs)
        return fig, ax

    def template(self, data: Optional[np.ndarray] = None) -> Self:
        """保留自身采样参数等元数据生成新实例, 方便快速创建同类对象"""
        new_Srs = type(self)(
            axis=self._axis,
            data=data,
            name=self.name,
            unit=self.unit,
            label=self.label,
        )
        return new_Srs


# --------------------------------------------------------------------------------------------#
class t_Axis(Axis):
    """
    时间坐标轴类

    Attributes
    ----------
    fs : float
        采样频率
    dt : float
        采样间隔
    t0 : float
        起始时间
    T : float
        采样时长
    data : np.ndarray
        时间坐标轴数据数组
    unit : str
        坐标轴数据单位, 固定为 "s"
    label : str
        坐标轴标签, 固定为 "时间[s]"
    lim : tuple
        时间坐标轴数据范围 (min, max)

    Methods
    -------
    copy() -> Self
        返回自身深拷贝
    """

    def __init__(
        self,
        N: Optional[int] = None,
        fs: Optional[float] = None,
        dt: Optional[float] = None,
        T: Optional[float] = None,
        t0: float = 0.0,
    ):
        """
        时间坐标轴类

        Parameters
        ----------
        fs : float, optional
            采样频率
        dt : float, optional
            采样间隔
        T : float, optional
            采样时长
        t0 : float, default: 0.0
            起始时间
        """
        # 输入参数检查
        if (not [N, fs, dt, T].count(None) == 2) or (fs is not None and dt is not None):
            raise ValueError("采样参数输入错误")
        # 采样参数初始化
        if fs is None:
            if dt is not None:
                fs = 1.0 / dt
            elif T is not None and N is not None:
                fs = N / T
            else:
                raise ValueError("采样参数输入错误")
        if N is None:
            if T is not None:
                N = int(T * fs)
            else:
                raise ValueError("采样参数输入错误")
        super().__init__(N=N, dx=1.0 / fs, x0=t0, unit="s", name="时间")

    # --------------------------------------------------------------------------------#
    # 动态属性
    # Axis类核心参数映射到t_Axis公开动态属性，支持读写
    @property
    def fs(self) -> float:
        """采样频率 (Hz), 修改同步至 dt"""
        return 1.0 / self._dx

    @fs.setter
    def fs(self, value: float):
        if value <= 0:
            raise ValueError("fs 必须大于 0")
        self._dx = 1.0 / float(value)

    @property
    def dt(self) -> float:
        """采样间隔 (s), 修改同步至 fs"""
        return self._dx

    @dt.setter
    def dt(self, value: float):
        if value <= 0:
            raise ValueError("dt 必须大于 0")
        self._dx = float(value)

    @property
    def t0(self) -> float:
        """起始时间 (s)"""
        return self._x0

    @t0.setter
    def t0(self, value: float):
        # t0 可为负数或零，无需严格检查
        self._x0 = float(value)

    @property
    def T(self) -> float:
        """采样时长 (s), 修改同步至 N"""
        return self.N * self.dt

    @T.setter
    def T(self, value: float):
        if value <= 0:
            raise ValueError("T 必须大于 0")
        # 固定 dt，调整 N
        self.N = max(1, int(value / self.dt))


class f_Axis(Axis):
    """
    频率坐标轴类

    Attributes
    ----------
    N : int
        频率采样点数
    df : float
        频率分辨率
    f0 : float
        频率起始点
    F : float
        频率分布宽度
    T : float
        等效时间窗长度
    data : np.ndarray
        频率坐标轴数据数组
    unit : str
        坐标轴数据单位, 固定为 "Hz"
    label : str
        坐标轴标签, 固定为 "频率[Hz]"
    lim : tuple
        频率坐标轴数据范围 (min, max)

    Methods
    -------
    copy() -> Self
        返回自身深拷贝
    """

    def __init__(self, N: int, df: float, f0: float = 0.0):
        """
        频率坐标轴类

        Parameters
        ----------
        N : int
            频率采样点数
        df : float
            频率分辨率
        f0 : float, default: 0.0
            频率起始点
        """
        super().__init__(dx=df, N=N, x0=f0, unit="Hz", name="频率")

    # --------------------------------------------------------------------------------#
    # 动态属性
    # Axis类核心参数映射到f_Axis公开动态属性，支持读写
    @property
    def df(self) -> float:
        """频率分辨率 (Hz)"""
        return self._dx

    @df.setter
    def df(self, value: float):
        if value <= 0:
            raise ValueError("df 必须大于 0")
        self._dx = float(value)

    @property
    def f0(self) -> float:
        """频率起始点 (Hz)"""
        return self._x0

    @f0.setter
    def f0(self, value: float):
        # f0 可为负数或零，无需严格检查
        self._x0 = float(value)

    @property
    def F(self) -> float:
        """频率分布宽度 (Hz), 修改同步至 N"""
        return self.N * self._dx  # 频率分布宽度

    @F.setter
    def F(self, value: float):
        if value <= 0:
            raise ValueError("F 必须大于 0")
        # 固定 df，调整 N
        self.N = max(1, int(value / self._dx))

    @property
    def T(self) -> float:
        """等效时间窗长度 (s), 修改同步至 df"""
        return 1.0 / self._dx

    @T.setter
    def T(self, value: float):
        if value <= 0:
            raise ValueError("T 必须大于 0")
        # 固定 N，调整 df
        self._dx = 1.0 / float(value)


# --------------------------------------------------------------------------------------------#
class Signal(Series):
    """
    一维时域信号类, 带有时间采样信息

    Attributes
    ----------
    t_axis : t_Axis
        时间坐标轴
    data : np.ndarray
        信号数据数组
    name : str
        信号数据名称
    unit : str
        信号数据单位
    label : str
        信号标签

    Methods
    -------
    set_label(label: str) -> Self
        修改信号标签并返回自身, 便于链式调用
    copy() -> Self
        返回自身深拷贝
    plot(**kwargs)
        绘制时域波形
    template(data: Optional[np.ndarray] = None) -> Self
        使用自身采样参数生成新实例
    to_Spectra() -> Spectra
        生成信号的频谱
    """

    def __init__(
        self,
        axis: t_Axis,
        data: Optional[np.ndarray] = None,
        name: str = "",
        unit: str = "",
        label: str = "",
    ):
        """
        初始化Signal对象

        Parameters
        ----------
        axis : t_Axis
            时间坐标轴
        data : np.ndarray, optional
            信号数据数组, 长度需与axis一致
        name : str, default: ""
            信号数据名称
        unit : str, default: ""
            信号数据单位
        label : str, default: ""
            信号标签
        """
        super().__init__(axis=axis, data=data, name=name, unit=unit, label=label)

    # _axis为Axis子类实例.
    # 内部通过_axis直接访问, 简化维护逻辑.
    # 外部通过t_axis/f_axis属性访问, 以支持子类坐标轴特性.
    @property
    def t_axis(self) -> t_Axis:
        """时间坐标轴"""
        return self._axis

    @t_axis.setter
    def t_axis(self, value: t_Axis):
        self._axis: t_Axis = value

    @property
    def f_axis(self) -> f_Axis:
        """频率坐标轴"""
        return f_Axis(df=1 / self.t_axis.T, N=self.t_axis.N)

    # --------------------------------------------------------------------------------#
    # 自带方法
    def plot(self, **kwargs) -> Tuple:
        """绘制信号的时域波形图"""
        from .._Plot_Module.LinePlot import PlotFunc_waveform

        fig, ax = PlotFunc_waveform(self, **kwargs)
        return fig, ax

    def to_Spectra(self, density: bool = False) -> "Spectra":
        """生成信号的频谱"""
        from .._Analysis_Module.SpectrumAnalysis import SpectrumAnalysis

        if density:
            Spc = SpectrumAnalysis(self).ft()
        else:
            Spc = SpectrumAnalysis(self).cft(padTimes=0)  # 保持原始长度, 不延拓
        return Spc


class Spectra(Series):
    """
    一维频谱类, 带有频率采样信息

    Attributes
    ----------
    f_axis : f_Axis
        频率坐标轴
    data : np.ndarray
        频谱数据数组
    name : str
        频谱数据名称
    unit : str
        频谱数据单位
    label : str
        频谱标签

    Methods
    -------
    set_label(label: str) -> Self
        修改频谱标签并返回自身, 便于链式调用
    copy() -> Self
        返回自身深拷贝
    plot(**kwargs)
        绘制频谱曲线
    template(data: Optional[np.ndarray] = None) -> Self
        使用自身采样参数生成新实例
    halfCut() -> Spectra
        返回单边谱
    """

    def __init__(
        self,
        axis: f_Axis,
        data: Optional[np.ndarray] = None,
        name: str = "",
        unit: str = "",
        label: str = "",
    ):
        """
        一维频谱类, 带有频率采样信息

        Parameters
        ----------
        axis : f_Axis
            频率坐标轴
        data : np.ndarray, optional
            频谱数据数组，长度需与axis一致
        name : str, default: ""
            频谱数据名称
        unit : str, default: ""
            频谱数据单位
        label : str, default: ""
            频谱标签
        """
        if data is None:
            data = np.zeros(len(axis), dtype=np.complex128)
        super().__init__(axis=axis, data=data, name=name, unit=unit, label=label)

    @property
    def f_axis(self) -> f_Axis:
        """频率坐标轴"""
        return self._axis

    @f_axis.setter
    def f_axis(self, value: f_Axis):
        self._axis: f_Axis = value

    # --------------------------------------------------------------------------------#
    # 自带方法
    def plot(self, **kwargs) -> Tuple:
        """绘制频谱"""
        from .._Plot_Module.LinePlot import PlotFunc_spectrum

        fig, ax = PlotFunc_spectrum(self, **kwargs)
        return fig, ax

    def halfCut(self) -> "Spectra":
        """裁剪为余弦形式单边谱"""
        if self.f_axis.f0 != 0.0:
            raise TypeError(f"f0={self.f_axis.f0}: 当前谱频率轴不完整, 无法进行单边谱截取")
        N = len(self)
        if N % 2 == 0:  # 偶数点，非对称
            half_N = N // 2
        else:  # 奇数点，对称
            half_N = (N + 1) // 2

        self._axis, self._data = (
            self._axis[:half_N],
            self._data[:half_N],
        )  # 原地修改当前对象
        self._data[1:] *= 2  # 除直流分量外幅值翻倍
        return self
