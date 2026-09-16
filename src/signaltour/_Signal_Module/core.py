"""
# core: Signal子包核心模块, 实现了坐标轴、序列与信号数据封装的基础类与通用方法

    - class:
        - `Axis`: 通用坐标轴类, 用于生成和管理一维顺序均匀采样坐标轴数据
        - `Series`: 通用序列数据类, 用于保存和管理以一维序列数据及其坐标轴
        - `t_Axis`: 时间坐标轴类
        - `f_Axis`: 频率坐标轴类
        - `Signal`: 一维时域信号类, 实现采样信息与数据的绑定, 支持混合运算并内置常用信号数据交互方法
        - `Spectra`: 一维频谱数据类, 实现采样信息与数据的绑定, 支持混合运算并内置常用频谱数据交互方法
"""

__all__ = ["Axis", "Series", "t_Axis", "f_Axis", "Signal", "Spectra"]

from .._Assist_Module.Dependencies import (
    NDArrayOperatorsMixin,
    Optional,
    Self,
    Tuple,
    deepcopy,
    np,
    numbers,
    re,
)


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class Axis:
    """
    通用坐标轴类, 用于生成和管理一维顺序均匀采样坐标轴数据

    Axis及其子类均通过维护核心参数(N, _dx, _x0)来动态生成坐标轴数据, 避免冗余存储和不同采样参数间的冲突

    Attributes
    ----------
    N : int
        坐标轴点数
    _dx : float
        坐标轴采样间隔
    _x0 : float
        坐标轴起始点
    name : str
        坐标轴名称
    unit : str
        坐标轴单位
    data : np.ndarray
        坐标轴数组
    lim : tuple
        坐标轴范围: (min, max)
    L : float
        坐标轴分布长度

    Methods
    -------
    - copy()
        返回拷贝对象, 与原对象完全独立

    - to_pos_index(key)
        将物理索引转换为位置索引
    """

    def __init__(self, N: int, dx: float, x0: float = 0.0, name: str = "", unit: str = "") -> None:
        """
        通用坐标轴类, 用于生成和管理一维顺序均匀采样坐标轴数据

        Parameters
        ----------
        N : int
            坐标轴点数
        dx : float
            坐标轴采样间隔
        x0 : float, default: 0.0
            坐标轴起始点
        name : str, optional
            坐标轴名称
        unit : str, optional
            坐标轴单位, 推荐使用标准单位或领域内通用单位. 支持$符号包裹的LaTeX语法, 以便绘图显示
        """
        # Axis类核心维护参数
        if (not isinstance(N, numbers.Integral)) or N <= 0:
            raise ValueError(f"N={N}: 坐标轴点数必须为正整数")
        if (not isinstance(dx, numbers.Real)) or dx <= 0:
            raise ValueError(f"dx={dx}: 坐标轴采样间隔必须为正数")
        self.N: int = N
        self._dx: float = dx
        self._x0: float = x0
        self.name: str = name
        self.unit: str = unit

    # --------------------------------------------------------------------------------#
    # 动态可读属性
    @property
    def data(self) -> np.ndarray:
        """坐标轴数组"""
        return self._x0 + np.arange(self.N) * self._dx  # x=[x0,x0+dx,x0+2dx,...,x0+(N-1)dx]

    @property
    def lim(self) -> tuple[float, float]:
        """坐标轴范围: (min, max)"""
        return (self._x0, self._x0 + self._dx * self.N)  # (x0, x0+N*dx)

    @property
    def L(self) -> float:
        """坐标轴分布长度"""
        return self.N * self._dx  # N*dx

    # --------------------------------------------------------------------------------#
    # 数组特性支持
    def __len__(self):
        return self.N

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item: float) -> bool:
        # 利用坐标轴顺序均匀采样特性, 判断坐标是否在坐标轴上, 避免遍历数组
        idx = (item - self._x0) / self._dx
        idx_round = round(idx)  # 取整数索引
        # 检查是否接近整数且在范围内
        is_contained = abs(idx - idx_round) < 1e-5 and 0 <= idx_round < self.N
        return is_contained

    def __getitem__(self, index):
        pos_idx = self.to_pos_index(index)
        if isinstance(pos_idx, float):
            idx_round = round(pos_idx)
            if abs(pos_idx - idx_round) < 1e-5:
                return self.data[idx_round]
            else:
                raise IndexError(f"index={index}: 不在坐标轴上, 无法索引到对应位置")
        # 处理顺序间隔索引以保持类型
        if isinstance(pos_idx, slice):
            start, stop, step = pos_idx.indices(self.N)
            if step > 0:
                new_axis = self.copy()
                # 调整核心参数
                new_axis.N = len(range(start, stop, step))
                new_axis._dx = self._dx * step
                new_axis._x0 = self._x0 + start * self._dx
                return new_axis
        # 其它情况直接返回array
        return self.data[pos_idx]

    # --------------------------------------------------------------------------------#
    # Python操作兼容
    def __call__(self):
        # 返回坐标轴数组
        return self.data  # Axis()返回.data属性，方便直接调用

    def __eq__(self, other) -> bool:
        if isinstance(other, Axis):
            return bool(
                self.N == other.N
                and np.isclose(self._dx, other._dx)
                and np.isclose(self._x0, other._x0)
                and self.unit == other.unit
            )
        if isinstance(other, np.ndarray):
            return np.allclose(self.data, other)
        return False

    def __str__(self):
        # 面向运行时
        return f"{type(self).__name__}({self.name}={self.data}{self.unit})"

    def __repr__(self):
        # 面向开发时
        return (
            f"{type(self).__name__}(N={self.N}, dx={self._dx}, x0={self._x0}, name='{self.name}', unit='{self.unit}')"  # noqa: E501
        )

    # --------------------------------------------------------------------------------#
    # numpy兼容
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return self.data.astype(dtype, copy=False)

    # --------------------------------------------------------------------------------#
    # 外部用户方法
    def copy(self):
        """返回拷贝对象, 与原对象完全独立"""
        return deepcopy(self)  # 确保子类可直接继承使用

    def to_pos_index(self, key):
        """将物理索引转换为位置索引, 单点索引为浮点数, 切片索引自动右对齐"""
        # 切片物理索引
        if isinstance(key, slice):
            start = self.to_pos_index(key.start)
            start = int(np.ceil(start - 1e-5)) if isinstance(start, float) else start
            stop = self.to_pos_index(key.stop)
            stop = int(np.ceil(stop - 1e-5)) if isinstance(stop, float) else stop
            return slice(start, stop, key.step)
        # 单点物理索引
        if isinstance(key, str):
            unit = re.sub(r"^\$+|\$+$", "", self.unit)
            pattern = r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*" + re.escape(unit)
            match = re.fullmatch(pattern, key)
            if not match:
                raise IndexError(f"index={key}: 物理索引解析失败. 请使用'value unit'格式的物理索引")
            val = float(match.group(1))  # 提取数值部分并转换为浮点数
            # 转换为位置索引, 支持非对齐索引
            idx = (val - self._x0) / self._dx
            idx = 0 if idx < 0 else idx  # 左越界设为0避免负索引
            return idx  # 由numpy处理右越界
        return key


# --------------------------------------------------------------------------------------------#
class Series(NDArrayOperatorsMixin):
    """
    通用序列数据类, 用于保存和管理以一维序列数据及其坐标轴

    Series类及其子类均支持各种运算符操作, 以及NumPy函数兼容, 并尽可能保持类型

    Attributes
    ----------
    axis : Axis
        序列坐标轴
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
    - set_label(label: str)
        修改序列标签并返回自身

    - copy()
        返回拷贝对象, 与原对象完全独立

    - plot(**kwargs) -> Tuple
        绘制序列数据的波形图

    - template(data: Optional[np.ndarray] = None)
        继承元信息生成新对象, 方便快速创建同类对象
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
        通用序列数据类, 用于保存和管理以一维序列数据及其坐标轴

        Parameters
        ----------
        axis : Axis
            序列坐标轴
        data : np.ndarray, optional
            序列数据数组
        name : str, optional
            序列数据名称
        unit : str, optional
            序列数据单位
        label : str, optional
            序列数据标签
        """
        # Series类核心维护参数
        self._axis: Axis = axis.copy()  # _axis优先级高于_data
        self.name: str = name
        self.unit: str = unit
        self.label: str = label
        self._data: np.ndarray = np.asarray(data, copy=self._COPY_ARRAY) if data is not None else np.zeros(len(axis))
        if self._data.flags.writeable is False:
            self._data = np.array(self._data, copy=True)
        if self._check_data(self._data) is False:
            raise ValueError(f"data={self._data}: 输入序列数据数组非法. 避免使用非一维数组, 长度不匹配或包含NaN值")

    _COPY_ARRAY = None  # 默认不复制, 若numpy判定复制不可避免则仍然复制

    # --------------------------------------------------------------------------------#
    # 动态可读属性
    @property
    def axis(self) -> Axis:
        """序列坐标轴"""
        return self._axis  # 支持内容修改

    @axis.setter
    def axis(self, value: Axis):
        """序列坐标轴"""
        self._axis = value  # 支持整体替换

    @property
    def data(self) -> np.ndarray:
        """序列数据数组"""
        arr = self._data.view()  # 返回源数据视图, 避免内存拷贝消耗
        arr.flags.writeable = False  # 防止用户意外修改._data属性
        return arr

    @data.setter
    def data(self, value: np.ndarray):
        # 支持整体替换数据, 但需合法
        if self._check_data(value) is False:
            raise ValueError(f"data={value}: 输入序列数据数组非法. 避免使用非一维数组, 长度不匹配或包含NaN值")
        self._data = np.asarray(value, copy=self._COPY_ARRAY)
        if self._data.flags.writeable is False:
            self._data = np.array(self._data, copy=True)

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
        return f"{type(self).__name__}[{self.label}]({self.name}={self._data}{self.unit}, {self._axis})"

    def __repr__(self) -> str:
        """面向开发时"""
        return f"{type(self).__name__}(axis={repr(self._axis)}, data={repr(self._data)}, name='{self.name}', unit='{self.unit}', label='{self.label}')"  # noqa: E501

    def __len__(self) -> int:
        return len(self._axis)

    def __eq__(self, other) -> bool:
        if isinstance(other, Series):
            return self._axis == other._axis and np.allclose(self._data, other._data) and self.unit == other.unit
        return False  # 与非Series类型比较均返回False

    # --------------------------------------------------------------------------------#
    # 数组特性支持
    def __getitem__(self, index):
        pos_idx = self._axis.to_pos_index(index)
        if isinstance(pos_idx, float):
            idx_round = round(pos_idx)
            if abs(pos_idx - idx_round) < 1e-5:
                return self._data[idx_round]
            else:
                raise IndexError(f"index={index}: 不在坐标轴上, 无法索引到对应位置")
        if isinstance(pos_idx, slice):
            new_axis = self._axis[pos_idx]
            if isinstance(new_axis, type(self._axis)):
                new_srs = type(self)(
                    axis=new_axis,
                    data=np.asarray(self._data[pos_idx], copy=self._COPY_ARRAY),
                    name=self.name,
                    unit=self.unit,
                    label=self.label,
                )
                return new_srs
        return self._data[pos_idx]

    def __setitem__(self, index, value):
        # 支持用户通过索引部分修改数据, 长度保持不变
        self._data[index] = value

    # --------------------------------------------------------------------------------#
    # numpy互操作性兼容

    # 普通接口函数兼容
    def __array_function__(self, func, types, args, kwargs):
        # 将输入中的Series对象转为array以便函数处理
        args = [x._data if isinstance(x, Series) else x for x in args]
        # 执行NumPy的函数操作
        res = func(*args, **kwargs)
        # 检查结果，保持返回类型一致
        if isinstance(res, np.ndarray) and res.shape == self._data.shape and np.issubdtype(res.dtype, np.number):
            new_srs = self.template(res)
            return new_srs
        else:
            return res

    # 底层运算函数兼容
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # ------------------------------------------------------------------------#
        # 支持out参数：将out中的Series对象替换为其_data属性便于in-place修改
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
        # ------------------------------------------------------------------------#
        # 将输入中的Series对象转为array以便ufunc处理
        args = [x._data if isinstance(x, Series) else x for x in inputs]
        # 根据method调用相应的ufunc方法
        if method == "at":  # 处理就地操作（如add.at等，不支持）
            return NotImplemented
        if (
            method == "reduce" or method == "reduceat" or method == "outer"
        ):  # 处理非逐元素操作（如add.reduce等，极少使用）
            res = getattr(ufunc, method)(*args, **kwargs)
            return res
        elif method == "__call__" or method == "accumulate":  # 处理逐元素运算（如abs、multiply等，常用）
            res = getattr(ufunc, method)(*args, **kwargs)
            # 如果指定了out参数, 则直接返回out
            if out is not None:
                return out if len(out) > 1 else out[0]
            # 检查结果，保持返回类型一致
            if isinstance(res, np.ndarray) and res.shape == self._data.shape and np.issubdtype(res.dtype, np.number):
                new_srs = self.template(res)
                return new_srs
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
        """修改序列标签并返回自身"""
        self.label = label
        return self

    def copy(self):
        """返回拷贝对象, 与原对象完全独立"""
        return deepcopy(self)

    def plot(self, **kwargs) -> Tuple:
        """绘制序列数据的波形图"""
        from .._Plot_Module.LinePlot import PlotFunc_waveform

        fig, axs = PlotFunc_waveform(self, **kwargs)
        return fig, axs

    def template(self, data: Optional[np.ndarray] = None):
        """继承元信息生成新对象, 方便快速创建同类对象"""
        new_srs = type(self)(
            axis=self._axis,
            data=data,
            name=self.name,
            unit=self.unit,
            label=self.label,
        )
        return new_srs


# --------------------------------------------------------------------------------------------#
class t_Axis(Axis):
    """
    时间坐标轴类

    Attributes
    ----------
    N : int
        采样点数
    fs : float
        采样频率[Hz]
    dt : float
        采样间隔[s]
    t0 : float
        采样起始时间[s]
    T : float
        采样时长[s]
    data : np.ndarray
        采样时刻数组[s]
    lim : tuple
        采样时间范围[s]: (min, max)

    Methods
    -------
    - copy()
        返回拷贝对象, 与原对象完全独立

    - to_pos_index(key)
        将物理索引转换为位置索引

    - to_f_axis(f0: float = 0.0)
        转换为频率坐标轴
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
        N : int, optional
            采样点数
        fs : float, optional
            采样频率[Hz]
        dt : float, optional
            采样间隔[s]
        T : float, optional
            采样时长[s]
        t0 : float, default: 0.0
            采样起始时间[s]
        """
        # 输入参数检查
        if (not [N, fs, dt, T].count(None) == 2) or (fs is not None and dt is not None):
            raise ValueError(
                f"N={N}, fs={fs}, dt={dt}, T={T}: 采样参数输入错误. 请确保输入两个参数, 且fs与dt不能同时输入"
            )
        # 采样参数初始化, 将所有可能输入类型转为fs和N参数
        if fs is None:
            if dt is not None:
                fs = 1.0 / dt
            elif T is not None and N is not None:
                fs = N / T
            else:
                raise ValueError(
                    f"N={N}, fs={fs}, dt={dt}, T={T}: 采样参数输入错误. 请输入fs或dt参数, 或同时输入N与T参数"
                )
        if N is None:
            if T is not None:
                N = int(T * fs)
            else:
                raise ValueError(f"N={N}, fs={fs}, dt={dt}, T={T}: 采样参数输入错误. 输入fs参数时, 需同时输入N或T参数")
        super().__init__(N=N, dx=1.0 / fs, x0=t0, unit="s", name="时间")

    # --------------------------------------------------------------------------------#
    # 动态可读属性. Axis类核心参数映射到子类自定义属性，支持读写
    @property
    def fs(self) -> float:
        """采样频率[Hz], 修改同步至 dt"""
        return 1.0 / self._dx

    @fs.setter
    def fs(self, value: float):
        if value <= 0:
            raise ValueError(f"fs={value}: 采样频率必须大于0")
        self._dx = 1.0 / float(value)

    @property
    def dt(self) -> float:
        """采样间隔[s], 修改同步至 fs"""
        return self._dx

    @dt.setter
    def dt(self, value: float):
        if value <= 0:
            raise ValueError(f"dt={value}: 采样间隔必须大于0")
        self._dx = float(value)

    @property
    def t0(self) -> float:
        """采样起始时间[s]"""
        return self._x0

    @t0.setter
    def t0(self, value: float):
        self._x0 = float(value)

    @property
    def T(self) -> float:
        """采样时长[s], 修改同步至 N"""
        return self.N * self.dt

    @T.setter
    def T(self, value: float):
        if value <= 0:
            raise ValueError(f"T={value}: 采样时长必须大于0")
        # 固定 dt，调整 N
        self.N = max(1, int(np.ceil(value / self.dt - 1e-5)))

    # ----------------------------------------------------------------------------#
    # 外部用户方法
    def to_f_axis(self, f0: float = 0.0) -> "f_Axis":
        """
        转换为频率坐标轴

        Parameters
        ----------
        f0 : float, default: 0.0
            频率起始点[Hz]

        Returns
        -------
        f_Axis
            频率坐标轴
        """
        return f_Axis(N=self.N, F=self.fs, f0=f0)


class f_Axis(Axis):
    """
    频率坐标轴类

    Attributes
    ----------
    N : int
        采样点数
    df : float
        频率分辨率[Hz]
    f0 : float
        频率起始点[Hz]
    F : float
        频率分布宽度[Hz]
    data : np.ndarray
        频率轴数组[Hz]
    lim : tuple
        频率分布范围[Hz]: (min, max)

    Methods
    -------
    - copy()
        返回拷贝对象, 与原对象完全独立

    - to_pos_index(key)
        将物理索引转换为位置索引
    """

    def __init__(self, N: Optional[int] = None, df: Optional[float] = None, F: Optional[float] = None, f0: float = 0.0):
        """
        频率坐标轴类

        Parameters
        ----------
        N : int, optional
            采样点数
        df : float, optional
            频率分辨率[Hz]
        F : float, optional
            频率分布宽度[Hz]
        f0 : float, default: 0.0
            频率起始点[Hz]
        """
        if not [N, df, F].count(None) == 1:
            raise ValueError(f"N={N}, df={df}, F={F}: 频率参数输入错误. 请确保输入两个参数")
        if df is None:
            if F is not None and N is not None:
                df = F / N
            else:
                raise ValueError(f"N={N}, df={df}, F={F}: 频率参数输入错误. 请输入df参数, 或同时输入N与F参数")
        if N is None:
            if F is not None:
                N = int(F / df)
            else:
                raise ValueError(f"N={N}, df={df}, F={F}: 频率参数输入错误. 输入df参数时, 需同时输入N或F参数")
        super().__init__(N=N, dx=df, x0=f0, unit="Hz", name="频率")

    # --------------------------------------------------------------------------------#
    # 动态可读属性. Axis类核心参数映射到子类自定义属性，支持读写
    @property
    def df(self) -> float:
        """频率分辨率[Hz]"""
        return self._dx

    @df.setter
    def df(self, value: float):
        if value <= 0:
            raise ValueError(f"df={value}: 频率分辨率必须大于0")
        self._dx = float(value)

    @property
    def f0(self) -> float:
        """频率起始点[Hz]"""
        return self._x0

    @f0.setter
    def f0(self, value: float):
        self._x0 = float(value)

    @property
    def F(self) -> float:
        """频率分布宽度[Hz], 修改同步至 N"""
        return self.N * self._dx  # 频率分布宽度

    @F.setter
    def F(self, value: float):
        if value <= 0:
            raise ValueError(f"F={value}: 频率分布宽度必须大于0")
        # 固定 df，调整 N
        self.N = max(1, int(np.ceil(value / self._dx - 1e-5)))


# --------------------------------------------------------------------------------------------#
class Signal(Series):
    """
    一维时域信号类, 实现采样信息与数据的绑定, 支持混合运算并内置常用信号数据交互方法

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
    - set_label(label: str)
        修改序列标签并返回自身

    - copy()
        返回拷贝对象, 与原对象完全独立

    - plot(**kwargs) -> Tuple
        绘制序列数据的波形图

    - template(data: Optional[np.ndarray] = None)
        继承元信息生成新对象, 方便快速创建同类对象

    - to_Spectra() -> Spectra
        转换信号到其频谱
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
        一维时域信号数据类, 实现采样信息与数据的绑定, 支持混合运算并内置常用信号数据交互方法

        Parameters
        ----------
        axis : t_Axis
            时间坐标轴
        data : np.ndarray, optional
            信号数据数组
        name : str, default: ""
            信号数据名称
        unit : str, default: ""
            信号数据单位
        label : str, default: ""
            信号标签
        """
        super().__init__(axis=axis, data=data, name=name, unit=unit, label=label)

    # --------------------------------------------------------------------------------#
    # 动态可读属性: 坐标轴别名映射
    @property
    def t_axis(self) -> t_Axis:
        """时间坐标轴"""
        return self._axis

    @t_axis.setter
    def t_axis(self, value: t_Axis):
        """时间坐标轴"""
        self._axis: t_Axis = value

    # --------------------------------------------------------------------------------#
    # 外部用户方法
    def to_Spectra(self) -> "Spectra":
        """转换信号到其频谱"""
        from .._Analysis_Module.SpectrumAnalysis import SpectrumAnalysis

        spc = SpectrumAnalysis(self).cft(winType="矩形窗", padTimes=0)  # 保持原始长度, 不延拓
        return spc


class Spectra(Series):
    """
    一维频谱数据类, 实现采样信息与数据的绑定, 支持混合运算并内置常用频谱数据交互方法

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
    - set_label(label: str)
        修改序列标签并返回自身

    - copy()
        返回拷贝对象, 与原对象完全独立

    - plot(**kwargs) -> Tuple
        绘制序列数据的波形图

    - template(data: Optional[np.ndarray] = None)
        继承元信息生成新对象, 方便快速创建同类对象

    - halfCut() -> Self
        裁剪为单边谱
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
        一维频谱数据类, 实现采样信息与数据的绑定, 支持混合运算并内置常用信号数据交互方法

        Parameters
        ----------
        axis : f_Axis
            频率坐标轴
        data : np.ndarray
            频谱数据数组
        name : str
            频谱数据名称
        unit : str
            频谱数据单位
        label : str
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
        """频率坐标轴"""
        self._axis: f_Axis = value

    # --------------------------------------------------------------------------------#
    # 外部用户方法
    def halfCut(self) -> Self:
        """裁剪为单边谱"""
        if self.f_axis.f0 != 0.0:
            raise ValueError(f"f0={self.f_axis.f0}: 当前谱频率轴不完整, 无法进行单边谱裁剪")
        N = len(self)
        if N % 2 == 0:  # 偶数点，非对称
            half_N = N // 2
        else:  # 奇数点，对称
            half_N = (N + 1) // 2  # 包含fn频率点, 但幅值一般为0

        self._axis, self._data = (
            self._axis[:half_N],
            self._data[:half_N],
        )  # 原地修改当前对象
        self._data[1:] *= 2  # 除直流分量外幅值翻倍
        return self
