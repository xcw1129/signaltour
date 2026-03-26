# /// script
# requires-python = "==3.11.12"
# dependencies = [
#     "marimo>=0.20.2",
#     "numpy==2.0.0",
#     "scipy==1.14.0",
#     "matplotlib==3.9.0",
#     "pandas==2.2.2",
#     "anytree==2.13.0",
#     "pyzmq",
#     "pytest",
#     "openai",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", app_title="test_Signal")

with app.setup(hide_code=True):
    import warnings

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore", category=UserWarning)

    from signaltour import Signal


@app.cell(hide_code=True)
def _():
    mo.md("""
    # signaltour-Signal子包功能测试
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1. core模块
    """)
    return


@app.function
def IS_Like_array(obj, array):
    """检查输入对象是否表现为array类"""
    # 测试数组行为
    assert len(obj) == len(array)
    np.testing.assert_allclose([x for x in obj], array)
    np.testing.assert_allclose(
        obj[array > array.mean()], array[array > array.mean()]
    )
    assert obj[0] in obj
    assert obj[0] + 1e-8 not in obj
    assert isinstance(obj[1:6:2], type(obj))  # 索引结果类型继承
    # 测试与numpy的兼容性
    np.testing.assert_allclose(np.asarray(obj), array)
    np.testing.assert_allclose(np.mean(obj), np.mean(array))
    np.testing.assert_allclose(np.square(obj), np.square(array))


@app.function
def IS_Support_operator(obj, array):
    """检查输入对象是否支持运算符兼容性与numpy互操作性"""
    cls = type(obj)

    # -------------------------------------------------------------------------#
    # 1. 标准算术运算 (检查类型继承与数值正确性)
    # 以 + 运算为例做详尽检查
    assert isinstance(obj + 1, cls)
    assert isinstance(obj + array, cls)
    assert isinstance(obj + obj, cls)
    np.testing.assert_allclose((obj + 1).data, array + 1)
    np.testing.assert_allclose((obj + array).data, array + array)

    # 反向运算 (Reflected)
    assert isinstance(1 + obj, cls)
    assert isinstance(array + obj, cls)
    np.testing.assert_allclose((1 + obj).data, 1 + array)

    # 其他运算符简要检查
    operators = {
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
        "**": lambda a, b: a**b,
        "//": lambda a, b: a // b,
        "%": lambda a, b: a % b,
    }

    for name, op in operators.items():
        res = op(obj, 2)
        assert isinstance(res, cls), f"运算符 {name} 类型保持失败"
        # 反向
        res_r = op(2, obj)
        assert isinstance(res_r, cls), f"运算符 {name} 反向类型保持失败"

    # -------------------------------------------------------------------------#
    # 2. 一元运算符 (Unary Operators)
    assert isinstance(-obj, cls)
    np.testing.assert_allclose((-obj).data, -array)

    assert isinstance(abs(obj), cls)
    np.testing.assert_allclose(abs(obj).data, np.abs(array))

    # -------------------------------------------------------------------------#
    # 3. 就地运算符 (In-place Operators)
    obj_copy = obj.copy()
    original_data = obj_copy.data.base

    obj_copy += 1
    assert np.shares_memory(obj_copy.data, original_data), "+= 产生了新对象"
    np.testing.assert_allclose(obj_copy.data, array + 1)

    obj_copy *= 2
    assert np.shares_memory(obj_copy.data, original_data), "*= 产生了新对象"
    np.testing.assert_allclose(obj_copy.data, (array + 1) * 2)

    # -------------------------------------------------------------------------#
    # 4. 比较运算符 (Comparison Operators)
    res_gt = obj > 0
    assert isinstance(res_gt, np.ndarray), "比较运算不应返回 Series 实例"
    assert res_gt.dtype == bool
    np.testing.assert_array_equal(res_gt, array > 0)

    # -------------------------------------------------------------------------#
    # 5. NumPy 通用函数兼容性 (Basic ufunc check)
    res_sin = np.sin(obj)
    assert isinstance(res_sin, cls), "np.sin 应该返回类实例"
    np.testing.assert_allclose(res_sin.data, np.sin(array))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Axis()
    """)
    return


@app.function
def test_Axis():
    # 创建实例
    axis = Signal.Axis(N=10, dx=1, x0=0, name="位移", unit="mm")
    assert isinstance(axis, Signal.Axis)
    # 测试属性
    np.testing.assert_allclose(
        axis.data, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    )
    np.testing.assert_allclose(
        axis(), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    )
    axis._dx, axis._x0, axis.N = 2, -2, 3
    np.testing.assert_allclose(axis.data, np.array([-2, 0, 2]))
    axis._dx, axis._x0, axis.N = 1, 0, 10
    assert axis.lim == (0, 10) and axis.L == 10
    assert (
        isinstance(axis.label, str)
        and "位移" in axis.label
        and "mm" in axis.label
    )
    # 测试相等判断
    assert axis == Signal.Axis(N=10, dx=1, x0=0, unit="mm")
    assert axis != Signal.Axis(N=10, dx=1, x0=0, unit="cm")
    assert axis != Signal.Axis(N=10, dx=2, x0=0, unit="mm")
    assert axis != axis.data
    # 测试物理坐标索引
    assert axis["2mm"] == 2
    assert axis["1.5mm":"6.5mm":2] == axis[2:7:2]
    # 测试array行为
    IS_Like_array(axis, axis.data)
    # 测试方法
    # 测试.copy()
    assert axis.copy() == axis and axis.copy() is not axis


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Series()
    """)
    return


@app.function
def test_Series():
    # 创建实例
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    axis = Signal.Axis(N=5, dx=0.1, x0=0, name="时间", unit="s")
    Signal.Series._COPY_DATA_WHEN_INIT = False
    series = Signal.Series(
        data=data, axis=axis, name="压力", unit="Pa", label="锅炉压力"
    )
    assert isinstance(series, Signal.Series)
    # 测试属性
    np.testing.assert_allclose(series.data, data)
    assert series.axis == axis and series.axis is not axis
    assert (
        isinstance(series.name, str)
        and isinstance(series.unit, str)
        and isinstance(series.label, str)
    )
    # 测试对data的维护方式
    assert series.data is not data
    assert series.data.flags["OWNDATA"] is False
    assert series.data.flags["WRITEABLE"] is False
    assert series.data.view().flags["WRITEABLE"] is False
    assert np.shares_memory(series.data, data) is True
    Signal.Series._COPY_DATA_WHEN_INIT = True
    series.data = data
    assert np.shares_memory(series.data, data) is False
    Signal.Series._COPY_DATA_WHEN_INIT = False
    series.data = data
    # 测试相等判断
    assert series == Signal.Series(
        data=data, axis=axis, name="压力", unit="Pa"
    )
    assert series != Signal.Series(
        data=data, axis=axis, name="压力", unit="MPa"
    )
    assert series != Signal.Series(
        data=data + 1, axis=axis, name="压力", unit="MPa"
    )
    # 测试物理坐标索引
    assert series["0.1s"] == 2.0
    assert series["0.05s":"0.35s":2] == series[1:4:2]
    # 测试array行为
    IS_Like_array(series, data)
    # 测试运算符兼容性与numpy互操作性
    IS_Support_operator(series, data)
    # 测试方法
    # 测试拷贝操作
    series_copy = series.copy()
    assert series_copy == series and series_copy is not series
    assert series_copy.axis is not series.axis
    assert series_copy.data.base is not series.data.base
    # 测试链式调用
    fig, axs = series.template(data).set_label("水箱压力").plot()
    assert isinstance(fig, plt.Figure)
    mo.output.append(axs.flatten()[0])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Signal()
    """)
    return


@app.function
def test_Signal():
    # 创建实例
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    axis = Signal.t_Axis(len(data), fs=10)
    Signal.Signal._COPY_DATA_WHEN_INIT = False
    signal = Signal.Signal(
        data=data, axis=axis, name="振动", unit="$m/s^2$", label="测点信号"
    )
    assert isinstance(signal, Signal.Signal)
    # 测试方法
    fig, axs = signal.plot()
    assert isinstance(fig, plt.Figure)
    mo.output.append(fig)
    fig, axs = np.abs(signal.to_Spectra()).halfCut().plot()
    mo.output.append(fig)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Spectra()
    """)
    return


@app.function
def test_Spectra():
    # 创建实例
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    axis = Signal.f_Axis(len(data), df=1)
    spectra = Signal.Spectra(
        data=data, axis=axis, name="幅值", unit="$m/s^2$", label="测点信号"
    )
    assert isinstance(spectra, Signal.Spectra)
    # 测试方法
    fig, axs = spectra.plot()
    assert isinstance(fig, plt.Figure)
    mo.output.append(fig)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. SignalRead模块
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Files()
    """)
    return


@app.cell
def _():
    files = Signal.Files(
        root=r"R:\Data\PHM数据库\学术公开数据集\寿命预测\XJTU_轴承加速退化振动数据集\Data\35Hz12kN\Bearing1_1",
        type="csv",
    )
    files
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Folder()
    """)
    return


@app.cell
def _(dataset):
    folder = dataset["12k Drive End Bearing Fault Data"]
    folder.info()
    return (folder,)


@app.cell
def _(folder):
    folder.loadMatch(match="0007", merge=False)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Dataset()
    """)
    return


@app.cell
def _():
    dataset = Signal.Dataset(
        root=r"R:\Data\PHM数据库\学术公开数据集\故障诊断\CWRU_轴承故障振动数据集\Data",
        type=".mat",
        name="CWRU轴承故障振动数据集",
    )
    dataset.info()
    return (dataset,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. SignalSimulate模块
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.periodic()
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.impulse()
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.modulation()
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. SignalSample模块
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.resample()
    """)
    return


@app.cell
def _():
    t_axis = Signal.t_Axis(N=1000, fs=1000)
    data = np.random.randn(1000)
    signal = Signal.Signal(data=data, axis=t_axis, name="随机信号", unit="V")
    x = signal.data[:100]
    x = x[::2]
    x.flags
    return


if __name__ == "__main__":
    app.run()
