# /// script
# requires-python = "==3.11.12"
# dependencies = [
#     "marimo",
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

__generated_with = "0.19.7"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import warnings

    import marimo as mo
    import numpy as np

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
    # 测试数组行为
    assert len(obj) == len(array)
    np.testing.assert_allclose([x for x in obj], array)
    np.testing.assert_allclose(obj[array > array.mean()], array[array > array.mean()])
    assert obj[0] in obj
    assert obj[0] + 1e-8 not in obj
    assert obj[0] + 1e-10 in obj  # 判断容差
    assert isinstance(obj[1:6:2], type(obj))  # 索引结果类型继承
    # 测试与numpy的兼容性
    np.testing.assert_allclose(np.asarray(obj), array)
    np.testing.assert_allclose(np.mean(obj), np.mean(array))
    np.testing.assert_allclose(np.square(obj), np.square(array))


@app.function
def IS_Support_operator(obj, array):
    # + 运算
    assert isinstance(obj + 1, type(obj)), isinstance(
        obj + array, type(obj)
    ) and isinstance(obj + obj, type(obj))
    np.testing.assert_allclose(obj + array, array + array)
    assert isinstance(1 + obj, type(obj)), isinstance(array + obj, type(obj))
    np.testing.assert_allclose(array + obj, array + array)
    # - 运算
    assert isinstance(obj - 1, type(obj)), isinstance(
        obj - array, type(obj)
    ) and isinstance(obj - obj, type(obj))
    np.testing.assert_allclose(obj - array, array - array)
    assert isinstance(1 - obj, type(obj)), isinstance(array - obj, type(obj))
    np.testing.assert_allclose(array - obj, array - array)
    # * 运算
    assert isinstance(obj * 2, type(obj)), isinstance(
        obj * array, type(obj)
    ) and isinstance(obj * obj, type(obj))
    np.testing.assert_allclose(obj * array, array * array)
    assert isinstance(2 * obj, type(obj)), isinstance(array * obj, type(obj))
    np.testing.assert_allclose(array * obj, array * array)
    # / 运算
    assert isinstance(obj / 2, type(obj)), isinstance(
        obj / array, type(obj)
    ) and isinstance(obj / obj, type(obj))
    np.testing.assert_allclose(obj / array, array / array)
    assert isinstance(2 / obj, type(obj)), isinstance(array / obj, type(obj))
    np.testing.assert_allclose(array / obj, array / array)
    # ** 运算
    assert isinstance(obj**2, type(obj)), isinstance(
        obj**array, type(obj)
    ) and isinstance(obj**obj, type(obj))
    np.testing.assert_allclose(obj**array, array**array)
    assert isinstance(2**obj, type(obj)), isinstance(array**obj, type(obj))
    np.testing.assert_allclose(2**obj, 2**array)
    # % 运算
    assert isinstance(obj % 2, type(obj)), isinstance(
        obj % array, type(obj)
    ) and isinstance(obj % obj, type(obj))
    np.testing.assert_allclose(obj % 0.5, array % 0.5)
    assert isinstance(2 % obj, type(obj)), isinstance(array % obj, type(obj))
    np.testing.assert_allclose(10 % obj, 10 % array)
    # // 运算
    assert isinstance(obj // 2, type(obj)), isinstance(
        obj // array, type(obj)
    ) and isinstance(obj // obj, type(obj))
    np.testing.assert_allclose(obj // 0.5, array // 0.5)
    assert isinstance(2 // obj, type(obj)), isinstance(array // obj, type(obj))
    np.testing.assert_allclose(0.5 // obj, 0.5 // array)


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
    np.testing.assert_allclose(axis.data, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    np.testing.assert_allclose(axis(), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    axis._dx, axis._x0, axis.N = 2, -2, 3
    np.testing.assert_allclose(axis.data, np.array([-2, 0, 2]))
    axis._dx, axis._x0, axis.N = 1, 0, 10
    assert axis.lim == (0, 10) and axis.L == 10
    assert isinstance(axis.label, str) and "位移" in axis.label and "mm" in axis.label
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
    assert np.shares_memory(series.data, data) == True
    assert series.data is not data
    assert series.data.flags["WRITEABLE"] == False
    Signal.Series._COPY_DATA_WHEN_INIT = True
    series.data = data
    assert np.shares_memory(series.data, data) == False
    Signal.Series._COPY_DATA_WHEN_INIT = False
    # 测试相等判断
    assert series == Signal.Series(data=data, axis=axis, name="压力", unit="Pa")
    assert series != Signal.Series(data=data, axis=axis, name="压力", unit="MPa")
    assert series != Signal.Series(data=data + 1, axis=axis, name="压力", unit="MPa")
    # 测试物理坐标索引
    assert series["0.1s"] == 2.0
    assert series["0.15s":"0.5s"] == series[2:5]
    # 测试array行为
    IS_Like_array(series, data)
    # 测试算术运算兼容
    IS_Support_operator(series, data)
    # 测试拷贝操作
    _copy = series.copy()
    assert _copy == series and _copy is not series
    # 测试链式调用
    _series_new = series.template(data).set_label("水箱压力")
    assert _series_new == series and _series_new.label == "水箱压力"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Signal()
    """)
    return


@app.function
def test_Signal():
    # 创建实例
    _data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _axis = Signal.t_Axis(len(_data), fs=10)
    Signal.Signal._COPY_DATA_WHEN_INIT = False
    _sig = Signal.Signal(
        data=_data, axis=_axis, name="振动", unit="m/s^2", label="测点1信号"
    )
    assert isinstance(_sig, Signal.Signal)
    # 测试属性
    np.testing.assert_allclose(_sig.data, _data)
    assert _sig.axis == _axis and _sig.axis is not _axis
    assert (
        isinstance(_sig.name, str)
        and isinstance(_sig.unit, str)
        and isinstance(_sig.label, str)
    )
    # 测试对data的维护方式
    assert np.shares_memory(_sig.data, _data) == True
    assert _sig.data is not _data
    assert _sig.data.flags["WRITEABLE"] == False
    Signal.Signal._COPY_DATA_WHEN_INIT = True
    _sig.data = _data
    assert np.shares_memory(_sig.data, _data) == False
    Signal.Signal._COPY_DATA_WHEN_INIT = False
    # 测试相等判断
    assert _sig == Signal.Signal(data=_data, axis=_axis, name="振动", unit="m/s^2")
    assert _sig != Signal.Signal(data=_data, axis=_axis, name="振动", unit="g")
    assert _sig != Signal.Signal(data=_data + 1, axis=_axis, name="振动", unit="g")
    # 测试物理坐标索引
    assert _sig["0.1s"] == 2.0
    assert _sig["0.15s":"0.5s"] == _sig[2:5]
    # 测试array行为
    IS_Like_array(_sig, _data)
    # 测试算术运算兼容
    IS_Support_operator(_sig, _data)
    # 测试拷贝操作
    _copy = _sig.copy()
    assert _copy == _sig and _copy is not _sig
    # 测试链式调用
    _sig_new = _sig.template(_data).set_label("测点2")
    assert _sig_new == _sig and _sig_new.label == "测点2"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Spectra()
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. SignalRead模块
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Signal.Dataset()
    """)
    return


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


@app.cell
def _():
    def _AM(t):
        return 50 + 200 * t**2 / 900

    def _FM(t):
        return 1.5 * t / 30

    _Sig = Signal.modulation(fs=1024, T=30, fc=0.25, AM=_AM, FM=_FM)

    _Sig.plot()
    return


if __name__ == "__main__":
    app.run()
