# /// script
# requires-python = "==3.11.12"
# dependencies = [
#     "marimo",
#     "numpy==2.0.0",
#     "scipy==1.14.0",
#     "matplotlib==3.9.0",
#     "pandas==2.2.2",
#     "pyarrow==22.0.0",
#     "anytree==2.13.0",
#     "pyzmq",
#     "pytest",
#     "openai",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App()

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from signaltour import Analysis


@app.cell(hide_code=True)
def _():
    mo.md("""
    # signaltour-Analysis子包功能测试
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1. core模块
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. StatsTrendAnalysis模块
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. SpectrumAnalysis模块
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. TimeFreqAnalysis模块
    """)
    return


@app.function
def test_CWTAnalysis(Sig):
    analyzer = Analysis.CWTAnalysis(Sig=Sig)
    # 测试 get_scale 方法
    scale = analyzer.get_scale(b=3, j=5, v=10)
    assert scale[0] == 1
    assert all(scale[1:] < 1)
    assert len(scale) == 50
    np.testing.assert_allclose(scale[:-1] / scale[1:], 3 ** (1 / 10))
    # 测试 show_TFcover 方法
    time = Sig.t_axis()
    freq = 5 / scale
    analyzer.show_TFcover(
        time=time[::10], freq=freq, dfreq=10 / scale, boxArea=1
    )
    # 测试 get_wavelet 方法
    waveletMat = analyzer.get_wavelet(
        type="B-Spline",
        param={"fc": 5, "fb": 5, "p": 3},
        scale=scale,
        N=1000,
        normalize="能量",
    )
    assert waveletMat.shape == (50, 1000)
    # 测试 cwt 方法
    time, freq, Wf = analyzer.cwt(flow=20, nperoctave=50)
    assert len(time) == len(Sig)
    assert Wf.shape == (len(time), len(freq))


if __name__ == "__main__":
    app.run()
