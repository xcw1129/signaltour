# /// script
# requires-python = "==3.11.12"
# dependencies = [
#     "marimo>=0.19.0",
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

__generated_with = "0.22.0"
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


@app.cell
def _():
    from signaltour import Signal

    Sig_test=Signal.periodic(
        fs=4000,
        T=2,
        CosParams=((50, 18, 2.3), (120, 12, 1.8), (300, 8, 1.5)),
        noise=0.5,
    )
    Sig_test.plot()
    plt.gcf()
    return (Sig_test,)


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


@app.cell
def _(Sig_test):
    Analysis.convolveCycle(Sig_test.data, Sig_test.data)
    return


@app.cell
def _(Sig_test):
    Analysis.Spectrum(Sig_test,isPlot=True).cft()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. TimeFreqAnalysis模块
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. WaveletAnalysis模块
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. ModeAnalysis模块
    """)
    return


if __name__ == "__main__":
    app.run()
