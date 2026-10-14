# /// script
# dependencies = [
#     "anytree==2.13.0",
#     "marimo",
#     "matplotlib==3.11.2",
#     "numpy==2.5.3",
#     "pandas==3.0.6",
#     "scipy==1.18.1",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    import warnings

    warnings.filterwarnings(
        "ignore",
        message="FigureCanvasAgg is non-interactive",
        category=UserWarning,
    )
    import signaltour as st


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Application test1: 航空发动机振动响应信号分析
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. 数据导入
    """)
    return


@app.cell
def _():
    Dataset = st.Dataset(
        root=r"F:\OneDrive\Database\横向课题数据集\微型涡喷发动机试车振动数据集(第三批)\Data",
        type=".txt",
    )
    Dataset.info()
    return (Dataset,)


@app.cell
def _(Dataset):
    Datafiles: st.Files = Dataset["我方自采数据"]["振动测试实验"][
        "第二次测试17点"
    ].files
    Datafiles.preview(
        sep="\t",
        header=None,
        names=["机端径向振动信号", "机端轴向振动信号", "台架径向振动信号"],
    )
    return (Datafiles,)


@app.cell(expand_output=True)
def _(Datafiles: st.Files):
    Datafiles.set_read_params(
        filetype=".txt",
        sep="\t",
        header=None,
        names=["机端径向振动信号", "机端轴向振动信号", "台架径向振动信号"],
    )
    Loaddata: st.LoadData = Datafiles.load(isParallel=True, usePyarrow=True)
    Loaddata.merge(mode="vstack")
    Dictsig = Loaddata.to_Signal(fs=51200, name="加速度", unit="$g$")[0]
    _Listfig = []
    for _sig in Dictsig.values():
        _Listfig.append(_sig.plot()[0])
    mo.vstack(_Listfig, align="center")
    return (Dictsig,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. 数据分析预处理
    """)
    return


@app.cell
def _(Dictsig):
    _Listfig = []
    for _sig in Dictsig.values():
        st.SpectrumAnalysis(sig=_sig, isPlot=True).cft()
        _Listfig.append(plt.gcf())
    mo.vstack(_Listfig, align="center")
    return


if __name__ == "__main__":
    app.run()
