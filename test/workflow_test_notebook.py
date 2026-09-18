# /// script
# dependencies = [
#     "anytree==2.13.0",
#     "marimo>=0.23.3",
#     "matplotlib==3.11.2",
#     "numpy==2.5.3",
#     "pandas==3.0.5",
#     "pyarrow==25.0.1",
#     "python-lsp-ruff==2.3.4",
#     "python-lsp-server==1.15.0",
#     "scipy==1.18.1",
#     "websockets==17.1",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt

    import warnings

    warnings.filterwarnings(
        "ignore",
        message="FigureCanvasAgg is non-interactive",
        category=UserWarning,
    )
    import logging, sys
    logger = logging.getLogger("signaltour")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: \n\t%(message)s"))
    logger.addHandler(handler)

    import signaltour as st


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # signaltour完整工作流测试
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. 数据读取
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - .mat类型数据文件
    """)
    return


@app.cell
def _():
    dataset = st.Dataset(
        root=r"F:\OneDrive\Database\学术公开数据集\故障诊断\CWRU_轴承故障振动数据集\Data",
        type=".mat",
        name="CWRU故障轴承振动数据集",
    )
    dataset.info()
    return (dataset,)


@app.cell
def _(dataset):
    datafolder = dataset["12k Drive End Bearing Fault Data"]
    data=datafolder.loadMatch(match="0007",filter="_0",merge=False)
    data.table
    return data, datafolder


@app.cell
def _(data):
    data["chain.str.contains('Outer Race')"].table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - .csv类型数据文件
    """)
    return


@app.cell
def _():
    dataset_csv=st.Dataset(root=r"F:\OneDrive\Database\学术公开数据集\寿命预测\XJTU_轴承加速退化振动数据集\Data",type='.csv')
    dataset_csv.info()
    return (dataset_csv,)


@app.cell
def _(datafolder, dataset_csv):
    datafolder_csv=dataset_csv['35Hz12kN']
    datafolder.stats
    return (datafolder_csv,)


@app.cell
def _(datafolder_csv):
    data_csv=datafolder_csv.loadAll(isParallel=True,parallelNum=10,usePyarrow=True).merge('vstack')
    data_csv.table
    return (data_csv,)


@app.cell
def _(data_csv):
    _Listfig=[]
    for _sig in data_csv.to_Signal(fs=35000,name='加速度'):
        _sig.plot()
        _Listfig.append(plt.gcf())
    mo.vstack(_Listfig,align='center')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. 信号预处理
    """)
    return


@app.cell
def _(Listsig):
    _Listfig=[]
    for _sig in Listsig:
        _sig.plot()
        _Listfig.append(plt.gcf())
    mo.vstack(_Listfig,align='center')
    return


@app.cell
def _(sig):
    _stat = st.StatsTrendAnalysis(sig).evaluate()
    st.SpectrumAnalysis(sig, isPlot=True).cft()
    mo.hstack([plt.gcf(), _stat], align="center")
    return


@app.cell
def _(sig):
    sig_filtered = st.filtFIR(
        sig, cutoff=(3000, 3800), order=128, btype="bandpass"
    )
    _stat = st.StatsTrendAnalysis(sig_filtered).evaluate()
    st.SpectrumAnalysis(sig_filtered, isPlot=True).cft()
    mo.hstack([plt.gcf(), _stat], align="center")
    return (sig_filtered,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. 信号特征提取
    """)
    return


@app.cell
def _(sig_filtered):
    spc_IA = st.HilbertAnalysis(sig_filtered, isPlot=True).envelopeSpectrum()
    spc_IA[:10] = 0
    spc_IA.plot(xlim=(0, 300))
    mo.vstack([plt.gcf()], align="center")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. 批量对比分析
    """)
    return


@app.cell
def _(dataset):
    dataset.loadMatch(match="12k, Drive End, 021", filter="_0")
    return


@app.cell
def _(dataset):
    _datafiles_dict = dataset.loadMatch(
        match="12k, Drive End, 0021", filter="_0"
    )
    for _loc, _df in _datafiles_dict.items():
        _fig_list = []
        _sig = st.Signal(
            axis=st.t_Axis(N=len(_df), fs=12000),
            data=_df.iloc[:, 0],
            name="加速度",
            unit="$g$",
            label=f"驱动端{_df.columns[0]}振动信号",
        )
        _sig.plot()
        _fig_list.append(plt.gcf())
        _spc_IA = st.HilbertAnalysis(_sig).envelopeSpectrum()
        _spc_IA[:10] = 0
        _spc_IA["0Hz":"300Hz"].plot()
        _fig_list.append(plt.gcf())
        mo.output.append(mo.vstack(_fig_list, align="center"))
    return


if __name__ == "__main__":
    app.run()
