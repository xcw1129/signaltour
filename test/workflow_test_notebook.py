import marimo

__generated_with = "0.24.2"
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
    # signaltour完整工作流测试
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. 数据导入
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.1 数据文件读取
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
    dataset_mat = st.Dataset(
        root=r"F:\OneDrive\Database\学术公开数据集\故障诊断\CWRU_轴承故障振动数据集\Data",
        type=".mat",
        name="CWRU故障轴承振动数据集",
    )
    dataset_mat.info()
    return (dataset_mat,)


@app.cell
def _(dataset_mat):
    datafolder_mat = dataset_mat["12k Drive End Bearing Fault Data"]
    data_mat = datafolder_mat.loadMatch(match="Outer, 0021", filter="_0")
    data_mat()
    return (data_mat,)


@app.cell
def _(data_mat):
    _df_mat_list = data_mat.to_Dataframe()
    _df_mat_list
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - .csv类型数据文件
    """)
    return


@app.cell
def _():
    dataset_csv = st.Dataset(
        root=r"F:\OneDrive\Database\学术公开数据集\寿命预测\XJTU_轴承加速退化振动数据集\Data",
        type=".csv",
        name="XJTU轴承加速退化振动数据集",
    )
    dataset_csv.info()
    return (dataset_csv,)


@app.cell
def _(dataset_csv):
    datafolder_csv = dataset_csv["35Hz12kN"]
    data_csv = datafolder_csv.loadAll(
        isParallel=True, parallelNum=10, usePyarrow=True
    ).merge("vstack")
    data_csv()
    return (data_csv,)


@app.cell
def _(data_csv):
    _sig_csv_list = data_csv.to_Signal(fs=35000, name="加速度", unit="g")[0]
    for _sig in _sig_csv_list:
        _sig.plot()
        mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.2 仿真信号生成
    """)
    return


@app.cell
def _():
    _sig = st.periodic(
        fs=10000,
        T=1,
        CosParams=(
            (3.3, 2.1, 1.5),
            (23, 3.2, 7.4),
            (75, 1.7, 2.6),
            (127, 6.4, 9.2),
        ),
        noise=5,
    )
    _sig.plot()
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell
def _():
    _sig = st.impulse(
        fs=10000,
        T=1,
        ImpParams=(
            2300,
            100,
            0.02,
            lambda t: 10 + 10 * np.cos(2 * np.pi * 20 * t),
            0.002,
        ),
        noiseParams=(20, 10),
    )
    _sig.plot()
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell
def _():
    _sig = st.modulation(
        fs=10000,
        T=1,
        fc=50,
        AM=lambda t: 2 + (5 * t) ** 2,
        FM=lambda t: 50 * t,
    )
    _sig.plot()
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell
def _():
    sig = st.Signal(axis=st.t_Axis(fs=10000, T=1), label="仿真信号")
    sig += st.periodic(
        fs=10000,
        T=1,
        CosParams=(
            (3.3, 2.1, 1.5),
            (23, 3.2, 7.4),
            (75, 1.7, 2.6),
            (127, 6.4, 9.2),
        ),
        noise=5,
    )
    sig += st.impulse(
        fs=10000,
        T=1,
        ImpParams=(
            2300,
            100,
            0.02,
            lambda t: 10 + 10 * np.cos(2 * np.pi * 20 * t),
            0.002,
        ),
        noiseParams=(20, 10),
    )
    sig += st.modulation(
        fs=10000,
        T=1,
        fc=50,
        AM=lambda t: 2 + (5 * t) ** 2,
        FM=lambda t: 50 * t,
    )
    sig.plot()
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return (sig,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. 信号预处理
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2.1 数据转换
    """)
    return


@app.cell
def _(sig):
    _sig = sig.resample(type="extreme", dt=0.001)
    _sig.plot()
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    sig.plot()
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell
def _(sig):
    _sig = sig.pad(pad_width=1000, mode="reflect")
    _sig.plot()
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    sig.plot()
    plt.xlim(-0.1,1.1)
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell
def _(sig):
    _sig_seg_list = sig.segment(nseg=15, overlap=0.5)
    for _sig in _sig_seg_list:
        _sig.plot()
        mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2.2 信号滤波
    """)
    return


@app.cell
def _(sig):
    _sig = sig.filtFIR(
        cutoff=(2000, 3000), order=256, btype="bandpass", zero_phase=True
    )
    _sig.plot()
    plt.ylim(-20, 20)
    plt.yticks([-15, -10, -5, 0, 5, 10, 15])
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell
def _(sig):
    _sig = sig.filtIIR(cutoff=(2000, 3000), btype="bandpass", ftype="butter")
    _sig.plot()
    plt.ylim(-20, 20)
    plt.yticks([-15, -10, -5, 0, 5, 10, 15])
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell
def _(sig):
    _sig = sig.filtMedian(size=11)
    _sig.plot()
    plt.ylim(-50,50)
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    sig.plot()
    plt.ylim(-50,50)
    mo.output.append(mo.vstack([plt.gcf()], align="center"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. 信号特征提取
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
