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

__generated_with = "0.19.4"
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
def test_STFTAnalysis(Sig):      
    analyzer = Analysis.STFTAnalysis(Sig=Sig)      
    # 测试 stft 方法
    time, freq, Sf = analyzer.stft(df=50, dt=0.1)      
    assert freq[0] == 0      
    assert freq[1] <= 50      
    assert time[1] - time[0] <= 0.1      
    assert Sf.shape == (len(time), len(freq))      
    analyzer.isPlot = True      
    analyzer.stft(df=40)      
    mo.output.append(plt.gcf())


@app.function
def test_WVDAnalysis(Sig):      
    analyzer = Analysis.WVDAnalysis(Sig=Sig)      
    # 测试 wvd 方法
    time, freq, Wf = analyzer.wvd(dt=0.001)      
    assert len(freq) == len(Sig)      
    assert time[1] - time[0] <= 0.01      
    assert Wf.shape == (len(time), len(freq))      
    analyzer.isPlot = True      
    analyzer.wvd(dt=0.01)      
    mo.output.append(plt.gcf())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. WaveletAnalysis模块
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
    # 测试 get_wavelet 方法      
    wavelettype = "B-Spline"      
    waveletparam = {"fc": 10, "fb": 5, "p": 3}      
    wavelets = analyzer.get_wavelets_discrete(      
        type=wavelettype,      
        param=waveletparam,      
        scale=scale,      
        N=1000,      
        normalized="能量",      
    )
    assert wavelets.shape == (50, 1000)      
    analyzer.get_wavelets_discrete(      
        type=wavelettype,      
        param=waveletparam,      
        isPlot=True,      
    )
    mo.output.append(plt.gcf())      
    # 测试 cwt 方法
    time, freq, Wf = analyzer.cwt(flow=10, nperoctave=10)      
    assert freq[0] == 0      
    assert freq[-1] < Sig.t_axis.fs / 2      
    assert len(time) == len(Sig)      
    assert Wf.shape == (len(time), len(freq))      
    analyzer.isPlot = True      
    analyzer.cwt(      
        flow=20, nperoctave=50, wavelet=wavelettype, param=waveletparam      
    )
    mo.output.append(plt.gcf())


@app.function
def test_DWTnalysis(Sig):      
    analyzer = Analysis.DWTAnalysis(Sig=Sig, title=f"{Sig.label}DWT分解结果",spectrum=False)      
    # 测试 dwt 方法
    wavelet = "db4"  
    SigList = analyzer.dwt(wavelet=wavelet, level=4)      
    assert len(SigList) == 5  
    assert sum([len(s) for s in SigList]) == len(Sig)
    analyzer.isPlot = True   
    analyzer.dwt(wavelet=wavelet, level=4)      
    mo.output.append(plt.gcf())


if __name__ == "__main__":
    app.run()
