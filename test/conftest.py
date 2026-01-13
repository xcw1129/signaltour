import pytest
import numpy as np
from signaltour import Signal


@pytest.fixture
def Sig():
    """
    提供一个预定义的信号对象 Sig，包含周期分量、冲击分量和调制分量。
    """
    _Sig = Signal.Signal(
        Signal.t_Axis(fs=4000.0, T=1),
        name="位移",
        unit="μm",
        label="仿真多分量混叠信号",
    )
    # 周期谐波
    _Sig1 = Signal.periodic(
        fs=_Sig.t_axis.fs,
        T=_Sig.t_axis.T,
        CosParams=((70, 10, 94), (140, 5.2, 24), (210, 3, 12)),
        noise=5.0,
    )
    _Sig += _Sig1
    # 冲击序列
    _Sig2 = Signal.impulse(
        fs=_Sig.t_axis.fs,
        T=_Sig.t_axis.T,
        ImpParams=(1400, 20, 0.05, 20, 0.01),
        noiseParams=(5, 10),
    )
    _Sig += _Sig2

    # 快变调频
    def _AM(t):
        return (0.5 * np.sin(2 * np.pi * 10 * t) + 1.5) * 3

    def _FM(t):
        return 500 * t + 100 * np.sin(2 * np.pi * 2 * t)

    _Sig3 = Signal.modulation(fs=_Sig.t_axis.fs, T=_Sig.t_axis.T, fc=480.0, AM=_AM, FM=_FM)
    _Sig += _Sig3
    return _Sig
