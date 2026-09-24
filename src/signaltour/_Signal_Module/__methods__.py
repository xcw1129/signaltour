from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:  # 仅用于类型注解, 避免与_core模块循环导入
    from ._core import Signal


class SignalFilter:
    def filtFIR(
        self: "Signal",
        cutoff: float | Tuple[float, float],
        order: int = 128,
        btype: str = "lowpass",
        window: str = "hamming",
        zero_phase: bool = False,
    ) -> "Signal":
        from ._SignalFilter import filtFIR

        return filtFIR(self, cutoff=cutoff, order=order, btype=btype, window=window, zero_phase=zero_phase)

    def filtIIR(
        self: "Signal",
        cutoff: float | Tuple[float, float],
        order: int = 4,
        btype: str = "lowpass",
        ftype: str = "butter",
        zero_phase: bool = False,
    ) -> "Signal":
        from ._SignalFilter import filtIIR

        return filtIIR(self, cutoff=cutoff, order=order, btype=btype, ftype=ftype, zero_phase=zero_phase)

    def filtMedian(self: "Signal", size: int = 5) -> "Signal":
        from ._SignalFilter import filtMedian

        return filtMedian(self, size=size)


class SignalSample:
    def resample(
        self: "Signal",
        type: str = "spacing",
        dt: Optional[float] = None,
        t0: Optional[float] = None,
        T: Optional[float] = None,
    ) -> "Signal":
        from ._SignalSample import resample

        return resample(self, type=type, dt=dt, t0=t0, T=T)

    def pad(self: "Signal", pad_width: int, mode: str = "constant") -> "Signal":
        from ._SignalSample import pad

        return pad(self, pad_width=pad_width, mode=mode)

    def segment(
        self: "Signal",
        nseg: Optional[int] = None,
        tperseg: Optional[float] = None,
        overlap: float = 0.5,
    ) -> List["Signal"]:
        from ._SignalSample import segment

        return segment(self, nseg=nseg, tperseg=tperseg, overlap=overlap)
