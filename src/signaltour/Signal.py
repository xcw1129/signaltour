"""
# Signal: 信号数据读取、生成、封装和预处理等数据管理子包

---

## 可用的接口

### core
    - class:
        - `Axis`: 通用坐标轴类, 用于生成和管理一维顺序均匀采样坐标轴数据
        - `Series`: 通用一维序列数据类, 用于保存和处理坐标轴和对应序列数据
        - `t_Axis`: 时间坐标轴类
        - `f_Axis`: 频率坐标轴类
        - `Signal`: 一维时域信号类, 带有时间采样信息
        - `Spectra`: 一维频谱类, 带有频率采样信息
### SignalRead
    - function:
        - `set_logging_level`: 设置当前模块的日志显示级别
    - class:
        - `Files`: 数据文件批量管理类, 支持同根目录下单一格式文件的快速筛选与批量加载
        - `Folder`: 数据文件夹管理类, 支持快速预览和批量检索加载数据文件
        - `Dataset`: 数据集文件夹扫描与管理类, 支持自动识别层级结构并发现、加载数据文件
### SignalSimulate
    - function:
        - `periodic`: 生成仿真含噪准周期信号
        - `impulse`: 生成仿真冲击序列和噪声冲击复合信号
        - `modulation`: 生成仿真含噪调制信号
### SignalSampling
    - function:
        - `resample`: 对信号序列 Sig 进行任意时间段的重采样，支持下采样与上采样多种方式。
        - `pad`: 对信号对象进行边界延拓处理，支持镜像延拓和零填充方式
        - `slice`: 对信号进行滑窗跳步分段，首尾段自动延拓
### SignalFilt
    - function:
        - `filtFIR`: 基于有限冲击响应滤波器对信号进行各种类型滤波
        - `filtIIR`: 基于无限冲击响应滤波器对信号进行各种类型滤波
        - `filtMedian`: 基于中值滤波器对信号进行去噪处理
"""

# ruff: noqa: F403
# ruff: noqa: I001

from ._Signal_Module.core import *
from ._Signal_Module.SignalRead import *
from ._Signal_Module.SignalSimulate import *
from ._Signal_Module.SignalSample import *
from ._Signal_Module.SignalFilt import *

if __name__ == "__main__":
    from script.docstring import update_package_docstring

    update_package_docstring(
        __file__, summary="信号数据读取、生成、封装和预处理等数据管理子包"
    )
