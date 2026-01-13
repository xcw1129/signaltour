# ruff: # noqa: D100
# ruff: noqa: F401
# ruff: noqa: I001
# PYTHON基础库
import re  # 正则表达式
import os  # 操作系统交互
import logging  # 日志记录打印
from pathlib import Path  # 文件路径封装
from typing import Dict, List, Tuple, Set, Callable, Optional, Self, Literal, Any  # 类型注解
from typing import Annotated, TypeAlias, TypeGuard  # 类型检查增强
from typing import get_origin, get_args  # 类型注解工具
from collections import deque  # 双端队列数据
from functools import wraps  # 函数装饰器
import inspect  # 函数检查
from copy import copy as shallowcopy  # 浅拷贝
from copy import deepcopy  # 深拷贝
from concurrent.futures import ThreadPoolExecutor  # 多线程执行
from importlib import resources  # 资源管理
from importlib import util  # 模块导入工具
from time import time  # 时间计时
import random  # 随机操作

# 向量数值计算库
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin  # ndarray运算符重载基类

# 高级数学分析库
from scipy import signal  # 信号处理包
from scipy import fft  # 快速傅里叶变换包
from scipy import linalg  # 线性代数包
from scipy import stats  # 统计分析包
from scipy import interpolate  # 插值分析包
from scipy.io import loadmat  # MAT文件读取

# 表格数据处理库
import pandas as pd

# 可视化绘图库
import matplotlib.pyplot as plt
from cycler import cycler  # 循环颜色
from matplotlib import animation  # 动画绘图
from matplotlib import font_manager  # 字体管理
from matplotlib import ticker  # 坐标轴刻度管理

# 系统目录树结构管理
import anytree

# 数学常数
FLOAT_EPS: float = float(np.finfo(dtype=float).eps)  # 机器精度
PI: float = np.pi  # 圆周率
