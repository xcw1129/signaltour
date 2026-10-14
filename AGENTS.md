# AGENTS.md

本文件是 signaltour 仓库的协作开发指南，供 Codex/其他自动化开发代理在修改代码前阅读。

## 1. 事实来源与优先级

- 当前可运行实现以 `src/signaltour/` 为准，构建元数据以 `pyproject.toml` 为准。
- `README.md`、`doc/` 和 `.github/copilot-instructions.md` 存在滞后内容，不能直接作为接口事实来源；修改公共 API 时同步核对并更新相关文档。
- 当前版本号为 `1.3.0`，位置是 `pyproject.toml` 和 `src/signaltour/__init__.py`。修改版本时两处必须一致。
- 包名与分发名存在历史不一致：项目名为 `signaltour`，发布工作流使用 `signaltour-xcw`，README 的安装示例使用 `signaltour`。发布相关改动前先确认 PyPI 项目身份，不要自行假定。
- 不要回退工作区中已有的未提交改动。当前已知未提交内容包括 `_SpectrumAnalysis.py` 的修改和 `test/` 下的 marimo 会话文件。

## 2. 项目速览

- Python 包：`signaltour`，面向一维时序振荡数据的加载、仿真、预处理、分析与可视化。
- Python 版本：`>=3.11`；实际本地虚拟环境为 `.venv`，解释器是 Python 3.13。
- 核心依赖：NumPy、SciPy、Matplotlib、pandas、anytree、pyarrow。
- 源码布局：`src/` layout，安装包位于 `src/signaltour/`。
- 测试目录：`test/`；当前没有 pytest 测试用例，现有 `test/*.py` 是 marimo notebook，依赖外部数据集，不能当作 CI 单元测试。

## 3. 常用命令

在仓库根目录使用本地虚拟环境：

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import signaltour as st; print(st.__version__)"
.\.venv\Scripts\python.exe -m ruff check src
.\.venv\Scripts\python.exe -m ruff format --check src
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ty check src
```

说明：

- `ruff check src` 当前应通过；不要用全仓库 `ruff check` 把 `script/` 和 marimo 测试文件的历史问题当成源码回归。
- `pytest` 当前收集不到测试。新增功能时优先补充独立 pytest 用例，不要依赖 marimo notebook 做自动验证。
- `ty check src` 当前有 33 条基线诊断，包含真实接口错配；它可用于发现新问题，但不能要求“全绿”后才能提交。
- 绘图测试使用 `pattern="return"`，不要打开交互式窗口。无界面环境下应在导入绘图模块前设置 Matplotlib `Agg` 后端。

## 4. 代码结构

### 4.1 公共入口

- `src/signaltour/__init__.py`：顶层聚合入口，重新导出 `Signal`、`Plot`、`Analysis` 三个公共子包。
- `src/signaltour/_Signal.py`：Signal 公共入口，按顺序星号导入 `_Signal_Module` 的实现模块。
- `src/signaltour/_Plot.py`：Plot 公共入口，按顺序星号导入 `_Plot_Module` 的实现模块。
- `src/signaltour/_Analysis.py`：Analysis 公共入口，按顺序星号导入 `_Analysis_Module` 的实现模块。
- 用户代码推荐使用 `import signaltour as st`，然后调用 `st.Signal`、`st.SpectrumAnalysis`、`st.LinePlot` 等公开对象。

### 4.2 内部实现

- `_Signal_Module/_core.py`：`Axis`、`Series`、`t_Axis`、`f_Axis`、`Signal`、`Spectra`。
- `_Signal_Module/_SignalFilter.py`：FIR、IIR、中值滤波。
- `_Signal_Module/_SignalRead.py`：`LoadData`、`Files`、`Folder`、`Dataset` 以及 csv/txt/xlsx/mat 读取。
- `_Signal_Module/_SignalSample.py`：重采样、边界延拓、滑窗分段。
- `_Signal_Module/_SignalSimulate.py`：周期、冲击、调制仿真信号。
- `_Signal_Module/__methods__.py`：把滤波和采样函数挂载为 `Signal` 实例方法的 mixin。
- `_Analysis_Module/_core.py`：`BaseAnalysis` 与 `_plot` 装饰器。
- `_Analysis_Module/_SpectrumAnalysis.py`：窗函数、谱线检测、卷积、`SpectrumAnalysis`。
- `_Analysis_Module/_TimeFreqAnalysis.py`：STFT、WVD、CWT、DWT。
- `_Analysis_Module/_ModeAnalysis.py`：EMD、VMD 及包络、趋势等辅助函数。
- `_Analysis_Module/_StatsTrendAnalysis.py`：统计指标、分布和趋势。
- `_Plot_Module/_core.py`：任务队列、插件基类、Matplotlib 全局配置与 `show()`。
- `_Plot_Module/_LinePlot.py`：波形、频谱、趋势对比、轨迹图。
- `_Plot_Module/_ImagePlot.py`：时频图、热力图。
- `_Plot_Module/_PlotPlugin.py`：峰值标注和正负值掩码插件。
- `_Assist_Module/`：目前仅包含随包字体；历史文档提到的 `Dependencies.py`、`Decorators.py` 不存在。

### 4.3 公共 API 导出规则

- 每个实现模块在文件顶部定义 `__all__`，`script/docstring.py` 依赖该列表和接口 docstring 首行生成包级文档。
- 新增公共类或函数时，必须同时：
  1. 在实现模块定义对象并加入 `__all__`；
  2. 通过对应的 `_Signal.py`、`_Plot.py` 或 `_Analysis.py` 星号导出；
  3. 使用 NumPy 风格 docstring；
  4. 运行文档字符串生成流程或手动保持模块顶部接口摘要同步。
- 内部对象不要为了方便测试而从公共入口导出。测试可以显式导入带下划线的内部模块。
- 包入口的 `if __name__ == "__main__"` 代码块用于自动重写文档字符串，会直接修改文件；只在明确需要更新接口摘要时运行。

## 5. 数据模型约定

### 5.1 Axis

- `Axis` 只维护 `_N`、`_dx`、`_x0`，坐标数据按需生成；不要在子类中重复存储坐标数组。
- `N` 是只读属性，`T/F` 等会改变长度的 setter 已删除；`fs/dt/t0/df/f0` 等不改变长度的 setter 保留。
- `Axis.is_compatible(other)` 只比较 `_dx` 和 `unit`，不比较 `_x0`、长度或具体子类类型。
- `Axis.lim` 是半开区间边界 `(x0, x0 + N*dx)`，不是最后一个采样点；绘图和切片逻辑依赖这一语义。
- `t_Axis` 由 `fs`/`dt` 与 `N`/`T` 中恰好两个独立参数确定；`t0` 默认 0。
- `f_Axis` 由 `N`、`df`、`F` 中恰好两个参数确定。
- `to_pos_index` 支持物理索引字符串，例如 `"0.5 s"`；切片使用右对齐索引语义。

### 5.2 Series、Signal、Spectra

- `Series` 必须满足“一维数据、长度与坐标轴一致、不含 NaN”；这些约束由 `_check_data` 集中校验。
- `Series.data` 返回只读视图，公共代码通过 `.data` 读取；库内部需要高性能或原地写入时使用 `._data`，但要明确说明副作用。
- `Series` 的 `name` 表示物理量名称，`unit` 表示单位，`label` 表示实例标签。绘图标题和图例主要使用这些元信息。
- 数值运算通过 NumPy mixin 实现：`__array_ufunc__` 校验所有 `Series` 操作数的轴兼容性，并在形状和数值 dtype 兼容时通过 `template()` 保持类型与元信息。
- `__array_function__` 只把顶层 `Series` 参数替换为 `_data`，并直接返回 NumPy 原始结果，不自动包装为 `Signal`/`Spectra`。
- `Signal.t_axis` 是 `_axis` 的别名，`Spectra.f_axis` 同理。
- `Spectra.halfCut()` 是原地修改，会裁剪单边谱并把非直流幅值乘 2。调用前确认是否需要保留原对象。
- `Signal.to_Spectra()` 返回矩形窗、无零填充的 `Spectra`，供快速查看和绘图；需要更完整的谱分析时使用 `SpectrumAnalysis`。

### 5.3 Signal 实例方法扩展

- `Signal` 的滤波与采样方法由 `__methods__.py` 中的 `SignalFilter`、`SignalSample` mixin 提供。
- mixin 方法只做参数转发和延迟导入，具体实现位于对应的 `_SignalFilter.py`、`_SignalSample.py`。
- 新增实例方法时，不要直接在 `_core.py` 里堆实现；遵循“mixin 转发 + 独立实现模块 + `_Signal.py` 导出”的现有模式。

## 6. 数据读取约定

- `Files` 表示单一目录的静态文件注册表，不自动感知目录更新；需要重新扫描时重建或使用 `Dataset.refresh()`。
- `Dataset` 在初始化时扫描目录树，用 `Folder` 表示层级结构；字符串索引按文件夹名精确匹配。
- `LoadData.table` 是索引表，`_payload` 是数据内容；公共读取使用 `table`、迭代器和 `to_Signal()`，不要依赖 payload 的私有键格式。
- 读取状态统一为 `"完成"`、`"跳过"`、`"失败"`；单个文件失败不应中断批量读取。
- 支持文件类型是 `.csv`、`.txt`、`.xlsx`、`.mat`。读取参数通过 `Files.set_read_params(filetype, **kwargs)` 管理，并可按调用覆盖。
- `LoadData.merge(mode="hstack"|"vstack")` 是原地操作；合并后同一工况链的数据以 `(chain, "")` 索引。
- `LoadData.to_Signal(fs, name, unit)` 返回 `List[Dict[str, Signal]]`，每列一个信号，并会丢弃该列的 NaN。
- 批量读取性能路径涉及 `ThreadPoolExecutor` 和可选 pyarrow；修改时保持串行/并行结果顺序和状态表一致。

## 7. Analysis 约定

- 所有分析类继承 `BaseAnalysis`，初始化签名通常是 `(sig, isPlot=False, **kwargs)`，并将额外绘图参数保存到 `plot_kwargs`。
- 分析方法是数据计算的主体，返回值应是结构化对象：`Spectra`、`Signal`、`Series`、NumPy 数组元组或 `list[Signal]`。
- 需要自动绘图时使用 `@BaseAnalysis._plot(PlotFunc)`。如果方法返回 tuple，装饰器会展开后传给 `PlotFunc`；否则把单个返回值传入。
- 分析方法本身不要调用 `plt.show()`，不要直接创建 Figure；绘图由 `Plot` 层负责。
- `SpectrumAnalysis.ft()` 返回幅值密度谱；`cft()` 返回周期信号幅值谱；`psd()` 返回功率或功率密度谱。修改这些方法的语义时必须同步更新 docstring、示例和测试。
- `STFTAnalysis.stft()` 返回 `(time, freq, matrix)`；绘图时只返回单边幅值，计算时返回完整复数谱。不要混淆两种返回模式。
- EMD 返回的列表最后一项是残余模态；VMD 返回各时域模态列表。标签更新依赖 `_make_label()` 的 `"name: number"` 约定。

## 8. Plot 约定

- `BasePlot` 使用 FIFO 任务队列，任务字典格式为 `{"data", "kwargs", "function", "plugins"}`。
- 绘图方法（如 `waveform`、`spectrum`、`spectrogram`）只注册任务并返回 `self`，链式调用不应立即渲染。
- `show()` 支持 `pattern="plot"`、`"return"`、`"save"`。自动测试和 notebook 内嵌输出优先使用 `"return"`。
- `show(pattern="return")` 返回 `(figure, axs)`，其中 `axs` 始终是二维数组；单子图也要按二维数组处理。
- 每个子图的绘制异常会被捕获并写入日志，`show()` 本身不一定抛出。验证绘图改动时必须检查返回的 Figure/Axes 内容，而不是只判断调用未报错。
- 参数优先级固定为：方法默认值 < 实例 `self.kwargs` < 单次方法调用 `kwargs`。
- 单条曲线的样式通过 `plot={label: style_dict}` 传入；不要新增与既有参数优先级冲突的样式入口。
- 当前公开方法名是 `set_params()` 和 `add_plugin()`，不是旧文档中的 `set_params_to_task()` 或 `add_plugin_to_task()`。
- `ImagePlot.spectrogram()` 会就地归一化传入矩阵，并转置矩阵显示。调用方不要假设矩阵保持原值。

## 9. 编码与文档规范

- 使用 Ruff 配置：行宽 120、双引号、4 空格、Python 3.11 目标版本。
- 启用规则包括 `E`、`F`、`W`、`D`、`I`、`ANN201`。公共函数必须有返回类型注解，公共对象必须有 NumPy 风格 docstring。
- 异常信息使用“参数=值: 原因/期望”的中文风格；保持现有错误消息的可检索性。
- 模块内日志使用 `logging.getLogger(__name__)`，库代码不要用 `print`；`Dataset.info()` 和字体加载失败提示属于现有例外。
- 新增或修改接口时同步更新：
  - 实现模块顶部 `__all__` 与接口摘要；
  - 公共入口星号导出；
  - `README.md` 或 `doc/source/` 中的用户示例（若 API 已公开）；
  - 版本号相关位置（仅发布时）。
- 文档中的数学内容使用项目既有中文术语和 Markdown/MyST 写法；不要把未实现的理论或计划写成现有能力。

## 10. 测试策略

当前没有正式 pytest 测试。新增测试时：

- 在 `test/` 下创建 `test_*.py`，使用 pytest 原生断言。
- 优先使用 `_SignalSimulate.py` 的仿真信号和临时目录，不依赖 `F:\OneDrive\Database` 等外部数据。
- 对数据模型测试至少覆盖：长度/NaN 校验、物理索引、切片、运算后类型和元信息保持、`halfCut()` 副作用。
- 对分析测试至少覆盖：返回类型、轴长度、已知频率峰值、`isPlot=False/True` 不崩溃，以及 `isPlot=True` 时是否返回原始分析结果。
- 对绘图测试设置 `matplotlib.use("Agg")`，使用 `show(pattern="return")`，断言 `fig`、`axs` 和必要的线条/图像 Artist 数量。
- 对读取测试使用临时 csv/txt/xlsx；`.mat` 可用 `scipy.io.savemat` 生成，避免提交二进制样例。
- 修复回归时先补一个最小失败测试，再改实现。

## 11. 当前基线与已知问题

以下问题是当前仓库状态，不应被误当成设计规范。修改相关代码时优先修复并补测试：

1. `StatsTrendAnalysis.trend()` 调用了旧接口：
   - 现状：`segment(self.sig, segNum=..., pad_mode="edge")`；
   - 实际接口：`segment(sig, nseg=None, tperseg=None, overlap=0.5) -> List[Signal]`；
   - 实测会抛出 `TypeError: segment() got an unexpected keyword argument 'segNum'`。
2. `VMDAnalysis.vmd()` 调用了旧接口：
   - 现状：`pad(Sig=..., length=..., mode="mirror")`；
   - 实际接口：`pad(sig, pad_width, mode="constant")`，模式由 NumPy `np.pad` 支持；
   - 实测会抛出 `TypeError: pad() got an unexpected keyword argument 'Sig'`。
3. `ty check src` 有 33 条基线诊断，集中在返回类型、旧参数名、NumPy/Spectra 类型推断和 Matplotlib 重载；修复时优先处理被修改文件和真实运行时错误。
4. `README.md` 快速示例使用 `Analysis.Spectrum`，当前实际类名是 `SpectrumAnalysis`。
5. `doc/source/index.rst` 和 `doc/source/usage.rst` 使用不存在的 `Signal.from_array()` 和 `.run()`，文档版本仍为 `7.5.4`；当前包版本为 `1.3.0`。
6. `.github/copilot-instructions.md` 的路径和模块清单已过时，例如把入口写成 `signaltour/Signal.py`，并提到不存在的 `_Assist_Module/Dependencies.py`、`Decorators.py`。
7. `SignalSimulate.impulse()` 的冲击位置滑移仍使用全局 `np.random.randint`，`seed` 只控制噪声冲击，不能完整复现所有随机分量。
8. `_StatsTrendAnalysis.py` 的 `trend()` 返回类型与调用约定同时过期：它既使用了错误关键字，也把 `segment()` 当成返回 `(data, time)` 的旧函数。

## 12. 修改检查清单

完成任何代码改动前，按需执行：

```powershell
.\.venv\Scripts\python.exe -m ruff check src
.\.venv\Scripts\python.exe -m ruff format --check src
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ty check src
```

并人工确认：

- 是否只修改了与任务相关的模块；
- 是否保留用户已有未提交改动；
- 公共 API 是否已同步 `__all__`、入口导出和 docstring；
- 是否新增或更新了可自动运行的 pytest 测试；
- 绘图改动是否用 `pattern="return"` 验证了 Figure/Axes；
- 数据读取改动是否覆盖失败、跳过和空文件状态；
- 是否误把 `README.md` 或 `doc/` 的旧示例当成当前接口；
- 若修复了第 11 节中的已知问题，是否同步删除对应风险说明并补充测试。
