"""
# SignalRead: 数据读取模块, 提供数据文件批量管理、文件夹预览与数据集扫描加载等方法

    - class:
        - `Files`: 数据文件批量管理类, 支持单一目录下指定类型数据文件的快速筛选与批量加载
        - `Folder`: 数据文件夹管理类, 支持快速预览和批量检索、筛选和加载数据文件
        - `Dataset`: 数据集扫描与管理类, 支持自动识别层级结构并发现、加载数据文件, 支持嵌套键索引
"""

__all__ = ["Files", "Folder", "Dataset"]

from .._Assist_Module._Dependencies import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Path,
    Self,
    ThreadPoolExecutor,
    TypeAlias,
    anytree,
    loadmat,
    logging,
    np,
    os,
    pd,
    random,
    re,
    time,
    util,
)

# 初始化日志记录器
logger = logging.getLogger(__name__)
Filesdata: TypeAlias = pd.DataFrame | Dict[str, pd.DataFrame]


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class Files:
    """
    数据文件批量管理类, 支持单一目录下指定类型数据文件的快速筛选与批量加载

    该 Files 类为静态管理, 仅初始化时支持文件扫描, 不支持自动扫描更新

    Attributes
    ----------
    rootpath : Path
        数据文件目录路径
    filetype : str
        数据文件类型
    filenames : List[str]
        数据文件名列表
    filepaths : List[Path]
        数据文件路径列表

    Methods
    -------
    - filter(pattern) -> Self
        使用文件名正则模式筛选数据文件
    - query(expr) -> Self
        使用 pandas query 语法筛选数据文件
    - sorted(by='name', ascending=True, natural=True) -> Self
        对数据文件进行排序
    - load(merge=True, mode='hstack', isParallel=False, parallelNum=None, usePyarrow=False, **kwargs)
        批量加载数据文件
    - preview(num=1, **kwargs)
        使用指定读取参数, 随机加载数据文件
    - show_read_params(filetype)
        展示指定文件类型的读取参数
    - set_read_params(filetype, **kwargs)
        设置指定类型文件的读取参数
    - clean_read_params(filetype)
        清空指定类型文件的读取参数
    """

    def __init__(
        self, root: str, type: str, names: Optional[List[str]] = None, records: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        数据文件批量管理类, 支持单一目录下指定类型数据文件的快速筛选与批量加载

        该 Files 类为静态管理, 仅初始化时支持文件扫描, 不支持自动扫描更新

        Parameters
        ----------
        root : str
            目录路径
        type : str
            文件类型, 支持: 'csv', 'txt', 'xlsx', 'mat'
        names : List[str], optional
            文件名列表, 若传入则仅验证这些文件, 否则扫描目录下所有符合类型的文件
        records : List[Dict[str, Any]], optional
            文件元数据列表
        """
        filetype = Files._check_filetype(type)
        rootpath = Path(root).resolve()
        start_time = time()
        is_internal = records is not None  # Dataset 内部扫描路径, 不输出常规构造日志
        rejected = 0
        if not is_internal:
            mode = "scan" if names is None else "names"
            logger.info(f"Files初始化开始: root={rootpath}, type={filetype}, mode={mode}")
            if not rootpath.exists() or not rootpath.is_dir():
                raise ValueError(f"输入的目录路径不存在或不是文件夹: {rootpath}")
            # ----------------------------------------------------------------#
            # 筛选有效数据文件名列表
            if names is None:
                # 若未传入 names 则扫描所有符合类型的文件
                valid_names: List[str] = [
                    f.name for f in rootpath.iterdir() if f.is_file() and f.suffix.lower() == filetype
                ]
            else:
                # 若传入 names 则仅验证这些文件的有效性
                valid_names: List[str] = []
                for name in names:
                    fp = rootpath / name
                    if fp.exists() and fp.is_file() and fp.suffix.lower() == filetype:
                        valid_names.append(name)
                    else:
                        # 拆分无效原因, 便于用户排查传入的 names
                        if not fp.exists():
                            reason = "文件不存在"
                        elif not fp.is_file():
                            reason = "不是文件"
                        else:
                            reason = "类型不匹配"
                        rejected += 1
                        logger.warning(f"Files初始化跳过: root={rootpath}, name={name}, reason={reason}")
            # ----------------------------------------------------------------#
            # 对目标数据文件进行元数据收集
            records = []
            for fn in valid_names:
                fp = rootpath / fn
                stat = fp.stat()
                # 逐一记录文件的基础信息
                records.append(
                    {
                        "name": fp.name,
                        "size[MB]": stat.st_size,
                        "modifiedTime": stat.st_mtime,
                    }
                )
        else:
            mode = "records"
        # ------------------------------------------------------------------------#
        # 构建核心注册表 (保持传入顺序)
        self._fileTable: pd.DataFrame = pd.DataFrame(records)
        if self._fileTable.empty:  # 传入或扫描的数据文件注册表为空
            self._fileTable = pd.DataFrame(columns=Files._fileTableCols)
        # 处理表格数据并记录元数据属性
        self._fileTable["size[MB]"] = self._fileTable["size[MB]"] / (1024 * 1024)  # 转换为MB单位
        self._fileTable["modifiedTime"] = pd.to_datetime(self._fileTable["modifiedTime"], unit="s").dt.round("s")
        self._fileTable.attrs["rootpath"] = rootpath
        self._fileTable.attrs["filetype"] = filetype
        # 进行最终注册表验证
        if self._fileTable.columns.tolist() != Files._fileTableCols:
            logger.warning(f"Files初始化失败: root={rootpath}, reason=文件注册表列校验未通过")
            self._fileTable = pd.DataFrame(columns=Files._fileTableCols)
        # ------------------------------------------------------------------------#
        # 输出构造结果 (Dataset 内部扫描路径仅在空结果时提示)
        valid = len(self._fileTable)
        consumed_time = time() - start_time
        if valid == 0:
            logger.warning(
                f"Files初始化完成: root={rootpath}, mode={mode}, valid=0, elapsed={consumed_time:.2f}s, "
                f"reason=未发现有效数据文件. 请检查目录内容与目标文件类型"
            )
        elif not is_internal:
            rejected_msg = f", rejected={rejected}" if rejected else ""
            logger.info(
                f"Files初始化完成: root={rootpath}, mode={mode}, valid={valid}{rejected_msg}, "
                f"elapsed={consumed_time:.2f}s"
            )

    _fileTableCols: List[str] = ["name", "size[MB]", "modifiedTime"]

    # --------------------------------------------------------------------------------#
    # 动态可读属性
    @property
    def rootpath(self) -> Path:
        """数据文件根目录路径"""
        return self._fileTable.attrs["rootpath"]

    @property
    def filetype(self) -> str:
        """数据文件类型"""
        return self._fileTable.attrs["filetype"]

    @property
    def filenames(self) -> List[str]:
        """数据文件名列表"""
        return self._fileTable["name"].tolist()

    @property
    def filepaths(self) -> List[Path]:
        """数据文件路径列表"""
        return [self.rootpath / name for name in self.filenames]

    # --------------------------------------------------------------------------------#
    # Python特性支持
    def __len__(self) -> int:
        """返回数据文件数量"""
        return len(self._fileTable)

    def __iter__(self):
        """迭代器, 遍历数据文件路径"""
        return iter(self.filepaths)

    def _new_from_records(self, table: pd.DataFrame) -> Self:
        """从传入的数据文件注册表创建新的Files实例并继承元数据"""
        new_files = type(self).__new__(type(self))
        if table.empty:
            new_files._fileTable = pd.DataFrame(columns=Files._fileTableCols)
        else:
            new_files._fileTable = table.reset_index(drop=True)
        new_files._fileTable.attrs = self._fileTable.attrs.copy()
        return new_files

    def __getitem__(self, item) -> Self:
        """支持整数/切片/字符串/字符串列表索引, 返回子文件Files对象"""
        # 1. 整数与切片索引
        if isinstance(item, (int, slice)):
            # 统一转为列表或切片直接索引
            return self._new_from_records(
                self._fileTable.iloc[item] if isinstance(item, slice) else self._fileTable.iloc[[item]]
            )
        # ------------------------------------------------------------------------#
        # 2. 字符串与字符串列表索引
        elif isinstance(item, (str, list)):
            target_names = [item] if isinstance(item, str) else item
            mask = self._fileTable["name"].isin(target_names)
            if mask.any():
                target_table = self._fileTable[mask]
                return self._new_from_records(target_table)
            else:
                raise KeyError(f"{item}: 未找到指定名称的数据文件")
        else:
            raise KeyError("Files 索引仅支持整数、切片、字符串和字符串列表")

    def __repr__(self) -> str:
        size_total: float = self._fileTable["size[MB]"].sum()
        return f"Files(root=[{self.rootpath}], type={self.filetype}, count={len(self)}, size={size_total:.2f}MB)"

    # --------------------------------------------------------------------------------#
    # 外部用户方法
    def filter(self, pattern: str) -> Self:
        """使用文件名正则模式筛选数据文件"""
        mask = self._fileTable["name"].str.contains(pattern, case=False, regex=True, na=False)
        return self._new_from_records(self._fileTable[mask])

    def query(self, expr: str) -> Self:
        r"""
        使用 pandas query 语法筛选数据文件

        Parameters
        ----------
        expr : str
            符合 pandas query 语法的表达式

        Examples
        --------
        >>> files.query("`size[MB]` > 1.0")
        >>> files.query("modifiedTime > '2023-01-01'")
        >>> files.query("`size[MB]` > 1 and modifiedTime > '2023-01-01'")
        >>> files.query("`size[MB]` > @limit")
        >>> files.query("name.str.extract('(\\\\d+)', expand=False).astype('int') < 5")
        """
        return self._new_from_records(self._fileTable.query(expr))

    def sorted(self, by: str = "name", ascending: bool = True, natural: bool = True) -> Self:
        """
        对数据文件进行排序, 方便后续按序加载

        Parameters
        ----------
        by : str, default: 'name'
            排序参考列: 'name', 'size[MB]', 'modifiedTime'
        ascending : bool, default: True
            是否升序
        natural : bool, default: True
            name 列排序时是否应用自然排序算法

        Returns
        -------
        Self
            排序后的Files对象
        """

        def natural_sort_key(s):
            return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s))

        if self._fileTable.empty:
            return self
        df: pd.DataFrame = self._fileTable.copy()
        if by == "name" and natural:
            df["_sort_key"] = df["name"].apply(natural_sort_key)
            df = df.sort_values(by="_sort_key", ascending=ascending).drop(columns=["_sort_key"])
        else:
            df = df.sort_values(by=by, ascending=ascending)
        # 就地更新文件表
        self._fileTable = df.reset_index(drop=True)
        if self._fileTable.columns.tolist() != Files._fileTableCols:
            raise RuntimeError("数据文件表列异常, 无法完成排序操作")
        return self

    def load(
        self,
        merge: bool = True,
        mode: str = "hstack",
        isParallel: bool = False,
        parallelNum: Optional[int] = None,
        usePyarrow: bool = False,
        **kwargs,
    ) -> Filesdata | None:
        """
        批量加载数据文件

        Parameters
        ----------
        merge : bool, default: True
            是否合并结果
        mode : str, default: 'hstack'
            合并模式, 'hstack'列并排合并, 'vstack'列堆叠合并 (仅当merge=True时有效)
        isParallel : bool, default: False
            是否并行读取
        parallelNum : int, optional
            并行读取时的线程数
        usePyarrow : bool, default: False
            是否启用pyarrow加速(需安装pyarrow库)

        Returns
        -------
        Filesdata | None
            加载结果. 若merge=True则为单个DataFrame, 否则为文件名到DataFrame的字典. 若未读取到任何有效数据则返回 None
        """
        start_time = time()
        logger.info(f"Files加载开始: root={self.rootpath}, count={len(self)}")
        if usePyarrow:
            if util.find_spec("pyarrow") is not None:
                kwargs["engine"] = "pyarrow"
            else:
                logger.warning(
                    f"Files加载降级: root={self.rootpath}, reason=未安装pyarrow库. 如需加速可安装: pip install pyarrow"
                )
        present_read_params: Dict = Files._read_params[self.filetype].copy()
        present_read_params.update(kwargs)
        logger.info(f"读取开始: type={self.filetype}, params={present_read_params}")
        # ------------------------------------------------------------------------#
        # 批量读取文件
        Listdataframe: List[pd.DataFrame] = Files._read_batch(
            self.filepaths, lambda fp: Files._read_funcs[self.filetype](fp, **kwargs), isParallel, parallelNum
        )
        if len(Listdataframe) == 0:
            logger.warning(f"Files加载中止: root={self.rootpath}, reason=无待读取的数据文件")
            return None
        # ------------------------------------------------------------------------#
        # 合并加载结果
        if merge:
            # 组织为单个DataFrame返回
            if mode == "hstack":
                # 为避免列名冲突, 添加文件名前缀
                for df, fp in zip(Listdataframe, self.filepaths):
                    if df.empty:
                        continue
                    prefix = fp.stem
                    df.columns = [f"{prefix}#{col}" for col in df.columns]
                axis = 1
            elif mode == "vstack":
                axis = 0
            try:
                # 执行合并
                start_time_merge = time()
                filesdata = pd.concat(Listdataframe, axis=axis, ignore_index=True if axis == 0 else False)
                filesdata = filesdata.to_frame() if isinstance(filesdata, pd.Series) else filesdata
                consumed_time_merge = time() - start_time_merge
                logger.info(
                    f"合并完成: root={self.rootpath}, mode={mode}, "
                    f"rows={len(filesdata)}, cols={len(filesdata.columns)}, elapsed={consumed_time_merge:.2f}s"
                )
            except Exception:
                logger.warning(f"合并失败: root={self.rootpath}, mode={mode}", exc_info=True)
                return None
        else:
            # 组织为字典返回
            filesdata: Dict[str, pd.DataFrame] = {}
            for fp, df in zip(self.filepaths, Listdataframe):  # 读取顺序为文件列表顺序
                if not df.empty:
                    filesdata[fp.stem] = df
        consumed_time = time() - start_time
        done_count = sum(1 for df in Listdataframe if not df.empty)
        logger.info(
            f"Files加载完成: root={self.rootpath}, total={len(Listdataframe)}, done={done_count}, "
            f"empty={len(Listdataframe) - done_count}, elapsed={consumed_time:.2f}s"
        )
        return filesdata

    def preview(self, num: int = 1, **kwargs) -> List[pd.DataFrame] | None:
        r"""
        使用指定读取参数, 随机加载数据文件.
        方便快速确定合适的读取参数配置

        Parameters
        ----------
        num : int, default: 1
            随机加载的文件数量

        - read_csv:
        ```
        sep: str
            指定字段分隔符，例如 ',' 或 '\t'
        header: int, list, or None
            指定作为列名的行号，若无列名设为 None
        names: list or None
            列名列表，当 header=None 时非常有用
        index_col: int, str, or None
            指定作为行索引的列
        usecols: list
            仅读取指定的列（索引或名称），可大幅减少内存占用
        skiprows: int or list
            跳过文件开头的若干行或指定的行号
        skipfooter: int
            跳过文件末尾的若干行
        nrows: int
            仅读取文件前 N 行数据
        dtype: dict
            强制指定列的数据类型，如 {'ID': int}
        encoding: str
            指定文件编码格式，如 'utf-8' 或 'gbk'
        na_values: scalar, str, list-like, or dict
            指定哪些值应识别为 NaN
        ```

        Returns
        -------
        List[pd.DataFrame] | None
            随机加载结果
        """
        start_time = time()
        if len(self) == 0:
            logger.warning(f"Files预览中止: root={self.rootpath}, reason=无可预览文件")
            return None
        sample_size = min(num, len(self))
        logger.info(f"Files预览开始: root={self.rootpath}, count={len(self)}, num={sample_size}")
        preview_params: Dict = Files._read_params[self.filetype].copy()
        preview_params.update(kwargs)
        logger.info(f"读取开始: type={self.filetype}, params={preview_params}")
        sample_filepaths = random.sample(self.filepaths, sample_size)
        Listdataframe: List[pd.DataFrame] = []
        for fp in sample_filepaths:
            df: pd.DataFrame = Files._read_funcs[self.filetype](fp, **kwargs)
            Listdataframe.append(df)
        consumed_time = time() - start_time
        done_count = sum(1 for df in Listdataframe if not df.empty)
        logger.info(f"Files预览完成: root={self.rootpath}, done={done_count}, elapsed={consumed_time:.2f}s")
        return Listdataframe

    # --------------------------------------------------------------------------------#
    # 数据文件读取参数管理
    _read_params: Dict[str, Dict] = {
        ".csv": {},
        ".txt": {},
        ".xlsx": {},
        ".mat": {},
    }  # 数据读取全局参数

    @staticmethod
    def _check_filetype(filetype: str) -> str:
        """标准化文件类型扩展名"""
        legal: str = filetype.lower()
        if not legal.startswith("."):
            legal = "." + legal
        supported = list(Files._read_params.keys())
        if legal not in supported:
            raise ValueError(f"{filetype}: 不支持的文件类型, 仅支持: {supported}")
        return legal

    @staticmethod
    def show_read_params(filetype: str) -> Dict:
        """展示指定文件类型的读取参数"""
        return Files._read_params[Files._check_filetype(filetype)]

    @staticmethod
    def set_read_params(filetype: str, **kwargs) -> None:
        """设置指定类型文件的读取参数"""
        legal = Files._check_filetype(filetype)
        Files._read_params[legal].update(kwargs)
        logger.info(f"读取参数更新: type={legal}, action=set, params={Files._read_params[legal]}")

    @staticmethod
    def clean_read_params(filetype: str) -> None:
        """清空指定类型文件的读取参数"""
        legal = Files._check_filetype(filetype)
        Files._read_params[legal] = {}
        logger.info(f"读取参数更新: type={legal}, action=clear, params={{}}")

    # --------------------------------------------------------------------------------#
    # 数据文件读取内部方法
    @staticmethod
    def _read_batch(
        filepaths: List[Path],
        read_once_func: Callable[[Path], pd.DataFrame],
        isParallel: bool = False,
        parallelNum: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """批量数据文件读取方法"""
        if isParallel:
            cpu = os.cpu_count() or 1
            max_workers = parallelNum or min(cpu, len(filepaths))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                df_list = list(executor.map(read_once_func, filepaths))
        else:
            df_list: List[pd.DataFrame] = [read_once_func(fp) for fp in filepaths]
        return df_list

    @staticmethod
    def _read_csv_once(filepath: Path, **kwargs) -> pd.DataFrame:
        """读取单个CSV文件为DataFrame"""
        csv_read_params: Dict = Files._read_params[".csv"].copy()
        csv_read_params.update(kwargs)
        # 读取文件
        try:
            df: pd.DataFrame = pd.read_csv(filepath, **csv_read_params)
            if df.empty:
                logger.warning(f"读取跳过: name={filepath.name}, reason=文件为空")
            else:
                logger.info(f"读取完成: name={filepath.name}, rows={len(df)}, cols={len(df.columns)}")
        except Exception:
            logger.warning(f"读取失败: name={filepath.name}", exc_info=True)
            return pd.DataFrame()
        return df

    @staticmethod
    def _read_txt_once(filepath: Path, **kwargs) -> pd.DataFrame:
        """读取单个TXT文件为DataFrame"""
        txt_read_params: Dict = Files._read_params[".txt"].copy()
        txt_read_params.update(kwargs)
        # 读取文件
        try:
            df: pd.DataFrame = pd.read_csv(filepath, **txt_read_params)
            if df.empty:
                logger.warning(f"读取跳过: name={filepath.name}, reason=文件为空")
            else:
                logger.info(f"读取完成: name={filepath.name}, rows={len(df)}, cols={len(df.columns)}")
        except Exception:
            logger.warning(f"读取失败: name={filepath.name}", exc_info=True)
            return pd.DataFrame()
        return df

    @staticmethod
    def _read_xlsx_once(filepath: Path, **kwargs) -> pd.DataFrame:
        """读取单个XLSX文件为DataFrame"""
        xlsx_read_params: Dict = Files._read_params[".xlsx"].copy()
        xlsx_read_params.update(kwargs)
        # 读取文件
        try:
            df: pd.DataFrame = pd.read_excel(filepath, **xlsx_read_params)
            if df.empty:
                logger.warning(f"读取跳过: name={filepath.name}, reason=文件为空")
            else:
                logger.info(f"读取完成: name={filepath.name}, rows={len(df)}, cols={len(df.columns)}")
        except Exception:
            logger.warning(f"读取失败: name={filepath.name}", exc_info=True)
            return pd.DataFrame()
        return df

    @staticmethod
    def _read_mat_once(filepath: Path, **kwargs) -> pd.DataFrame:
        """读取单个MAT文件为DataFrame, 自动识别数据列与元数据"""
        mat_read_params: Dict = Files._read_params[".mat"].copy()
        mat_read_params.update(kwargs)
        # 读取文件
        try:
            mat = loadmat(filepath)
        except Exception:
            logger.warning(f"读取失败: name={filepath.name}", exc_info=True)
            return pd.DataFrame()
        # ------------------------------------------------------------------------#
        # 记录变量与元数据
        user_vars = {k: v for k, v in mat.items() if not k.startswith("__")}
        metadata = {}
        arr: Dict[str, np.ndarray] = {}
        skipped: List[str] = []
        for k, v in user_vars.items():
            data = np.asarray(v).squeeze()
            if data.ndim == 0:
                metadata[k] = data.item()
            elif data.ndim == 1:
                arr[k] = data
            else:
                skipped.append(k)  # 非一维变量无法作为数据列, 仅记录以便排查
        if not arr:
            logger.warning(f"读取跳过: name={filepath.name}, reason=未找到一维数组变量, vars={list(user_vars.keys())}")
            return pd.DataFrame()
        # ------------------------------------------------------------------------#
        # 构建 DataFrame 并记录元数据
        try:
            dfs_to_concat: List[pd.DataFrame] = []
            for k, v in arr.items():
                df_arr = pd.DataFrame({k: v})
                dfs_to_concat.append(df_arr)
            df = pd.concat(dfs_to_concat, axis=1) if dfs_to_concat else pd.DataFrame()
            # 记录元数据到 attrs
            df.attrs = metadata
            if df.empty:
                logger.warning(f"读取跳过: name={filepath.name}, reason=解析结果为空")
            else:
                log_msg = (
                    f"读取完成: name={filepath.name}, rows={len(df)}, cols={len(df.columns)}, "
                    f"vars={list(arr.keys())}, attrs={len(metadata)}"
                )
                if skipped:
                    log_msg += f", skipped={skipped}"
                logger.info(log_msg)
            return df
        except Exception:
            logger.warning(f"解析失败: name={filepath.name}", exc_info=True)
            return pd.DataFrame()

    _read_funcs: Dict[str, Callable[[Path], pd.DataFrame]] = {
        ".csv": _read_csv_once,
        ".txt": _read_txt_once,
        ".xlsx": _read_xlsx_once,
        ".mat": _read_mat_once,
    }  # 数据读取全局方法


# --------------------------------------------------------------------------------------------#
class Folder(anytree.Node):
    """
    数据文件夹管理类, 支持快速预览和批量检索、筛选和加载数据文件

    支持整数/切片/字符串/字符串列表索引, 返回子节点Folder对象, 多级索引嵌套实现工况链访问

    该 Folder 类未实现扫描构建和数据文件发现功能, 仅作为 Dataset 类的基类使用

    Attributes
    ----------
    rootpath : Path
        该Folder对应的系统路径
    files : Files
        该Folder直接挂载的Files对象
    allfiles : List[Files]
        该Folder挂载的所有Files对象
    allfolders : List[Folder]
        该Folder下所有子节点Folder
    stats : Dict
        该Folder当前状态, 包括有效叶子节点数, 数据文件总数, 总大小[MB]

    Methods
    -------
    - info() -> None
        打印数据集信息和文件夹结构
    - loadAll(**kwargs) -> Dict[str, Filesdata] | None
        加载当前数据文件夹及所有子节点挂载的 Files 对象
    - loadMatch(match, **kwargs) -> Dict[str, Filesdata] | None
        匹配筛选加载当前数据文件夹内及其所有子节点挂载的 Files 对象
    """

    # --------------------------------------------------------------------------------#
    # Python特性支持
    def __getitem__(self, item) -> Self | List[Self]:
        # 1. 整数与切片索引 (基于子节点列表顺序)
        if isinstance(item, (int, slice)):
            return self.children[item]
        # 2. 字符串与字符串列表索引
        elif isinstance(item, (str, list)):
            target_names = [item] if isinstance(item, str) else item
            valid_folders = [child for child in self.children if child.name in target_names]
            if valid_folders:
                # 字符串索引返回单个节点, 列表索引返回列表
                return valid_folders[0] if isinstance(item, str) else valid_folders
            else:
                raise KeyError(f"key={item}: 未找到对应名称的子节点")
        else:
            raise KeyError(f"key={item}: Folder对象索引仅支持整数、切片、字符串和字符串列表")

    # --------------------------------------------------------------------------------#
    # 动态可读属性
    @property
    def rootpath(self) -> Path:
        """该Folder对应的系统路径"""
        rootpath_base = self.root._rootpath
        for p in self.path[1:]:  # 跳过根节点, 拼接子节点路径
            rootpath_base = rootpath_base / p.name
        return rootpath_base

    @property
    def allfolders(self) -> List["Folder"]:
        """该Folder下所有子节点Folder"""
        return self.leaves

    @property
    def stats(self) -> Dict:
        """该Folder当前状态, 包括有效叶子节点数, 数据文件总数, 总大小[MB]"""
        allfiles = self.allfiles
        valid_node_count = len(allfiles)
        file_tables = [files._fileTable for files in allfiles]
        file_count = sum(len(table) for table in file_tables)
        size_total = float(sum(table["size[MB]"].sum() for table in file_tables))
        return {"valid_node_count": valid_node_count, "file_count": file_count, "size_total": size_total}

    # --------------------------------------------------------------------------------#
    # Files对象管理与加载
    @property
    def files(self) -> Files:
        """该Folder直接挂载的Files对象"""
        if hasattr(self, "_files"):
            return self._files
        else:
            raise AttributeError(f"Folder={self}: 该文件夹未发现直接包含的数据文件, 故无Files对象可用")

    @files.setter
    def files(self, value: Files):
        """该Folder直接挂载的Files对象"""
        self._files: Files = value

    @property
    def allfiles(self) -> List[Files]:
        """该Folder挂载的所有Files对象"""
        Listfiles: List[Files] = []
        for node in anytree.PreOrderIter(self):
            if hasattr(node, "_files"):
                Listfiles.append(node._files)
        return Listfiles

    # --------------------------------------------------------------------------------#
    # 外部用户方法
    def info(self) -> None:
        """打印数据集信息和文件夹结构"""
        if self.is_root:
            print(f"> {self}")
        else:
            print(f"> {self.root}\n> {self}")
        print("-" * 50)
        for pre, _, node in anytree.RenderTree(self):
            if hasattr(node, "_files"):
                tag = f": ★--{len(node._files)}"
            else:
                tag = ": ⦸" if node.is_leaf else ""
            print(f"{pre}{node.name}{tag}")
        print("★: 发现数据文件, ⦸: 未发现数据文件")
        print("-" * 50)
        node_count, file_count, size_total = self.stats.values()
        print(f"> Valid nodes: {node_count}")
        print(f"> Total files: {file_count}")
        print(f"> Total size: {size_total:.2f} MB")

    @staticmethod
    def _load_batch(Listfiles: List[Files], **kwargs) -> Dict[str, Filesdata]:
        Dictfilesdata: Dict[str, Filesdata] = {}
        for files in Listfiles:
            filesdata = files.load(**kwargs)  # 直接穿透传递读取参数
            if filesdata is None:
                continue
            Dictfilesdata[str(files.rootpath)] = filesdata  # 使用根路径区分不同 Files 加载结果
        return Dictfilesdata

    def loadAll(self, **kwargs) -> Dict[str, Filesdata] | None:
        """
        加载当前数据文件夹及所有子节点挂载的 Files 对象

        Parameters
        ----------
        merge : bool, default: True
            单个Files加载结果是否合并
        mode : str, default: 'hstack'
            合并模式, 'hstack'列并排合并, 'vstack'列堆叠合并 (仅当merge=True时有效)
        isParallel : bool, default: False
            单个Files加载是否并行读取
        parallelNum : int, optional
            并行读取时的线程数
        usePyarrow : bool, default: False
            单个Files加载是否启用 pyarrow 引擎加速文件读取(需安装 pyarrow 库)

        return
        ------
        Dict[str, Filesdata] | None
            汇总加载结果. 各个 Files 的根路径到其加载结果的字典. 若无任何有效加载结果则返回 None
        """
        # 搜集
        Listfiles = self.allfiles
        # ------------------------------------------------------------------------#
        # 加载
        start_time = time()
        logger.info(f"Folder加载开始: node={self.name}, count={len(Listfiles)}")
        Dictfilesdata = Folder._load_batch(Listfiles, **kwargs)
        if not Dictfilesdata:
            logger.warning(f"Folder加载中止: node={self.name}, reason=未从任何Files中读取到有效数据")
            return None
        consumed_time = time() - start_time
        logger.info(
            f"Folder加载完成: node={self.name}, count={len(Listfiles)}, done={len(Dictfilesdata)}, "
            f"elapsed={consumed_time:.2f}s"
        )
        return Dictfilesdata

    def loadMatch(
        self,
        match: str,
        filter: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Filesdata] | None:
        """
        匹配筛选加载当前数据文件夹内及其所有子节点挂载的 Files 对象.

        筛选方式包括Folder级别的工况链, Files级别的文件名和属性. 不对单个数据文件内容进行筛选

        Parameters
        ----------
        match : str
            Folder级筛选参数, 使用文件夹名关键词进行工况筛选, 例如 'testA, case1'
        filter : str, optional
            Files级筛选参数, 使用文件名正则模式进行数据文件筛选, 例如 '.*_test.*'
        query : str, optional
            Files级筛选参数, 使用文件属性pandas query语法进行数据文件筛选, 例如 '`size[MB]` > 1.0'
        merge : bool, default: True
            单个Files加载结果是否合并
        mode : str, default: 'hstack'
            合并模式, 'hstack'列并排合并, 'vstack'列堆叠合并 (仅当merge=True时有效)
        isParallel : bool, default: False
            单个Files加载是否并行读取
        parallelNum : int, optional
            并行读取时的线程数
        usePyarrow : bool, default: False
            单个Files加载是否启用 pyarrow 引擎加速文件读取(需安装 pyarrow 库)

        return
        ------
        Dict[str, Filesdata] | None
            汇总加载结果. 各个 Files 的根路径到其加载结果的字典. 若无任何有效加载结果则返回 None
        """
        start_time = time()
        # ------------------------------------------------------------------------#
        # 1. Folder级匹配筛选
        logger.info(f"Folder检索开始: node={self.name}, match={match}, filter={filter}, query={query}")
        # 解析搜索关键词
        patterns: List[str] = [p for p in re.split(r"[,，\s]+", match) if p]
        if not patterns:
            logger.warning(f"Folder检索中止: node={self.name}, match={match}, reason=检索关键词为空")
            return None
        matched_nodes: List[Folder] = []
        nodes_to_search: List[Folder] = [self]
        self_path_len = len(self.path)
        # 广度优先搜索子节点
        while nodes_to_search:
            node = nodes_to_search.pop(0)
            # 获取从 self 到当前 node 的工况名称链 (过滤掉 self 之前的祖先节点)
            chain_names = [n.name for n in node.path[self_path_len - 1 :]]
            # 校验工况链是否匹配所有关键词
            match_success = True
            for p in patterns:
                # 任一工况名称匹配该关键词即可
                if not any(re.search(p, name) for name in chain_names):
                    match_success = False
                    break
            if match_success:
                # 节点匹配成功, 子节点停止搜索
                matched_nodes.append(node)
            else:
                # 节点未匹配, 继续搜索下级子节点
                nodes_to_search.extend(list(node.children))
        if not matched_nodes:
            logger.warning(
                f"Folder检索中止: node={self.name}, match={match}, "
                f"reason=未找到任何匹配节点. 可检查 match 关键词拼写, 或用 info() 查看文件夹结构"
            )
            return None
        # ------------------------------------------------------------------------#
        # 2. Files级匹配筛选
        logger.info(f"检索完成: node={self.name}, match={match}, matched={len(matched_nodes)}")
        listFiles: List[Files] = []
        for node in matched_nodes:
            for sub_node in anytree.PreOrderIter(node):
                if hasattr(sub_node, "_files"):
                    matched_files: Files = sub_node._files
                    if filter:
                        matched_files = matched_files.filter(filter)
                    if query:
                        matched_files = matched_files.query(query)
                    if len(matched_files) > 0:
                        listFiles.append(matched_files)
        logger.info(f"筛选完成: node={self.name}, filter={filter}, query={query}, count={len(listFiles)}")
        # ------------------------------------------------------------------------#
        # 3. 对所有筛选后Files进行加载
        if not listFiles:
            logger.warning(
                f"Folder检索中止: node={self.name}, match={match}, filter={filter}, query={query}, "
                f"reason=筛选后无可加载文件. 可放宽 filter/query 条件"
            )
            return None
        Dictfilesdata = Folder._load_batch(listFiles, **kwargs)
        if not Dictfilesdata:
            logger.warning(f"Folder检索中止: node={self.name}, match={match}, reason=检索到的Files无任何数据")
            return None
        consumed_time = time() - start_time
        logger.info(
            f"Folder检索完成: node={self.name}, match={match}, found={len(matched_nodes)}, "
            f"done={len(Dictfilesdata)}, elapsed={consumed_time:.2f}s"
        )
        return Dictfilesdata


# --------------------------------------------------------------------------------------------#
class Dataset(Folder):
    """
    数据集扫描与管理类, 支持自动识别层级结构并发现、加载数据文件, 支持嵌套键索引

    支持整数/切片/字符串/字符串列表索引, 返回子节点Folder对象, 多级索引嵌套实现工况链访问

    Attributes
    ----------
    filetype : str
        数据集数据文件类型
    setname : str
        数据集名称
    rootpath : Path
       该Folder对应的系统路径
    files : Files
        该Folder直接挂载的Files对象
    allfiles : List[Files]
        该Folder挂载的所有Files对象
    allfolders : List[Folder]
        该Folder下所有子节点Folder
    stats : Dict
        该Folder当前状态, 包括有效叶子节点数, 数据文件总数, 总大小[MB]

    Methods
    -------
    - info() -> None
        打印数据集信息和文件夹结构
    - loadAll(**kwargs) -> Dict[str, Filesdata] | None
        加载整个数据集所有节点挂载的 Files 对象
    - loadMatch(match, filter=None, query=None, **kwargs) -> Dict[str, Filesdata] | None
        匹配筛选加载整个数据集所有节点挂载的 Files 对象
    - refresh() -> Self
        刷新数据集结构, 重新扫描磁盘目录
    """

    def __init__(self, root: str, type: str, name: str = "") -> None:
        """
        数据集扫描与管理类, 支持自动识别层级结构并发现、加载数据文件, 支持嵌套键索引

        支持整数/切片/字符串/字符串列表索引, 返回子节点Folder对象, 多级索引嵌套实现工况链访问

        Parameters
        ----------
        root : str
            根目录路径
        type : str
            目标文件类型, 支持: '.csv', '.txt', '.xlsx', '.mat'
        name : str, default: ''
            数据集名称
        """
        # 1. 基础信息处理
        self._rootpath = Path(root).resolve()
        self.filetype = Files._check_filetype(type)
        if not self._rootpath.exists() or not self._rootpath.is_dir():
            raise ValueError(f"root={self._rootpath}: 指定的路径不存在或不是文件夹")
        self.setname = name
        # 2. 初始化根节点
        super().__init__(name=self._rootpath.name)
        start_time = time()
        logger.info(f"Dataset初始化开始: root={self._rootpath}, type={self.filetype}")
        # 3. 全目录扫描
        self._scan_structure(self)
        consumed_time = time() - start_time
        node_count, file_count, size_total = self.stats.values()
        logger.info(
            f"Dataset初始化完成: root={self._rootpath}, nodes={node_count}, "
            f"files={file_count}, size={size_total:.2f}MB, elapsed={consumed_time:.2f}s"
        )

    # --------------------------------------------------------------------------------#
    # Python特性支持
    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.setname}, root=[{self._rootpath}], type={self.filetype}, nodes={len(self.allfiles)})"  # noqa: E501
        )

    # --------------------------------------------------------------------------------#
    # 目录扫描与结构构建内部方法
    def _scan_structure(self, node: Folder) -> None:
        path = node.rootpath
        # ------------------------------------------------------------------------#
        # 1. 扫描当前目录下的文件与子目录
        sub_folders: List[Path] = []
        find_records: List[Dict[str, Any]] = []
        try:
            with os.scandir(path) as it:
                # 扫描传入目录下的所有条目及其元数据
                entries = sorted(list(it), key=lambda e: e.name)
                # 遍历条目分别处理文件与子目录
                for entry in entries:
                    # 排除隐藏文件
                    if entry.name.startswith("."):
                        continue
                    # 处理文件: 收集目标类型文件的元数据
                    if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(self.filetype):
                        # 获取缓存的 stat 信息
                        stat = entry.stat(follow_symlinks=False)
                        find_records.append(
                            {
                                "name": entry.name,
                                "size[MB]": stat.st_size,
                                "modifiedTime": stat.st_mtime,
                            }
                        )  # 按照Files._fileTableCol的格式
                    # 处理目录: 收集子目录路径以待递归
                    elif entry.is_dir(follow_symlinks=False):
                        sub_folders.append(Path(entry.path))
        except Exception:
            logger.warning(f"节点扫描失败: path={path}", exc_info=True)
            return None
        # ------------------------------------------------------------------------#
        # 2. 挂载数据文件到传入扫描节点
        if find_records:
            try:
                node.files = Files(root=str(path), type=self.filetype, records=find_records)
            except Exception:
                logger.warning(f"节点挂载失败: path={path}", exc_info=True)
        logger.info(f"节点扫描完成: path={path}, nodes={len(sub_folders)}, files={len(find_records)}")
        # ------------------------------------------------------------------------#
        # 3. 递归扫描发现的子目录
        for folder in sub_folders:
            # 深度优先
            sub_node = Folder(name=folder.name, parent=node)
            self._scan_structure(sub_node)  # Dataset相关信息通过self传递, 无需额外参数

    # --------------------------------------------------------------------------------#
    # 外部用户接口
    def refresh(self) -> Self:
        """刷新数据集结构, 重新扫描磁盘目录"""
        start_time = time()
        logger.info(f"Dataset刷新开始: root={self._rootpath}")
        # 清空现有子节点
        self.children: list[Folder] = []
        # 重新扫描
        self._scan_structure(self)
        consumed_time = time() - start_time
        node_count, file_count, size_total = self.stats.values()
        logger.info(
            f"Dataset刷新完成: nodes={node_count}, files={file_count}, "
            f"size={size_total:.2f}MB, elapsed={consumed_time:.2f}s"
        )
        return self
