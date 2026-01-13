"""
# SignalRead

---

## 可用的接口

    - function:
        - `set_logging_level`: 设置当前模块的日志显示级别
    - class:
        - `Files`: 数据文件批量管理类, 支持同根目录下单一格式文件的快速筛选与批量加载
        - `Folder`: 数据文件夹管理类, 支持快速预览和批量检索加载数据文件
        - `Dataset`: 数据集文件夹扫描与管理类, 支持自动识别层级结构并发现、加载数据文件
"""

__all__ = ["set_logging_level", "Files", "Folder", "Dataset"]

from .._Assist_Module.Dependencies import (
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
filesData: TypeAlias = pd.DataFrame | Dict[str, pd.DataFrame]


def set_logging_level(level: int | str) -> None:
    """
    设置当前模块的日志显示级别

    Parameters
    ----------
    level : int or str
        日志级别, 如 logging.INFO, 'DEBUG', 'WARNING' 等
    """
    if isinstance(level, str):
        level = level.upper()
    logger.setLevel(level)
    # 确保至少有一个处理器
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("> %(levelname)s - %(asctime)s\n\t%(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info(f"模块级别设置: level={level}, module=SignalRead")


# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# ------------------------------------------------------------------------#
# ----------------------------------------------------------------#
class Files:
    """
    数据文件批量管理类, 支持同根目录下单一格式文件的快速筛选与批量加载

    该 Files 类为静态管理, 不自动扫描更新, 仅作为 Folder 类的挂载属性使用

    Parameters
    ----------
    root : str
        根目录路径
    type : str
        文件类型, 支持: 'csv', 'txt', 'xlsx', 'mat'
    names : List[str], optional
        文件名列表, 若传入则仅验证这些文件, 否则扫描根目录下所有符合类型的文件
    records : List[Dict[str, Any]], optional
        文件元数据列表

    Attributes
    ----------
    rootpath : Path
        数据文件根目录路径
    filetype : str
        数据文件类型
    names : List[str]
        数据文件名列表
    filepaths : List[Path]
        数据文件路径列表

    Methods
    -------
    filter(pattern) -> Self
        根据文件名正则模式筛选文件
    query(expr) -> Self
        使用 DataFrame.query 语法筛选文件
    sorted() -> Self
        对数据文件进行排序
    load(merge=False, isParallel=False, parallelCores=None)
        批量载入有效数据文件为DataFrame
    preview(num=1, **kwargs)
        随机预览部分文件内容
    show_read_params(filetype)
        显示指定文件类型的读取参数
    set_read_params(filetype, **kwargs)
        设置指定类型文件的读取参数
    clean_read_params(filetype)
        清空指定类型文件的读取参数
    """

    def __init__(
        self, root: str, type: str, names: Optional[List[str]] = None, records: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        filetype = Files._check_filetype(type)
        rootpath = Path(root).resolve()
        logger.debug(f"Files初始化开始: root={rootpath}, type={filetype}")
        # ------------------------------------------------------------------------#
        # 验证文件有效性并收集基础信息
        if records is None:
            logger.debug("Files初始化模式: mode=run-scan")
            if not rootpath.exists() or not rootpath.is_dir():
                logger.error(f"Files初始化失败: root={rootpath}, reason=路径不存在或非文件夹")
                raise ValueError(f"root={rootpath}: 指定的路径不存在或不是文件夹")
            # 筛选有效数据文件名列表
            if names is None:
                # 若未传入 names 则扫描所有符合类型的文件
                valid_names: List[str] = [
                    f.name for f in rootpath.iterdir() if f.is_file() and f.suffix.lower() == filetype
                ]
            else:
                valid_names: List[str] = []
                for name in names:
                    fp = rootpath / name
                    if fp.exists() and fp.is_file() and fp.suffix.lower() == filetype:
                        valid_names.append(name)
                    else:
                        logger.warning(f"Files初始化异常: name={name}, reason=文件不存在或类型不匹配")
            # ----------------------------------------------------------------#
            # 对目标数据文件进行元数据收集
            records = []
            for fn in valid_names:
                fp = rootpath / fn
                stat = fp.stat()
                records.append(
                    {
                        "name": fp.name,
                        "size[MB]": stat.st_size,
                        "modifiedTime": stat.st_mtime,
                    }
                )
        else:
            logger.debug("Files初始化模式:  pre-stat")
        # ------------------------------------------------------------------------#
        # 构建核心注册表 (保持传入顺序)
        self._fileTable: pd.DataFrame = pd.DataFrame(records)
        if self._fileTable.empty:
            self._fileTable = pd.DataFrame({"name": [], "size[MB]": [], "modifiedTime": []})
            logger.warning(f"Files初始化异常: root={rootpath}, reason=未找到任何有效数据文件")
        self._fileTable["size[MB]"] = self._fileTable["size[MB]"] / (1024 * 1024)  # 转换为MB单位
        self._fileTable["modifiedTime"] = pd.to_datetime(self._fileTable["modifiedTime"], unit="s").round("s")
        self._fileTable.attrs["rootpath"] = rootpath
        self._fileTable.attrs["filetype"] = filetype
        if self._fileTable.columns.tolist() != Files._fileTableCols:
            logger.error(
                f"Files初始化失败: root={rootpath}, columns={self._fileTable.columns.tolist()}, reason=数据文件表列异常"
            )
            raise RuntimeError("数据文件表列异常, 无法完成 Files 对象初始化")
        logger.info(
            f"Files初始化完成: root={rootpath}, total={len(names) if names is not None else 'N/A'}, valid={len(self._fileTable)}"  # noqa: E501
        )

    _fileTableCols: List[str] = ["name", "size[MB]", "modifiedTime"]

    @property
    def rootpath(self) -> Path:
        """数据文件根目录路径"""
        return self._fileTable.attrs["rootpath"]

    @property
    def filetype(self) -> str:
        """数据文件类型"""
        return self._fileTable.attrs["filetype"]

    @property
    def names(self) -> List[str]:
        """数据文件名列表"""
        return self._fileTable["name"].tolist()

    @property
    def filepaths(self) -> List[Path]:
        """数据文件路径列表"""
        return [self.rootpath / name for name in self.names]

    # --------------------------------------------------------------------------------#
    # Python特性支持
    def __len__(self) -> int:
        """返回数据文件数量"""
        return len(self._fileTable)

    def __iter__(self):
        """迭代器, 遍历数据文件路径"""
        return iter(self._fileTable["filepath"])

    def _new_from_table(self, table: pd.DataFrame) -> Self:
        """从已有数据文件表创建新的Files实例并继承元数据"""
        new_obj = type(self).__new__(type(self))
        new_obj._fileTable: pd.DataFrame = table.reset_index(drop=True)
        new_obj._fileTable.attrs = self._fileTable.attrs.copy()
        if new_obj._fileTable.empty:
            new_obj._fileTable = pd.DataFrame({"name": [], "size[MB]": [], "modifiedTime": []})
        if new_obj._fileTable.columns.tolist() != Files._fileTableCols:
            raise RuntimeError("数据文件表列异常, 无法完成 Files 对象初始化")
        return new_obj

    def __getitem__(self, item) -> Self:
        """支持整数/切片/字符串/字符串列表索引, 返回符合条件的 Files 子对象"""
        # 整数与切片索引
        if isinstance(item, (int, slice)):
            idxs = [item] if isinstance(item, int) else item
            return self._new_from_table(self._fileTable.iloc[idxs])
        # 字符串与字符串列表索引
        elif isinstance(item, (str, list)):
            target_names = [item] if isinstance(item, str) else item
            mask = self._fileTable["name"].isin(target_names)
            if mask.any():
                target_table = self._fileTable[mask]
                return self._new_from_table(target_table)
            else:
                raise IndexError(f"{item}: 未找到指定名称的数据文件")
        else:
            raise IndexError("Files 索引仅支持整数、切片、字符串和字符串列表")

    def __repr__(self) -> str:
        size_total: float = self._fileTable["size[MB]"].sum()
        return f"Files(root=[{self.rootpath.name}/], count={len(self)}, type={self.filetype}, size={size_total:.2f}MB)"

    # --------------------------------------------------------------------------------#
    # 外部用户接口
    def filter(self, pattern: str) -> Self:
        """根据文件名正则模式筛选文件"""
        mask = self._fileTable["name"].str.contains(pattern, case=False, regex=True, na=False)
        return self._new_from_table(self._fileTable[mask])

    def query(self, expr: str) -> Self:
        """使用 DataFrame.query 语法进行高级文件筛选 (例如 'size[MB] > 1.0')"""
        return self._new_from_table(self._fileTable.query(expr))

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
            对于 'name' 列排序时是否应用自然排序算法

        Returns
        -------
        Self
            排序后的Files对象
        """

        def natural_sort_key(s):
            return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s))

        logger.debug(f"Files排序开始: root={self.rootpath}, by={by}, ascending={ascending}, natural={natural}")
        if self._fileTable.empty:
            logger.warning("Files排序终止: reason=当前Files对象为空")
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
            logger.error(
                f"Files排序失败: root={self.rootpath}, columns={self._fileTable.columns.tolist()}, reason=数据文件表列异常"  # noqa: E501
            )
            raise RuntimeError("数据文件表列异常, 无法完成排序操作")
        logger.info(f"Files排序完成: root={self.rootpath}, count={len(df)}")
        return self

    def load(
        self,
        merge: bool = True,
        mode: str = "hstack",
        isParallel: bool = False,
        parallelCores: Optional[int] = None,
        usePyarrow: bool = False,
    ) -> filesData | None:
        """
        批量加载文件为DataFrame or 字典

        Parameters
        ----------
        merge : bool, default: True
            是否将所有文件按列合并为单个DataFrame返回
        mode : str, default: 'hstack'
            合并模式, 'hstack'按列合并, 'vstack'按行合并 (仅当merge=True时有效)
        isParallel : bool, default: False
            是否并行读取
        parallelCores : int, optional
            并行读取时的线程数
        usePyarrow : bool, default: False
            是否启用 pyarrow 引擎加速文件读取(需安装 pyarrow 库)

        Returns
        -------
        filesData | None
            merge=True时为单个DataFrame, 否则为{"文件名": DataFrame}字典
        """
        start_time = time()
        # 设置读取引擎
        if usePyarrow:
            if util.find_spec("pyarrow") is not None:
                for params in Files._read_params.keys():
                    Files._read_params[params]["engine"] = "pyarrow"
                logger.debug(f"Files读取引擎设置成功: root={self.rootpath}, currentEngine=pyarrow")
            else:
                # 继续使用pandas默认C引擎
                logger.warning(
                    f"Files读取引擎设置无效: root={self.rootpath}, currentEngine=C, reason=当前环境未安装pyarrow库"
                )
        logger.info(f"Files加载开始: root={self.rootpath}, count={len(self)}, merge={merge}, parallel={isParallel}")
        # ------------------------------------------------------------------------#
        # 批量读取文件
        df_list: List[pd.DataFrame] = Files._read_batch(
            self.filepaths, Files._read_funcs[self.filetype], isParallel, parallelCores
        )
        # 结果检查与处理
        if len(df_list) == 0:
            logger.warning(f"Files加载终止: root={self.rootpath}, reason=未从文件中读取到有效数据")
            return None
        valid_count = sum(1 for df in df_list if not df.empty)
        logger.info(f"Files预加载完成: root={self.rootpath}, total={len(df_list)}, success={valid_count}")
        # ------------------------------------------------------------------------#
        # 合并结果
        if merge:
            # 组织为单个DataFrame返回
            logger.debug(f"Files合并开始: root={self.rootpath}, mode={mode}")
            if mode == "hstack":
                # 为避免列名冲突, 添加文件名前缀
                for df, fp in zip(df_list, self.filepaths):
                    if df.empty:
                        continue
                    prefix = fp.stem
                    df.columns = [f"{prefix}#{col}" for col in df.columns]
                axis = 1
            elif mode == "vstack":
                axis = 0
            try:
                # 执行合并
                all_df = pd.concat(df_list, axis=axis, ignore_index=True if axis == 0 else False)
                all_df = all_df.to_frame() if isinstance(all_df, pd.Series) else all_df
                logger.debug(f"Files合并完成: root={self.rootpath}, lines={len(all_df)}, columns={all_df.columns}")
            except Exception as e:
                logger.error(f"Files合并失败: root={self.rootpath}, error={e}")
                return None
        else:
            # ----------------------------------------------------------------#
            # 组织为字典返回
            all_df: Dict[str, pd.DataFrame] = {}
            for fp, df in zip(self.filepaths, df_list):  # 读取顺序为文件列表顺序
                if not df.empty:
                    all_df[fp.stem] = df
        consumed_time = time() - start_time
        logger.info(f"Files加载完成: root={self.rootpath}, time={consumed_time:.2f}s")
        return all_df

    def preview(self, num: int = 1, **kwargs) -> List[pd.DataFrame] | None:
        """
        使用指定读取参数, 随机加载数据文件供预览

        Parameters
        ----------
        num : int, default: 1
            预览文件数量

        Returns
        -------
        List[pd.DataFrame] | None
            预览的DataFrame列表, 若无有效文件则返回 None
        """
        if len(self) == 0:
            return None
        sample_size = min(num, len(self))
        sample_filepaths = random.sample(self.filepaths, sample_size)
        df_list: List[pd.DataFrame] = []
        for fp in sample_filepaths:
            df: pd.DataFrame = Files._read_funcs[self.filetype](fp, **kwargs)
            df_list.append(df)
        return df_list

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
        """显示指定文件类型的读取参数"""
        return Files._read_params[Files._check_filetype(filetype)]

    @staticmethod
    def set_read_params(filetype: str, **kwargs) -> None:
        """设置指定类型文件的读取参数"""
        Files._read_params[Files._check_filetype(filetype)].update(kwargs)

    @staticmethod
    def clean_read_params(filetype: str) -> None:
        """清空指定类型文件的读取参数"""
        Files._read_params[Files._check_filetype(filetype)] = {}

    # --------------------------------------------------------------------------------#
    # 数据文件读取内部方法
    @staticmethod
    def _read_batch(
        filepaths: List[Path],
        read_once_func: Callable[[Path], pd.DataFrame],
        isParallel: bool = False,
        parallelCores: Optional[int] = None,
    ) -> List[pd.DataFrame]:
        """通用批量读取文件方法"""
        if isParallel:
            cpu = os.cpu_count() or 1
            max_workers = parallelCores or min(cpu * 2, len(filepaths))
            logger.debug(f"并行读取开启: max_workers={max_workers}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                df_list = list(executor.map(read_once_func, filepaths))
        else:
            df_list: List[pd.DataFrame] = []
            for i, fp in enumerate(filepaths):
                df_list.append(read_once_func(fp))
                logger.debug(f"读取进度: {i + 1}/{len(filepaths)} -> name={fp.name}")
        return df_list

    @staticmethod
    def _read_csv_once(filepath: Path, **kwargs) -> pd.DataFrame:
        """读取单个CSV文件为DataFrame"""
        csv_read_params: Dict = Files._read_params[".csv"].copy()
        csv_read_params.update(kwargs)
        logger.debug(f"读取参数详情: type=csv, params={csv_read_params}")
        # 读取文件
        try:
            df: pd.DataFrame = pd.read_csv(filepath, **csv_read_params)
            if df.empty:
                logger.warning(f"读取异常: name={filepath.name}, status=内容为空")
            else:
                logger.debug(f"读取成功: name={filepath.name}, lines={len(df)}, columns={df.columns}")
        except Exception as e:
            logger.error(f"读取失败: name={filepath.name}, error={e}")
            return pd.DataFrame()
        return df

    @staticmethod
    def _read_txt_once(filepath: Path, **kwargs) -> pd.DataFrame:
        """读取单个TXT文件为DataFrame"""
        txt_read_params: Dict = Files._read_params[".txt"].copy()
        txt_read_params.update(kwargs)
        logger.debug(f"读取参数详情: type=txt, params={txt_read_params}")
        # 读取文件
        try:
            df: pd.DataFrame = pd.read_csv(filepath, **txt_read_params)
            if df.empty:
                logger.warning(f"读取异常: name={filepath.name}, status=内容为空")
            else:
                logger.debug(f"读取成功: name={filepath.name}, lines={len(df)}, columns={df.columns}")
        except Exception as e:
            logger.error(f"读取失败: name={filepath.name}, error={e}")
            return pd.DataFrame()
        return df

    @staticmethod
    def _read_xlsx_once(filepath: Path, **kwargs) -> pd.DataFrame:
        """读取单个XLSX文件为DataFrame"""
        xlsx_read_params: Dict = Files._read_params[".xlsx"].copy()
        xlsx_read_params.update(kwargs)
        logger.debug(f"读取参数详情: type=xlsx, params={xlsx_read_params}")
        # 读取文件
        try:
            df: pd.DataFrame = pd.read_excel(filepath, **xlsx_read_params)
            if df.empty:
                logger.warning(f"读取异常: name={filepath.name}, status=内容为空")
            else:
                logger.debug(f"读取成功: name={filepath.name}, lines={len(df)}, columns={df.columns}")
        except Exception as e:
            logger.error(f"读取失败: name={filepath.name}, error={e}")
            return pd.DataFrame()
        return df

    @staticmethod
    def _read_mat_once(filepath: Path, **kwargs) -> pd.DataFrame:
        """读取单个MAT文件为DataFrame, 自动识别数据列与元数据"""
        mat_read_params: Dict = Files._read_params[".mat"].copy()
        mat_read_params.update(kwargs)
        logger.debug(f"读取参数详情: type=mat, params={mat_read_params}")
        # 读取文件
        try:
            mat = loadmat(filepath)
            logger.debug(f"预读取成功: name={filepath.name}, vars={list(mat.keys())}")
        except Exception as e:
            logger.error(f"读取失败: name={filepath.name}, error={e}")
            return pd.DataFrame()
        # ------------------------------------------------------------------------#
        # 记录变量与元数据
        user_vars = {k: v for k, v in mat.items() if not k.startswith("__")}
        metadata = {}
        arr: Dict[str, np.ndarray] = {}
        for k, v in user_vars.items():
            data = np.asarray(v).squeeze()
            if data.ndim == 0:
                metadata[k] = data.item()
            elif data.ndim == 1:
                arr[k] = data
            else:
                continue
        if not arr:
            logger.warning("数据解析异常: stats=未找到数组类变量")
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
                logger.warning("数据解析异常: status=解析后数据为空")
            else:
                logger.debug(f"数据解析成功: vars={list(arr.keys())}, attrs={df.attrs}")
                logger.debug(f"读取成功: name={filepath.name}, lines={len(df)}, columns={df.columns}")
            return df
        except Exception as e:
            logger.error(f"数据解析失败: error={e}")
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
    数据文件夹管理类, 支持快速预览和批量检索加载数据文件

    该 Folder 类未实现扫描构建和数据文件发现功能, 仅作为 Dataset 类的基类使用

    Attributes
    ----------
    files : Files
        当前节点直接挂载的 Files 对象

    Methods
    -------
    info() -> None
        打印文件夹树形结构, 直观展示数据集节点内容
    loadAll(**kwargs) -> Dict[str, filesData] | None
        加载当前数据文件夹及所有子节点挂载的 Files 对象
    loadMatch(match, filter=None, query=None, **kwargs) -> Dict[str, filesData] | None
        搜索当前数据文件夹匹配子节点, 加载Files并对数据表格匹配筛选
    """

    # --------------------------------------------------------------------------------#
    # Python特性支持
    def __getitem__(self, item):
        """支持通过名称(str/list)或索引(int/slice)访问直接子文件夹节点"""
        # 1. 整数与切片索引 (基于子节点列表顺序)
        if isinstance(item, (int, slice)):
            return self.children[item]
        # 2. 字符串与字符串列表索引
        elif isinstance(item, (str, list)):
            target_names = [item] if isinstance(item, str) else item
            results = [child for child in self.children if child.name in target_names]
            if results:
                # 字符串索引返回单个节点, 列表索引返回列表
                return results[0] if isinstance(item, str) else results
            else:
                raise IndexError(f"Folder '{self.name}': 未找到指定名称的子节点 '{item}'")
        else:
            raise IndexError("Folder 索引仅支持整数、切片、字符串和字符串列表")

    # --------------------------------------------------------------------------------#
    # Files对象管理与加载
    @property
    def files(self) -> Files:
        """获取该节点直接挂载的 Files 对象"""
        return getattr(self, "_files")

    @files.setter
    def files(self, value: Files):
        """设置该节点直接挂载的 Files 对象"""
        self._files: Files = value

    @staticmethod
    def load_batch(listFiles: List[Files], **kwargs) -> Dict[str, filesData]:
        """批量加载多个 Files 对象"""
        dictfilesData: Dict[str, filesData] = {}
        for i, files in enumerate(listFiles):
            df = files.load(**kwargs)
            if df is None:
                continue
            dictfilesData[str(files.rootpath)] = df  # 使用根路径区分不同 Files 加载结果
            logger.debug(f"Files批量加载进度: {i + 1}/{len(listFiles)} -> Files={files}")
        return dictfilesData

    # --------------------------------------------------------------------------------#
    # 外部用户接口
    def info(self) -> None:
        """打印文件夹树形结构, 直观展示数据集节点内容"""
        print(f"[{self.path}]")
        for pre, _, node in anytree.RenderTree(self):
            tag = f" → <Files> x {len(node._files)}" if hasattr(node, "_files") else ""
            print(f"{pre}{node.name}{tag}")

    def loadAll(self, **kwargs) -> Dict[str, filesData] | None:
        """加载当前数据文件夹及所有子节点挂载的 Files 对象"""
        # 搜集
        listFiles: List[Files] = []
        for node in anytree.PreOrderIter(self):
            if hasattr(node, "_files"):
                if not node.is_leaf:
                    logger.warning(f"Folder加载异常: node={node.name}, reason=非叶子节点挂载Files对象")
                listFiles.append(node._files)
        if not listFiles:
            logger.warning(f"Folder加载终止: node={self.name}, reason=未找到Files对象")
            return None
        # ------------------------------------------------------------------------#
        # 加载
        logger.debug(f"Folder加载开始: node={self.name}, count={len(listFiles)}")
        dictfilesData = Folder.load_batch(listFiles, **kwargs)
        if not dictfilesData:
            logger.warning(f"Folder加载终止: node={self.name}, reason=未从任何Files中读取到有效数据")
            return None
        logger.info(f"Folder加载完成: node={self.name}, count={len(listFiles)}, success={len(dictfilesData)}")
        return dictfilesData

    def loadMatch(
        self,
        match: str,
        filter: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, filesData] | None:
        """搜索当前数据文件夹匹配子节点, 加载Files并对数据表格匹配筛选"""
        # ------------------------------------------------------------------------#
        # 1. 匹配子节点搜索
        logger.debug(f"Folder检索开始: node={self.name}, match={match}")
        # 解析搜索关键词
        patterns: List[str] = [p.strip() for p in re.split(r"[,，\s]+", match) if p.strip()]
        if not patterns:
            logger.warning(f"Folder检索加载终止: node={self.name}, reason=检索关键词为空")
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
            logger.warning(f"Folder检索加载终止: node={self.name}, match={match}, reason=未找到任何匹配节点")
            return None
        # ------------------------------------------------------------------------#
        # 2. 对所有匹配节点的Files文件表进行筛选操作
        logger.debug(f"Folder检索完成: node={self.name}, match={match}, found={len(matched_nodes)}")
        listFiles: List[Files] = []
        for node in matched_nodes:
            # 搜索该子树下所有挂载 Files 的节点 (使用 PreOrderIter 深度搜索子树)
            for sub_node in anytree.PreOrderIter(node):
                if hasattr(sub_node, "_files"):
                    matched_files: Files = sub_node._files
                    if filter:
                        matched_files = matched_files.filter(filter)
                    if query:
                        matched_files = matched_files.query(query)
                    if len(matched_files) > 0:
                        listFiles.append(matched_files)
        logger.debug(f"Folder筛选完成: node={self.name}, filter={filter}, query={query}, count={len(listFiles)}")
        # ------------------------------------------------------------------------#
        # 3. 对所有匹配节点的筛选后Files进行加载
        if not listFiles:
            logger.warning(f"Folder检索加载终止: node={self.name}, reason=筛选后无可加载文件")
            return None
        logger.debug(f"Folder加载开始: node={self.name}, count={len(listFiles)}")
        # 调用静态批量加载接口
        dictfilesData = Folder.load_batch(listFiles, **kwargs)
        if not dictfilesData:
            logger.warning(f"Folder检索加载终止: node={self.name}, reason=检索到的Files无任何数据")
            return None
        logger.info(
            f"Folder检索加载完成: node={self.name}, match={match}, filter={filter}, query={query}, count={len(dictfilesData)}"  # noqa: E501
        )
        return dictfilesData


# --------------------------------------------------------------------------------------------#
class Dataset(Folder):
    """
    数据集文件夹扫描与管理类, 支持自动识别层级结构并发现、加载数据文件

    Parameters
    ----------
    root : str
        数据集根目录路径
    type : str
        目标数据文件类型, 支持: '.csv', '.txt', '.xlsx', '.mat'

    Attributes
    ----------
    rootpath : Path
        数据集根目录路径
    filetype : str
        数据集数据文件类型
    files : Files
        当前节点直接挂载的 Files 对象, 若该节点无数据文件则无该属性, 调用直接报错


    Methods
    -------
    info() -> None
        打印数据集树形结构, 直观展示数据集节点内容
    loadAll(**kwargs) -> Dict[str, filesData] | None
        加载整个数据集所有节点挂载的 Files 对象
    loadMatch(match, filter=None, query=None, **kwargs) -> Dict[str, filesData] | None
        搜索数据集匹配节点, 加载Files并对数据表格匹配筛选
    refresh() -> Self
        刷新数据集结构, 重新扫描磁盘目录
    """

    def __init__(self, root: str, type: str) -> None:
        # 1. 基础信息处理
        self._rootpath = Path(root).resolve()
        self._filetype = Files._check_filetype(type)
        self._validnode: int = 0  # 挂载数据文件节点计数
        if not self._rootpath.exists() or not self._rootpath.is_dir():
            logger.error(f"Dataset初始化失败: root={self._rootpath}, reason=路径不存在或非文件夹")
            raise ValueError(f"root={self._rootpath}: 指定的路径不存在或不是文件夹")
        # 2. 初始化树根节点 (继承自 Folder/anytree.Node)
        super().__init__(name=self._rootpath.name)
        logger.debug(f"Dataset初始化开始: root={self._rootpath}, type={self._filetype}")
        # 3. 触发目录扫描
        self._scan_structure(self._rootpath, self)
        logger.info(f"Dataset初始化完成: root={self._rootpath}")

    @property
    def rootpath(self) -> Path:
        """数据集根目录路径"""
        return self._rootpath

    @property
    def filetype(self) -> str:
        """数据集数据文件类型"""
        return self._filetype

    # --------------------------------------------------------------------------------#
    # Python特性支持
    def __len__(self) -> int:
        """返回数据集数据节点数量"""
        return self._validnode

    def __repr__(self) -> str:
        return f"Dataset(root=[{self.rootpath.name}/], type={self.filetype}, count={self._validnode})"

    # --------------------------------------------------------------------------------#
    # 目录扫描与结构构建内部方法
    def _scan_structure(self, path: Path, node: Folder) -> None:
        """递归扫描目录并构建 Folder 树, 自动挂载到self节点下"""
        sub_paths: List[Path] = []
        target_records: List[Dict[str, Any]] = []
        # ------------------------------------------------------------------------#
        # 1. 扫描当前目录下的文件与子目录
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
                    if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(self._filetype):
                        # 获取缓存的 stat 信息
                        stat = entry.stat(follow_symlinks=False)
                        target_records.append(
                            {
                                "name": entry.name,
                                "size[MB]": stat.st_size,
                                "modifiedTime": stat.st_mtime,
                            }
                        )
                    # 处理目录: 收集子目录路径以待递归
                    elif entry.is_dir(follow_symlinks=False):
                        sub_paths.append(Path(entry.path))
        except Exception as e:
            logger.warning(f"节点扫描失败: path={path}, error={e}")
            return None
        # ------------------------------------------------------------------------#
        # 2. 挂载数据文件
        if target_records:
            self._validnode += 1  # 不直接含有数据文件的非叶子节点不会重复计数
            logger.debug(f"节点发现文件: node={node.name}, count={len(target_records)}")
            try:
                # 为该节点挂载 Files 对象, 传入预读取的 records
                node.files = Files(root=str(path), type=self._filetype, records=target_records)
                if not node.is_leaf:
                    logger.warning(f"节点扫描异常: path={path}, reason=非叶子节点发现Files对象")
            except Exception as e:
                logger.error(f"节点挂载文件失败: path={path}, error={e}")
        # ------------------------------------------------------------------------#
        # 3. 递归处理子目录
        for sub_path in sub_paths:
            # 创建子 Folder 节点并自动建立父子关系
            sub_node = Folder(name=sub_path.name, parent=node)
            self._scan_structure(sub_path, sub_node)

        logger.debug(f"节点扫描完成: path={path}, count={self._validnode}")

    # --------------------------------------------------------------------------------#
    # 外部用户接口
    def refresh(self) -> Self:
        """刷新数据集结构, 重新扫描磁盘目录"""
        # 清空现有子节点
        self.children: list[Folder] = []
        self._validnode = 0
        # 重新扫描
        self._scan_structure(self._rootpath, self)
        logger.info(f"Dataset刷新完成: root={self._rootpath}, count={self._validnode}")
        return self
