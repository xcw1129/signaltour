from pathlib import Path
import sys

DOC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOC_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # 确保 signaltour 包可被 autodoc 导入

# -- 项目信息 ---------------------------------------------------------------
project = "signaltour"
author = "Xiong Chengwen"
copyright = "2025, " + project
release = "7.5.4"

# -- Sphinx 核心扩展 ---------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # 从 docstring 提取自动文档
    "sphinx.ext.autosummary",  # 构建 APIs 的摘要页
    "numpydoc",  # 兼容 numpy 风格注释
    "myst_parser",  # 解析 Markdown 文件
    "sphinx.ext.doctest",  # 执行 docstring 示例
    "sphinx.ext.intersphinx",  # 交叉引用其他库（Python/NumPy/Scipy）
    "sphinx.ext.mathjax",  # 渲染 LaTeX 数学公式
    "sphinx.ext.viewcode",  # 为源码生成链接
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

language = "zh_CN"  # 生成中文文档

templates_path = ["_templates"]
exclude_patterns = []

# -- autosummary 与 autodoc 行为 ------------------------------------------------
autosummary_generate = True  # 在 docs/source/automodules 中生成 stub
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": False,
    "special-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

# -- numpydoc 定制化 ----------------------------------------------------------
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False
numpydoc_xref_param_type = True
numpydoc_validation_checks = {"all"}  # 全面检查 docstring 是否合规
numpydoc_xref_aliases = {
    "array_like": ":term:`array_like`",
    "scalar": ":term:`scalar`",
    "ndarray": "~numpy.ndarray",
    "iterable": ":term:`iterable`",
    "callable": ":term:`callable`",
    "PathLike": "os.PathLike",
    "bool": ":py:class:`bool`",
    "int": ":py:class:`int`",
    "float": ":py:class:`float`",
    "str": ":py:class:`str`",
    "list": ":py:class:`list`",
    "tuple": ":py:class:`tuple`",
    "dict": ":py:class:`dict`",
    "set": ":py:class:`set`",
    "None": ":py:obj:`None`",
}

# -- 跨项目引用 ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML 外观与输出 ----------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}
html_static_path = ["_static"]
