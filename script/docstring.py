import ast
import os
import re


def _extract_all_list(tree: ast.Module):
    """从模块代码的AST中提取__all__列表"""
    all_names = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, (ast.Constant, ast.Str)):
                                all_names.append(elt.value)
                    # 假设__all__只定义一次，且在模块顶级
                    return all_names
    return all_names


def _extract_interface_docstrings(tree: ast.Module, all_names: list):
    """从模块代码的AST中提取__all__中指定接口的文档字符串首行"""
    functions_docs = {}
    classes_docs = {}

    # 遍历AST节点，查找函数和类定义
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in all_names:
                docstring = ast.get_docstring(node)
                summary = docstring.strip().split("\n")[0] if docstring else ""
                functions_docs[node.name] = summary
        elif isinstance(node, ast.ClassDef):
            if node.name in all_names:
                docstring = ast.get_docstring(node)
                summary = docstring.strip().split("\n")[0] if docstring else ""
                classes_docs[node.name] = summary
    return functions_docs, classes_docs


def update_module_docstring(fpath: str, title: str = "", summary: str = ""):
    """更新指定模块接口实现文件的文档字符串"""
    # 1. 删除现有的模块文档字符串
    fpath = os.path.abspath(fpath)
    if not os.path.exists(fpath):
        print(f"错误: 文件不存在 \n'{fpath}'")
        return ""
    with open(fpath, "r", encoding="utf-8") as f:
        original_lines = f.readlines()
    original_content = "".join(original_lines)
    docstring_pattern = re.compile(r'^\s*("""|\'\'\').*?\1\s*\n*', re.DOTALL)
    match = docstring_pattern.match(original_content)
    content_after_docstring = original_content
    if match:
        content_after_docstring = original_content[match.end() :]
        content_after_docstring = re.sub(r"^\n+", "", content_after_docstring)

    # 2. 解析文件内容以获取__all__和接口文档
    try:
        tree = ast.parse(content_after_docstring)
    except SyntaxError as e:
        print(f"错误: 解析文件失败: {e} \n'{fpath}'")
        return
    all_names = _extract_all_list(tree)
    functions_docs, classes_docs = _extract_interface_docstrings(tree, all_names)

    # 3. 构建新的模块文档字符串
    docstring_head = f"# {os.path.splitext(os.path.basename(fpath))[0]}"  # 文件名作为模块标题
    if title == "" and summary == "":
        # 检查文件开头是否有文档字符串，若有则提取首行作为docstring_head
        docstring_pattern = re.compile(r'^\s*("""|\'\'\')(.*?)(\1)', re.DOTALL)
        match = docstring_pattern.match(original_content)
        if match:
            doc_lines = match.group(2).strip().splitlines()
            if doc_lines:
                docstring_head = f"{doc_lines[0]}"
    else:
        if title == "":
            title = os.path.splitext(os.path.basename(fpath))[0]  # 文件名作为模块标题
        docstring_head = f"# {title}: {summary}"
    # 按照__all__的顺序组成接口文档字符串, 且class排在function之前
    function_lines = []
    class_lines = []
    for name in all_names:
        if name in functions_docs:
            function_lines.append(f"\n        - `{name}`: {functions_docs[name]}")
        elif name in classes_docs:
            class_lines.append(f"\n        - `{name}`: {classes_docs[name]}")
    docstring_interface_section = []
    if class_lines:
        docstring_interface_section.append("    - class:")
        docstring_interface_section.extend(class_lines)
    if function_lines:
        if class_lines:
            docstring_interface_section.append("\n")
        docstring_interface_section.append("    - function:")
        docstring_interface_section.extend(function_lines)
    # 组合最终文档字符串: 标题行 + 空行 + 接口列表
    docstring_interface_section = "".join(docstring_interface_section)
    docstring = '"""' + f"\n{docstring_head}\n\n" + docstring_interface_section + '\n"""\n\n'

    # 4. 将新文档字符串写入文件
    final_content = docstring + content_after_docstring
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"文档字符串已更新: '{fpath}'")

    # 5. 返回用于接口聚合文件的文档字符串(节标题比模块标题低一级)
    section_title = docstring_head.lstrip("#").strip()
    return f"## {section_title}\n{docstring_interface_section}"


def _resolve_imported_module_file(fpath: str, module_name: str, level: int):
    """根据导入语句信息，解析并返回被导入模块的文件路径"""
    base_dir = os.path.dirname(fpath)
    if level == 0:
        if module_name.startswith("signaltour."):
            module_name = module_name[len("signaltour.") :]
            base_dir = os.path.dirname(base_dir)
        else:
            return ""
    else:
        for _ in range(max(level - 1, 0)):
            base_dir = os.path.dirname(base_dir)
    module_parts = module_name.split(".") if module_name else []
    if not module_parts:
        return ""
    return os.path.normpath(os.path.join(base_dir, *module_parts) + ".py")


def _collect_imported_module_paths(tree: ast.Module, fpath: str):
    """收集AST中所有自定义模块的导入路径"""
    module_files = []
    seen = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            module_parts = node.module.split(".")
            if not module_parts:
                continue
            if not (module_parts[0].startswith("_") or module_parts[0] == "signaltour"):
                continue
            candidate = _resolve_imported_module_file(fpath, node.module, node.level)
            if candidate and os.path.exists(candidate) and candidate not in seen:
                seen.add(candidate)
                module_files.append(candidate)
    return module_files


def update_package_docstring(fpath: str, summary: str):
    """更新包接口导出文件的文档字符串"""
    # 1. 删除现有的模块文档字符串
    fpath = os.path.abspath(fpath)
    if not os.path.exists(fpath):
        print(f"错误: 文件不存在 \n'{fpath}'")
        return ""
    with open(fpath, "r", encoding="utf-8") as f:
        original_lines = f.readlines()
    original_content = "".join(original_lines)
    docstring_pattern = re.compile(r'^\s*("""|\'\'\').*?\1\s*\n*', re.DOTALL)
    match = docstring_pattern.match(original_content)
    content_after_docstring = original_content
    if match:
        content_after_docstring = original_content[match.end() :]
        content_after_docstring = re.sub(r"^\n+", "", content_after_docstring)

    # 2. 解析文件内容，收集所有导入的子模块路径
    try:
        tree = ast.parse(content_after_docstring)
    except SyntaxError as e:
        print(f"错误: 解析文件失败: {e} \n'{fpath}'")
        return ""
    module_files = _collect_imported_module_paths(tree, fpath)

    # 3. 聚合所有子模块的接口文档
    collected_sections = []
    for module_path in module_files:
        doc_section = update_module_docstring(module_path)
        if doc_section:
            collected_sections.append(doc_section)

    # 4. 构建聚合模块文档字符串
    aggregate_title = os.path.splitext(os.path.basename(fpath))[0]
    header = f"\n# {aggregate_title}" + (f": {summary}" if summary else "") + "\n\n"
    docstring = '"""' + header + "\n".join(collected_sections) + '\n"""\n'

    # 5. 写入文件顶部
    final_content = docstring + content_after_docstring
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"文档字符串已更新: '{fpath}'")
    return docstring


def _extract_docstring_first_line(fpath: str):
    """提取文件现有文档字符串的首行(不含引号), 若不存在则返回空字符串"""
    if not os.path.exists(fpath):
        return ""
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.match(r'\s*("""|\'\'\')(.*?)\1', content, re.DOTALL)
    if not match:
        return ""
    lines = match.group(2).strip().splitlines()
    return lines[0].strip() if lines else ""


def _split_docstring_summary(header_line: str):
    """从'# 名称: 说明'形式的标题行中提取说明部分, 无说明时返回空字符串"""
    head = header_line.lstrip("#").strip()
    if ": " in head:
        return head.split(": ", 1)[1].strip()
    return ""


def _strip_docstring_quotes(docstring: str):
    """去除文档字符串首尾的三引号及多余空行"""
    section = docstring.strip()
    if section.startswith('"""'):
        section = section[3:]
    if section.endswith('"""'):
        section = section[:-3]
    return section.strip("\n")


def _collect_imported_subpackages(tree: ast.Module, fpath: str):
    """收集AST中导入的公有子包(接口导出文件)路径, 跳过私有模块"""
    module_files = []
    seen = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module and "." in node.module:
            continue  # 仅处理包的直接子模块, 跳过深层模块
        names = [node.module] if node.module else [alias.name for alias in node.names]
        for name in names:
            if not name or name == "*" or name.startswith("_"):
                continue
            candidate = _resolve_imported_module_file(fpath, name, 1)
            if candidate and os.path.exists(candidate) and candidate not in seen:
                seen.add(candidate)
                module_files.append(candidate)
    return module_files


def update_init_docstring(fpath: str, first_line: str = ""):
    """更新包初始化文件的文档字符串, 汇总所有导入子包的接口文档

    Parameters
    ----------
    fpath : str
        包初始化文件 (__init__.py) 的路径
    first_line : str
        文档字符串首行的完整内容, 留空时沿用文件现有文档首行, 仍无则使用包名

    Returns
    -------
    str
        生成的初始化文件文档字符串
    """
    # 1. 删除现有的模块文档字符串
    fpath = os.path.abspath(fpath)
    if not os.path.exists(fpath):
        print(f"错误: 文件不存在 \n'{fpath}'")
        return ""
    with open(fpath, "r", encoding="utf-8") as f:
        original_content = f.read()
    docstring_pattern = re.compile(r'^\s*("""|\'\'\').*?\1\s*\n*', re.DOTALL)
    match = docstring_pattern.match(original_content)
    content_after_docstring = original_content
    if match:
        content_after_docstring = original_content[match.end() :]
        content_after_docstring = re.sub(r"^\n+", "", content_after_docstring)

    # 2. 解析文件内容，收集所有导入的公有子包路径
    try:
        tree = ast.parse(content_after_docstring)
    except SyntaxError as e:
        print(f"错误: 解析文件失败: {e} \n'{fpath}'")
        return ""
    module_files = _collect_imported_subpackages(tree, fpath)

    # 3. 依次生成各子包的聚合文档, 并汇总到初始化文件
    if first_line == "":
        first_line = _extract_docstring_first_line(fpath)
    if first_line == "":
        first_line = f"# {os.path.basename(os.path.dirname(fpath))}"  # 上一级目录名作为包标题
    collected_sections = []
    for module_path in module_files:
        sub_summary = _split_docstring_summary(_extract_docstring_first_line(module_path))
        doc_section = update_package_docstring(module_path, summary=sub_summary)
        doc_section = _strip_docstring_quotes(doc_section)
        if doc_section:
            collected_sections.append(doc_section)

    # 4. 构建初始化模块文档字符串, 各子包文档之间用'---'分隔
    docstring_parts = [f"\n{first_line}"]
    for section in collected_sections:
        docstring_parts.append("\n\n---\n\n")
        docstring_parts.append(section)
    docstring = '"""' + "".join(docstring_parts) + '\n"""\n\n'

    # 5. 写入文件顶部
    final_content = docstring + content_after_docstring
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"文档字符串已更新: '{fpath}'")
    return docstring
