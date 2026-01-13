import ast
import os
import re


def _extract_all_list(tree: ast.Module):
    # 从AST中提取__all__列表
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
    # 从AST中提取__all__中指定接口的文档字符串首行
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


def generate_module_docstring(fpath: str, title: str = "", summary: str = ""):
    """
    生成并更新指定Python文件的模块顶部文档字符串

    该函数会读取目标文件，删除任何现有的模块级文档字符串，
    然后根据文件的`__all__`列表和其中定义的公共接口的文档字符串，
    生成一个新的模块级文档字符串，并将其写入文件顶部。
    """
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
    docstring = ["\n", docstring_head, "\n\n---\n\n", "## 可用的接口\n\n"]
    # 按照__all__的顺序组成接口文档字符串
    function_lines = []
    class_lines = []
    for name in all_names:
        if name in functions_docs:
            function_lines.append(f"\n        - `{name}`: {functions_docs[name]}")
        elif name in classes_docs:
            class_lines.append(f"\n        - `{name}`: {classes_docs[name]}")
    docstring_interface_section = []
    if function_lines:
        docstring_interface_section.append("    - function:")
        docstring_interface_section.extend(function_lines)
    if class_lines:
        if function_lines:
            docstring_interface_section.append("\n")
        docstring_interface_section.append("    - class:")
        docstring_interface_section.extend(class_lines)
    # 组合最终文档字符串
    docstring.extend(docstring_interface_section)
    docstring = '"""' + "".join(docstring) + '\n"""\n\n'

    # 4. 将新文档字符串写入文件
    final_content = docstring + content_after_docstring
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"成功: 文档字符串已更新 \n'{fpath}'")

    # 5. 返回用于接口聚合文件的文档字符串
    simplified_docstring = ["\n##", docstring_head, "\n"] + docstring_interface_section
    return "".join(simplified_docstring)


def _resolve_imported_module_file(fpath: str, module_name: str, level: int):
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


def generate_aggregate_docstring(fpath: str, summary: str):
    """
    生成并更新聚合接口文件（如顶层接口文件）的模块文档字符串。

    该函数会读取目标文件，删除任何现有的模块级文档字符串，
    然后自动收集所有通过import导入的子模块的接口文档，
    并将聚合后的接口描述写入文件顶部。
    """
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
        doc_section = generate_module_docstring(module_path)
        if doc_section:
            collected_sections.append(doc_section)

    # 4. 构建聚合模块文档字符串
    aggregate_title = os.path.splitext(os.path.basename(fpath))[0]
    header = f"\n# {aggregate_title}: {summary}"
    docstring_parts = [header, "\n\n---\n\n", "## 可用的接口\n"]
    if collected_sections:
        docstring_parts.extend(collected_sections)
    docstring = '"""' + "".join(docstring_parts) + '\n"""\n'

    # 5. 写入文件顶部
    final_content = docstring + content_after_docstring
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"文档字符串已更新: '{fpath}'")
    return docstring
