"""
Utilitas pembuatan kode untuk Darwin-Gödel Machine.
"""

import os
import re
import ast
import inspect
from typing import Dict, Any, List, Optional, Callable, Union, Tuple


def generate_code(spec: Dict[str, Any]) -> str:
    """
    Hasilkan kode berdasarkan spesifikasi.
    
    Args:
        spec: Spesifikasi kode
        
    Returns:
        Kode yang dihasilkan
    """
    code_type = spec.get("type", "function")
    
    if code_type == "function":
        return generate_function_code(spec)
    elif code_type == "class":
        return generate_class_code(spec)
    elif code_type == "module":
        return generate_module_code(spec)
    elif code_type == "agent":
        return generate_agent_code(spec)
    else:
        return f"# Unknown code type: {code_type}"


def generate_function_code(spec: Dict[str, Any]) -> str:
    """
    Hasilkan kode fungsi.
    
    Args:
        spec: Spesifikasi fungsi
        
    Returns:
        Kode fungsi
    """
    name = spec.get("name", "unnamed_function")
    params = spec.get("params", [])
    return_type = spec.get("return_type", "Any")
    docstring = spec.get("docstring", "")
    body = spec.get("body", ["pass"])
    
    # Buat parameter
    param_str = ""
    for i, param in enumerate(params):
        param_name = param.get("name", f"param_{i}")
        param_type = param.get("type", "Any")
        param_default = param.get("default")
        
        if param_default is not None:
            if isinstance(param_default, str):
                param_str += f", {param_name}: {param_type} = \"{param_default}\""
            else:
                param_str += f", {param_name}: {param_type} = {param_default}"
        else:
            param_str += f", {param_name}: {param_type}"
    
    # Hapus koma awal jika ada parameter
    if param_str:
        param_str = param_str[2:]
    
    # Buat docstring
    if docstring:
        docstring_str = f'    """\n    {docstring}\n    """\n'
    else:
        docstring_str = ""
    
    # Buat badan fungsi
    body_str = "\n".join(f"    {line}" for line in body)
    
    # Gabungkan semua
    return f"def {name}({param_str}) -> {return_type}:\n{docstring_str}{body_str}"


def generate_class_code(spec: Dict[str, Any]) -> str:
    """
    Hasilkan kode kelas.
    
    Args:
        spec: Spesifikasi kelas
        
    Returns:
        Kode kelas
    """
    name = spec.get("name", "UnnamedClass")
    base_classes = spec.get("base_classes", [])
    docstring = spec.get("docstring", "")
    attributes = spec.get("attributes", [])
    methods = spec.get("methods", [])
    
    # Buat string kelas dasar
    base_str = ", ".join(base_classes) if base_classes else ""
    
    # Buat docstring
    if docstring:
        docstring_str = f'    """\n    {docstring}\n    """\n'
    else:
        docstring_str = ""
    
    # Buat atribut kelas
    attr_str = ""
    for attr in attributes:
        attr_name = attr.get("name", "unnamed_attr")
        attr_type = attr.get("type", "Any")
        attr_value = attr.get("value")
        
        if attr_value is not None:
            if isinstance(attr_value, str):
                attr_str += f"    {attr_name}: {attr_type} = \"{attr_value}\"\n"
            else:
                attr_str += f"    {attr_name}: {attr_type} = {attr_value}\n"
        else:
            attr_str += f"    {attr_name}: {attr_type}\n"
    
    if attr_str:
        attr_str += "\n"
    
    # Buat metode
    method_str = ""
    for method in methods:
        method_code = generate_function_code(method)
        # Indentasi metode
        method_code = "    " + method_code.replace("\n", "\n    ")
        method_str += method_code + "\n\n"
    
    # Gabungkan semua
    class_def = f"class {name}"
    if base_str:
        class_def += f"({base_str})"
    
    if not (docstring_str or attr_str or method_str):
        # Kelas kosong
        return f"{class_def}:\n    pass"
    
    return f"{class_def}:\n{docstring_str}{attr_str}{method_str}"


def generate_module_code(spec: Dict[str, Any]) -> str:
    """
    Hasilkan kode modul.
    
    Args:
        spec: Spesifikasi modul
        
    Returns:
        Kode modul
    """
    docstring = spec.get("docstring", "")
    imports = spec.get("imports", [])
    functions = spec.get("functions", [])
    classes = spec.get("classes", [])
    
    # Buat docstring
    if docstring:
        docstring_str = f'"""\n{docstring}\n"""\n\n'
    else:
        docstring_str = ""
    
    # Buat impor
    import_str = ""
    for imp in imports:
        if "from" in imp:
            import_str += f"from {imp['from']} import {', '.join(imp['import'])}\n"
        else:
            import_str += f"import {', '.join(imp['import'])}\n"
    
    if import_str:
        import_str += "\n\n"
    
    # Buat fungsi
    function_str = ""
    for func in functions:
        function_str += generate_function_code(func) + "\n\n"
    
    # Buat kelas
    class_str = ""
    for cls in classes:
        class_str += generate_class_code(cls) + "\n\n"
    
    # Gabungkan semua
    return f"{docstring_str}{import_str}{function_str}{class_str}"


def generate_agent_code(spec: Dict[str, Any]) -> str:
    """
    Hasilkan kode agen.
    
    Args:
        spec: Spesifikasi agen
        
    Returns:
        Kode agen
    """
    agent_name = spec.get("name", "CustomAgent")
    base_class = spec.get("base_class", "BaseAgent")
    description = spec.get("description", "Custom agent for Darwin-Gödel Machine.")
    params = spec.get("params", [])
    methods = spec.get("methods", [])
    
    # Buat spesifikasi modul
    module_spec = {
        "docstring": description,
        "imports": [
            {"import": ["random"]},
            {"from": "typing", "import": ["Dict", "Any", "List", "Optional", "Callable"]},
            {"from": f"simple_dgm.agents.{base_class.lower()}", "import": [base_class]},
            {"from": "simple_dgm.agents.base_agent", "import": ["Tool"]}
        ],
        "classes": [
            {
                "name": agent_name,
                "base_classes": [base_class],
                "docstring": description,
                "methods": [
                    # Metode __init__
                    {
                        "name": "__init__",
                        "params": params + [
                            {"name": "tools", "type": "Optional[List[Tool]]", "default": "None"},
                            {"name": "memory_capacity", "type": "int", "default": 10},
                            {"name": "learning_rate", "type": "float", "default": 0.01},
                            {"name": "exploration_rate", "type": "float", "default": 0.1}
                        ],
                        "return_type": "None",
                        "docstring": f"Inisialisasi {agent_name}.",
                        "body": [
                            f"super().__init__(tools, memory_capacity, learning_rate, exploration_rate)",
                            ""
                        ] + [f"self.{param['name']} = {param['name']}" for param in params] + [
                            "",
                            "# Tambahkan alat khusus",
                            "self._add_custom_tools()",
                            "",
                            "# Tambahkan metadata khusus",
                            "self.metadata.update({",
                            f'    "type": "{agent_name}",'
                        ] + [f'    "{param["name"]}": self.{param["name"]},' for param in params] + [
                            "})"
                        ]
                    },
                    # Metode _add_custom_tools
                    {
                        "name": "_add_custom_tools",
                        "params": [],
                        "return_type": "None",
                        "docstring": "Tambahkan alat khusus untuk agen.",
                        "body": [
                            "# Tambahkan alat khusus di sini",
                            "pass"
                        ]
                    },
                    # Metode mutate
                    {
                        "name": "mutate",
                        "params": [],
                        "return_type": "None",
                        "docstring": f"Mutasi {agent_name}.",
                        "body": [
                            "super().mutate()  # Panggil mutasi dasar",
                            "",
                            "# Mutasi parameter khusus"
                        ] + [
                            f"self.{param['name']} *= random.uniform(0.8, 1.2)" if param.get("type") == "float" else
                            f"self.{param['name']} = int(self.{param['name']} * random.uniform(0.8, 1.2))" if param.get("type") == "int" else
                            f"if random.random() < 0.3:\n    self.{param['name']} = random.choice({param.get('options', [])})" if param.get("type") == "str" and "options" in param else
                            f"# No mutation for {param['name']}"
                            for param in params
                        ] + [
                            "",
                            "# Update metadata",
                            "self.metadata.update({",
                        ] + [f'    "{param["name"]}": self.{param["name"]},' for param in params] + [
                            "})"
                        ]
                    },
                    # Metode to_dict
                    {
                        "name": "to_dict",
                        "params": [],
                        "return_type": "Dict[str, Any]",
                        "docstring": f"Konversi {agent_name} ke dictionary.",
                        "body": [
                            "data = super().to_dict()",
                            "data.update({",
                        ] + [f'    "{param["name"]}": self.{param["name"]},' for param in params] + [
                            "})",
                            "return data"
                        ]
                    },
                    # Metode from_dict
                    {
                        "name": "from_dict",
                        "params": [
                            {"name": "cls", "type": "type"},
                            {"name": "data", "type": "Dict[str, Any]"},
                            {"name": "function_map", "type": "Dict[str, Callable]"}
                        ],
                        "return_type": f"'{agent_name}'",
                        "docstring": f"Buat {agent_name} dari dictionary.",
                        "body": [
                            "agent = super().from_dict(data, function_map)",
                        ] + [f'agent.{param["name"]} = data["{param["name"]}"]' for param in params] + [
                            "return agent"
                        ]
                    }
                ] + methods
            }
        ]
    }
    
    # Hasilkan kode modul
    return generate_module_code(module_spec)


def parse_code(code: str) -> Dict[str, Any]:
    """
    Parse kode untuk mendapatkan struktur.
    
    Args:
        code: Kode yang akan di-parse
        
    Returns:
        Struktur kode
    """
    try:
        tree = ast.parse(code)
        
        # Inisialisasi hasil
        result = {
            "imports": [],
            "functions": [],
            "classes": []
        }
        
        # Parse docstring modul
        if (len(tree.body) > 0 and 
            isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Str)):
            result["docstring"] = tree.body[0].value.s
        
        # Parse node
        for node in tree.body:
            if isinstance(node, ast.Import):
                # Import
                for name in node.names:
                    result["imports"].append({
                        "import": [name.name]
                    })
            
            elif isinstance(node, ast.ImportFrom):
                # Import from
                imports = [name.name for name in node.names]
                result["imports"].append({
                    "from": node.module,
                    "import": imports
                })
            
            elif isinstance(node, ast.FunctionDef):
                # Fungsi
                result["functions"].append(parse_function(node))
            
            elif isinstance(node, ast.ClassDef):
                # Kelas
                result["classes"].append(parse_class(node))
        
        return result
    
    except SyntaxError:
        return {"error": "Invalid syntax"}


def parse_function(node: ast.FunctionDef) -> Dict[str, Any]:
    """
    Parse definisi fungsi.
    
    Args:
        node: Node AST fungsi
        
    Returns:
        Struktur fungsi
    """
    # Inisialisasi hasil
    result = {
        "name": node.name,
        "params": [],
        "return_type": "Any",
        "body": []
    }
    
    # Parse docstring
    if (len(node.body) > 0 and 
        isinstance(node.body[0], ast.Expr) and 
        isinstance(node.body[0].value, ast.Str)):
        result["docstring"] = node.body[0].value.s
    
    # Parse parameter
    for arg in node.args.args:
        param = {"name": arg.arg}
        
        # Parse tipe parameter
        if arg.annotation:
            if isinstance(arg.annotation, ast.Name):
                param["type"] = arg.annotation.id
            elif isinstance(arg.annotation, ast.Subscript):
                # Tipe kompleks seperti List[int]
                param["type"] = ast.unparse(arg.annotation)
            else:
                param["type"] = "Any"
        else:
            param["type"] = "Any"
        
        result["params"].append(param)
    
    # Parse nilai default parameter
    defaults = node.args.defaults
    if defaults:
        for i, default in enumerate(defaults):
            idx = len(result["params"]) - len(defaults) + i
            if idx >= 0:
                if isinstance(default, ast.Constant):
                    result["params"][idx]["default"] = default.value
                else:
                    # Nilai default kompleks
                    result["params"][idx]["default"] = ast.unparse(default)
    
    # Parse tipe return
    if node.returns:
        if isinstance(node.returns, ast.Name):
            result["return_type"] = node.returns.id
        elif isinstance(node.returns, ast.Subscript):
            # Tipe kompleks seperti List[int]
            result["return_type"] = ast.unparse(node.returns)
        else:
            result["return_type"] = "Any"
    
    # Parse badan fungsi
    for stmt in node.body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Str) and stmt == node.body[0]):
            # Bukan docstring
            result["body"].append(ast.unparse(stmt))
    
    return result


def parse_class(node: ast.ClassDef) -> Dict[str, Any]:
    """
    Parse definisi kelas.
    
    Args:
        node: Node AST kelas
        
    Returns:
        Struktur kelas
    """
    # Inisialisasi hasil
    result = {
        "name": node.name,
        "base_classes": [],
        "attributes": [],
        "methods": []
    }
    
    # Parse kelas dasar
    for base in node.bases:
        if isinstance(base, ast.Name):
            result["base_classes"].append(base.id)
        else:
            # Kelas dasar kompleks
            result["base_classes"].append(ast.unparse(base))
    
    # Parse docstring
    if (len(node.body) > 0 and 
        isinstance(node.body[0], ast.Expr) and 
        isinstance(node.body[0].value, ast.Str)):
        result["docstring"] = node.body[0].value.s
    
    # Parse atribut dan metode
    for item in node.body:
        if isinstance(item, ast.AnnAssign):
            # Atribut dengan anotasi tipe
            attr = {"name": item.target.id}
            
            # Parse tipe atribut
            if item.annotation:
                if isinstance(item.annotation, ast.Name):
                    attr["type"] = item.annotation.id
                elif isinstance(item.annotation, ast.Subscript):
                    # Tipe kompleks seperti List[int]
                    attr["type"] = ast.unparse(item.annotation)
                else:
                    attr["type"] = "Any"
            else:
                attr["type"] = "Any"
            
            # Parse nilai atribut
            if item.value:
                if isinstance(item.value, ast.Constant):
                    attr["value"] = item.value.value
                else:
                    # Nilai kompleks
                    attr["value"] = ast.unparse(item.value)
            
            result["attributes"].append(attr)
        
        elif isinstance(item, ast.Assign):
            # Atribut tanpa anotasi tipe
            for target in item.targets:
                if isinstance(target, ast.Name):
                    attr = {"name": target.id, "type": "Any"}
                    
                    # Parse nilai atribut
                    if isinstance(item.value, ast.Constant):
                        attr["value"] = item.value.value
                    else:
                        # Nilai kompleks
                        attr["value"] = ast.unparse(item.value)
                    
                    result["attributes"].append(attr)
        
        elif isinstance(item, ast.FunctionDef):
            # Metode
            result["methods"].append(parse_function(item))
    
    return result