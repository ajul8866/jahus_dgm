"""
Implementasi agen pengkodean untuk Darwin-Gödel Machine.
"""

import random
import re
import ast
import os
from typing import Dict, Any, List, Optional, Callable, Union, Tuple

from simple_dgm.agents.base_agent import BaseAgent, Tool


class CodeAnalyzer:
    """
    Utilitas untuk menganalisis kode.
    """
    
    @staticmethod
    def analyze_complexity(code: str) -> Dict[str, Any]:
        """
        Analisis kompleksitas kode.
        
        Args:
            code: Kode yang akan dianalisis
            
        Returns:
            Metrik kompleksitas kode
        """
        try:
            # Parse kode
            tree = ast.parse(code)
            
            # Hitung metrik
            num_functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            num_classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            num_imports = len([node for node in ast.walk(tree) if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)])
            num_loops = len([node for node in ast.walk(tree) if isinstance(node, ast.For) or isinstance(node, ast.While)])
            num_conditionals = len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
            
            # Hitung kompleksitas siklomatik (sederhana)
            cyclomatic_complexity = 1 + num_conditionals + num_loops
            
            return {
                "num_functions": num_functions,
                "num_classes": num_classes,
                "num_imports": num_imports,
                "num_loops": num_loops,
                "num_conditionals": num_conditionals,
                "cyclomatic_complexity": cyclomatic_complexity,
                "loc": len(code.split('\n'))
            }
        except SyntaxError:
            # Jika kode tidak valid
            return {
                "error": "Invalid syntax",
                "loc": len(code.split('\n'))
            }
    
    @staticmethod
    def detect_bugs(code: str) -> List[Dict[str, Any]]:
        """
        Deteksi bug potensial dalam kode.
        
        Args:
            code: Kode yang akan dianalisis
            
        Returns:
            Daftar bug potensial
        """
        bugs = []
        
        # Deteksi penggunaan variabel yang tidak didefinisikan (sederhana)
        lines = code.split('\n')
        defined_vars = set()
        
        for i, line in enumerate(lines):
            # Deteksi definisi variabel (sederhana)
            var_def = re.findall(r'(\w+)\s*=', line)
            defined_vars.update(var_def)
            
            # Deteksi penggunaan variabel
            var_uses = re.findall(r'[^=\w](\w+)[^(]', line)
            for var in var_uses:
                if var not in defined_vars and var not in ['print', 'if', 'for', 'while', 'def', 'class', 'return', 'import', 'from']:
                    bugs.append({
                        "type": "undefined_variable",
                        "variable": var,
                        "line": i + 1,
                        "code": line
                    })
        
        # Deteksi penggunaan except tanpa spesifikasi exception
        for i, line in enumerate(lines):
            if re.search(r'except\s*:', line):
                bugs.append({
                    "type": "bare_except",
                    "line": i + 1,
                    "code": line,
                    "suggestion": "Specify the exception type to catch"
                })
        
        return bugs


class CodeGenerator:
    """
    Utilitas untuk menghasilkan kode.
    """
    
    @staticmethod
    def generate_function(name: str, params: List[str], body: List[str]) -> str:
        """
        Hasilkan definisi fungsi.
        
        Args:
            name: Nama fungsi
            params: Daftar parameter
            body: Daftar baris kode untuk badan fungsi
            
        Returns:
            Kode fungsi
        """
        params_str = ", ".join(params)
        body_str = "\n    ".join(body)
        return f"def {name}({params_str}):\n    {body_str}"
    
    @staticmethod
    def generate_class(name: str, methods: List[Dict[str, Any]]) -> str:
        """
        Hasilkan definisi kelas.
        
        Args:
            name: Nama kelas
            methods: Daftar metode (dict dengan keys: name, params, body)
            
        Returns:
            Kode kelas
        """
        class_code = f"class {name}:\n"
        
        for method in methods:
            method_name = method["name"]
            method_params = ["self"] + method.get("params", [])
            method_body = method.get("body", ["pass"])
            
            method_code = CodeGenerator.generate_function(method_name, method_params, method_body)
            # Indentasi metode
            method_code = "    " + method_code.replace("\n", "\n    ")
            
            class_code += method_code + "\n\n"
        
        return class_code


class CodingAgent(BaseAgent):
    """
    Agen khusus untuk tugas pengkodean.
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None, 
                 memory_capacity: int = 10,
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.1,
                 code_style: str = "clean",
                 preferred_language: str = "python"):
        """
        Inisialisasi agen pengkodean.
        
        Args:
            tools: Daftar alat yang tersedia untuk agen
            memory_capacity: Kapasitas memori agen
            learning_rate: Tingkat pembelajaran agen
            exploration_rate: Tingkat eksplorasi agen
            code_style: Gaya pengkodean ("clean", "verbose", "compact")
            preferred_language: Bahasa pemrograman yang disukai
        """
        super().__init__(tools, memory_capacity, learning_rate, exploration_rate)
        
        self.code_style = code_style
        self.preferred_language = preferred_language
        self.code_analyzer = CodeAnalyzer()
        self.code_generator = CodeGenerator()
        
        # Tambahkan alat khusus untuk pengkodean
        self._add_coding_tools()
        
        # Tambahkan metadata khusus
        self.metadata.update({
            "type": "CodingAgent",
            "code_style": code_style,
            "preferred_language": preferred_language
        })
    
    def _add_coding_tools(self) -> None:
        """
        Tambahkan alat khusus untuk pengkodean.
        """
        # Alat untuk menganalisis kode
        self.add_tool(Tool(
            name="analyze_code",
            function=self._analyze_code,
            description="Analisis kode untuk metrik kompleksitas dan bug potensial"
        ))
        
        # Alat untuk memperbaiki bug
        self.add_tool(Tool(
            name="fix_bugs",
            function=self._fix_bugs,
            description="Perbaiki bug dalam kode"
        ))
        
        # Alat untuk mengoptimalkan kode
        self.add_tool(Tool(
            name="optimize_code",
            function=self._optimize_code,
            description="Optimalkan kode untuk performa"
        ))
        
        # Alat untuk menghasilkan kode
        self.add_tool(Tool(
            name="generate_code",
            function=self._generate_code,
            description="Hasilkan kode berdasarkan spesifikasi"
        ))
    
    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analisis kode.
        
        Args:
            code: Kode yang akan dianalisis
            
        Returns:
            Hasil analisis
        """
        complexity = self.code_analyzer.analyze_complexity(code)
        bugs = self.code_analyzer.detect_bugs(code)
        
        return {
            "complexity": complexity,
            "bugs": bugs,
            "num_bugs": len(bugs)
        }
    
    def _fix_bugs(self, code: str, bugs: List[Dict[str, Any]]) -> str:
        """
        Perbaiki bug dalam kode.
        
        Args:
            code: Kode yang akan diperbaiki
            bugs: Daftar bug yang akan diperbaiki
            
        Returns:
            Kode yang diperbaiki
        """
        lines = code.split('\n')
        
        # Urutkan bug berdasarkan nomor baris (terbalik)
        bugs = sorted(bugs, key=lambda x: x["line"], reverse=True)
        
        for bug in bugs:
            line_num = bug["line"] - 1  # Indeks 0-based
            
            if bug["type"] == "undefined_variable":
                # Tambahkan definisi variabel
                var_name = bug["variable"]
                lines.insert(line_num, f"{var_name} = None  # Auto-added by CodingAgent")
            
            elif bug["type"] == "bare_except":
                # Ganti except: dengan except Exception:
                lines[line_num] = lines[line_num].replace("except:", "except Exception:")
        
        return '\n'.join(lines)
    
    def _optimize_code(self, code: str) -> str:
        """
        Optimalkan kode untuk performa.
        
        Args:
            code: Kode yang akan dioptimalkan
            
        Returns:
            Kode yang dioptimalkan
        """
        # Implementasi sederhana: ganti beberapa pola yang tidak efisien
        
        # Ganti range(len(x)) dengan enumerate(x)
        code = re.sub(
            r'for\s+(\w+)\s+in\s+range\(len\((\w+)\)\):',
            r'for \1, item_\1 in enumerate(\2):',
            code
        )
        
        # Ganti multiple append dengan list comprehension
        append_pattern = r'(\w+)\s*=\s*\[\]\s*\n\s*for\s+(\w+)\s+in\s+(\w+):\s*\n\s*\1\.append\((.+?)\)'
        code = re.sub(
            append_pattern,
            r'\1 = [\4 for \2 in \3]',
            code,
            flags=re.DOTALL
        )
        
        return code
    
    def _generate_code(self, spec: Dict[str, Any]) -> str:
        """
        Hasilkan kode berdasarkan spesifikasi.
        
        Args:
            spec: Spesifikasi kode
            
        Returns:
            Kode yang dihasilkan
        """
        code = ""
        
        if "imports" in spec:
            for imp in spec["imports"]:
                if "from" in imp:
                    code += f"from {imp['from']} import {', '.join(imp['import'])}\n"
                else:
                    code += f"import {', '.join(imp['import'])}\n"
            code += "\n"
        
        if "functions" in spec:
            for func in spec["functions"]:
                func_code = self.code_generator.generate_function(
                    func["name"],
                    func.get("params", []),
                    func.get("body", ["pass"])
                )
                code += func_code + "\n\n"
        
        if "classes" in spec:
            for cls in spec["classes"]:
                class_code = self.code_generator.generate_class(
                    cls["name"],
                    cls.get("methods", [])
                )
                code += class_code + "\n\n"
        
        if "main" in spec:
            code += "if __name__ == '__main__':\n"
            for line in spec["main"]:
                code += f"    {line}\n"
        
        return code
    
    def solve(self, problem: Dict[str, Any]) -> Any:
        """
        Selesaikan masalah pengkodean.
        
        Args:
            problem: Masalah pengkodean (dict dengan keys: type, data)
            
        Returns:
            Solusi untuk masalah
        """
        problem_type = problem.get("type", "")
        
        if problem_type == "analyze":
            return self._analyze_code(problem["code"])
        
        elif problem_type == "fix_bugs":
            bugs = self._analyze_code(problem["code"])["bugs"]
            return self._fix_bugs(problem["code"], bugs)
        
        elif problem_type == "optimize":
            return self._optimize_code(problem["code"])
        
        elif problem_type == "generate":
            return self._generate_code(problem["spec"])
        
        else:
            # Gunakan implementasi dasar
            return super().solve(problem)
    
    def mutate(self) -> None:
        """
        Mutasi agen pengkodean.
        """
        super().mutate()  # Panggil mutasi dasar
        
        # Mutasi parameter khusus
        styles = ["clean", "verbose", "compact"]
        languages = ["python", "javascript", "java", "c++", "rust"]
        
        # Dengan probabilitas tertentu, ubah gaya kode
        if random.random() < 0.3:
            self.code_style = random.choice(styles)
        
        # Dengan probabilitas tertentu, ubah bahasa yang disukai
        if random.random() < 0.2:
            self.preferred_language = random.choice(languages)
        
        # Update metadata
        self.metadata.update({
            "code_style": self.code_style,
            "preferred_language": self.preferred_language
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konversi agen pengkodean ke dictionary.
        """
        data = super().to_dict()
        data.update({
            "code_style": self.code_style,
            "preferred_language": self.preferred_language
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], function_map: Dict[str, Callable]) -> 'CodingAgent':
        """
        Buat agen pengkodean dari dictionary.
        """
        agent = super().from_dict(data, function_map)
        agent.code_style = data["code_style"]
        agent.preferred_language = data["preferred_language"]
        return agent