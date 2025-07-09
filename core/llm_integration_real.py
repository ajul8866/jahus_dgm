"""
Integrasi LLM yang sebenarnya untuk Darwin-Gödel Machine.

Modul ini berisi implementasi integrasi LLM yang sebenarnya untuk DGM,
menggunakan OpenAI API untuk meningkatkan kemampuan agen.
"""

import os
import json
import time
import random
from typing import List, Dict, Any, Optional, Callable, Tuple, Union, Type

import openai
from openai import OpenAI

from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.core.evolution_strategies import EvolutionStrategy
from simple_dgm.core.mutation_operators import MutationOperator
from simple_dgm.core.crossover_operators import CrossoverOperator
from simple_dgm.core.fitness_functions import FitnessFunction
from simple_dgm.core.diversity_metrics import DiversityMetric
from simple_dgm.core.archive_strategies import ArchiveStrategy

class LLMInterface:
    """
    Antarmuka LLM untuk DGM.
    
    Kelas ini menyediakan antarmuka untuk berinteraksi dengan model bahasa besar (LLM)
    untuk meningkatkan kemampuan DGM.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Inisialisasi antarmuka LLM.
        
        Args:
            api_key: API key untuk LLM (opsional, jika tidak diberikan akan menggunakan environment variable)
            model: Model LLM yang akan digunakan
        """
        self.model = model
        
        # Dapatkan API key dari parameter atau environment variable
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Inisialisasi klien OpenAI
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.available = True
        else:
            self.available = False
            print("Warning: No API key provided for LLM. LLM integration will not be available.")
    
    def query(self, prompt: str, system_message: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Kirim kueri ke LLM.
        
        Args:
            prompt: Prompt untuk LLM
            system_message: Pesan sistem untuk LLM (opsional)
            temperature: Nilai temperature untuk LLM
            
        Returns:
            Respons LLM
        """
        if not self.available:
            return "LLM not available. Please provide a valid API key."
        
        try:
            messages = []
            
            # Tambahkan pesan sistem jika diberikan
            if system_message:
                messages.append({"role": "system", "content": system_message})
            else:
                messages.append({"role": "system", "content": "You are a helpful assistant for a Darwin-Gödel Machine, an advanced AI system that combines evolutionary algorithms and self-improvement."})
            
            # Tambahkan prompt pengguna
            messages.append({"role": "user", "content": prompt})
            
            # Kirim permintaan ke OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return f"Error: {str(e)}"
    
    def query_json(self, prompt: str, system_message: Optional[str] = None, temperature: float = 0.2) -> Dict[str, Any]:
        """
        Kirim kueri ke LLM dan dapatkan respons dalam format JSON.
        
        Args:
            prompt: Prompt untuk LLM
            system_message: Pesan sistem untuk LLM (opsional)
            temperature: Nilai temperature untuk LLM
            
        Returns:
            Respons LLM dalam format JSON
        """
        if not self.available:
            return {"error": "LLM not available. Please provide a valid API key."}
        
        try:
            if system_message:
                system_prompt = system_message + "\n\nYou must respond in valid JSON format only."
            else:
                system_prompt = "You are a helpful assistant for a Darwin-Gödel Machine, an advanced AI system that combines evolutionary algorithms and self-improvement. You must respond in valid JSON format only."
            
            # Kirim permintaan ke OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse respons JSON
            response_text = response.choices[0].message.content
            return json.loads(response_text)
        except json.JSONDecodeError:
            print(f"Error decoding JSON response: {response_text}")
            return {"error": "Invalid JSON response from LLM"}
        except Exception as e:
            print(f"Error querying LLM for JSON: {e}")
            return {"error": str(e)}
    
    def improve_agent(self, agent: BaseAgent, task_description: str) -> BaseAgent:
        """
        Tingkatkan agen menggunakan LLM.
        
        Args:
            agent: Agen yang akan ditingkatkan
            task_description: Deskripsi tugas yang akan dilakukan agen
            
        Returns:
            Agen yang ditingkatkan
        """
        if not self.available:
            print("LLM not available. Cannot improve agent.")
            return agent
        
        # Buat prompt untuk LLM
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in agent.tools])
        
        prompt = f"""
        I have an agent with the following tools:
        {tools_description}
        
        The agent has the following parameters:
        - memory_capacity: {agent.memory_capacity}
        - learning_rate: {agent.learning_rate}
        - exploration_rate: {agent.exploration_rate}
        
        The agent needs to perform the following task:
        {task_description}
        
        Please suggest improvements to this agent. Your response should be in JSON format with the following structure:
        {{
            "tools": [
                {{"name": "tool_name", "description": "tool_description"}}
            ],
            "parameters": {{
                "memory_capacity": value,
                "learning_rate": value,
                "exploration_rate": value
            }},
            "reasoning": "Your reasoning for the suggested improvements"
        }}
        """
        
        # Kirim prompt ke LLM
        response = self.query_json(prompt)
        
        # Jika ada error, kembalikan agen asli
        if "error" in response:
            print(f"Error improving agent: {response['error']}")
            return agent
        
        # Buat agen baru berdasarkan agen yang ada
        new_agent = BaseAgent(
            memory_capacity=agent.memory_capacity,
            learning_rate=agent.learning_rate,
            exploration_rate=agent.exploration_rate
        )
        
        # Salin alat dari agen lama
        for tool in agent.tools:
            new_agent.add_tool(tool)
        
        # Perbarui parameter berdasarkan respons LLM
        if "parameters" in response:
            params = response["parameters"]
            if "memory_capacity" in params:
                new_agent.memory_capacity = params["memory_capacity"]
            if "learning_rate" in params:
                new_agent.learning_rate = params["learning_rate"]
            if "exploration_rate" in params:
                new_agent.exploration_rate = params["exploration_rate"]
        
        # Tambahkan alat baru yang disarankan oleh LLM
        if "tools" in response:
            for tool_info in response["tools"]:
                # Periksa apakah alat sudah ada
                if not any(t.name == tool_info["name"] for t in new_agent.tools):
                    # Tambahkan alat baru jika implementasinya tersedia
                    if hasattr(self, f"_implement_{tool_info['name']}"):
                        tool_function = getattr(self, f"_implement_{tool_info['name']}")
                        new_agent.add_tool(Tool(
                            name=tool_info["name"],
                            function=tool_function,
                            description=tool_info["description"]
                        ))
        
        # Cetak alasan peningkatan
        if "reasoning" in response:
            print(f"\nLLM Reasoning for Improvements:\n{response['reasoning']}")
        
        return new_agent
    
    def generate_evolution_strategy(self, task_description: str) -> Dict[str, Any]:
        """
        Hasilkan strategi evolusi menggunakan LLM.
        
        Args:
            task_description: Deskripsi tugas
            
        Returns:
            Strategi evolusi
        """
        if not self.available:
            print("LLM not available. Cannot generate evolution strategy.")
            return {}
        
        prompt = f"""
        I need to design an evolution strategy for a Darwin-Gödel Machine that will solve the following task:
        {task_description}
        
        Please suggest an appropriate evolution strategy. Your response should be in JSON format with the following structure:
        {{
            "strategy_type": "tournament_selection", // or other strategy type
            "parameters": {{
                "tournament_size": value,
                "offspring_size": value,
                // other parameters specific to the strategy
            }},
            "reasoning": "Your reasoning for the suggested strategy"
        }}
        """
        
        # Kirim prompt ke LLM
        response = self.query_json(prompt)
        
        # Jika ada error, kembalikan strategi kosong
        if "error" in response:
            print(f"Error generating evolution strategy: {response['error']}")
            return {}
        
        # Cetak alasan pemilihan strategi
        if "reasoning" in response:
            print(f"\nLLM Reasoning for Evolution Strategy:\n{response['reasoning']}")
        
        return response
    
    def generate_mutation_operator(self, task_description: str) -> Dict[str, Any]:
        """
        Hasilkan operator mutasi menggunakan LLM.
        
        Args:
            task_description: Deskripsi tugas
            
        Returns:
            Operator mutasi
        """
        if not self.available:
            print("LLM not available. Cannot generate mutation operator.")
            return {}
        
        prompt = f"""
        I need to design a mutation operator for a Darwin-Gödel Machine that will solve the following task:
        {task_description}
        
        Please suggest an appropriate mutation operator. Your response should be in JSON format with the following structure:
        {{
            "operator_type": "parameter_mutation", // or other operator type
            "parameters": {{
                "mutation_rate": value,
                "mutation_strength": value,
                // other parameters specific to the operator
            }},
            "reasoning": "Your reasoning for the suggested operator"
        }}
        """
        
        # Kirim prompt ke LLM
        response = self.query_json(prompt)
        
        # Jika ada error, kembalikan operator kosong
        if "error" in response:
            print(f"Error generating mutation operator: {response['error']}")
            return {}
        
        # Cetak alasan pemilihan operator
        if "reasoning" in response:
            print(f"\nLLM Reasoning for Mutation Operator:\n{response['reasoning']}")
        
        return response
    
    def generate_crossover_operator(self, task_description: str) -> Dict[str, Any]:
        """
        Hasilkan operator crossover menggunakan LLM.
        
        Args:
            task_description: Deskripsi tugas
            
        Returns:
            Operator crossover
        """
        if not self.available:
            print("LLM not available. Cannot generate crossover operator.")
            return {}
        
        prompt = f"""
        I need to design a crossover operator for a Darwin-Gödel Machine that will solve the following task:
        {task_description}
        
        Please suggest an appropriate crossover operator. Your response should be in JSON format with the following structure:
        {{
            "operator_type": "blend_crossover", // or other operator type
            "parameters": {{
                "alpha": value,
                // other parameters specific to the operator
            }},
            "reasoning": "Your reasoning for the suggested operator"
        }}
        """
        
        # Kirim prompt ke LLM
        response = self.query_json(prompt)
        
        # Jika ada error, kembalikan operator kosong
        if "error" in response:
            print(f"Error generating crossover operator: {response['error']}")
            return {}
        
        # Cetak alasan pemilihan operator
        if "reasoning" in response:
            print(f"\nLLM Reasoning for Crossover Operator:\n{response['reasoning']}")
        
        return response
    
    def analyze_evolution_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis hasil evolusi menggunakan LLM.
        
        Args:
            results: Hasil evolusi
            
        Returns:
            Analisis hasil evolusi
        """
        if not self.available:
            print("LLM not available. Cannot analyze evolution results.")
            return {}
        
        # Konversi hasil evolusi ke string
        results_str = json.dumps(results, indent=2)
        
        prompt = f"""
        I have the following results from running evolution in a Darwin-Gödel Machine:
        {results_str}
        
        Please analyze these results and provide insights. Your response should be in JSON format with the following structure:
        {{
            "insights": [
                "Insight 1",
                "Insight 2",
                // more insights
            ],
            "recommendations": [
                "Recommendation 1",
                "Recommendation 2",
                // more recommendations
            ],
            "analysis": "Your detailed analysis of the results"
        }}
        """
        
        # Kirim prompt ke LLM
        response = self.query_json(prompt)
        
        # Jika ada error, kembalikan analisis kosong
        if "error" in response:
            print(f"Error analyzing evolution results: {response['error']}")
            return {}
        
        return response
    
    def generate_code(self, task_description: str, language: str = "python") -> str:
        """
        Hasilkan kode menggunakan LLM.
        
        Args:
            task_description: Deskripsi tugas
            language: Bahasa pemrograman
            
        Returns:
            Kode yang dihasilkan
        """
        if not self.available:
            print("LLM not available. Cannot generate code.")
            return ""
        
        prompt = f"""
        Please generate {language} code for the following task:
        {task_description}
        
        The code should be well-documented, efficient, and follow best practices.
        """
        
        # Kirim prompt ke LLM
        response = self.query(prompt)
        
        # Ekstrak kode dari respons
        import re
        code_pattern = r"```(?:python)?\n(.*?)```"
        code_match = re.search(code_pattern, response, re.DOTALL)
        
        if code_match:
            return code_match.group(1).strip()
        else:
            return response
    
    def _implement_multiply(self, x, y):
        """
        Implementasi fungsi perkalian.
        
        Args:
            x: Angka pertama
            y: Angka kedua
            
        Returns:
            Hasil perkalian
        """
        return x * y
    
    def _implement_divide(self, x, y):
        """
        Implementasi fungsi pembagian.
        
        Args:
            x: Angka pertama
            y: Angka kedua
            
        Returns:
            Hasil pembagian
        """
        return x / y if y != 0 else 0
    
    def _implement_power(self, x, y):
        """
        Implementasi fungsi pangkat.
        
        Args:
            x: Basis
            y: Eksponen
            
        Returns:
            Hasil pangkat
        """
        return x ** y
    
    def _implement_sqrt(self, x):
        """
        Implementasi fungsi akar kuadrat.
        
        Args:
            x: Angka
            
        Returns:
            Akar kuadrat
        """
        return x ** 0.5 if x >= 0 else 0
    
    def _implement_log(self, x, base=None):
        """
        Implementasi fungsi logaritma.
        
        Args:
            x: Angka
            base: Basis logaritma (opsional)
            
        Returns:
            Hasil logaritma
        """
        import math
        if x <= 0:
            return float('nan')
        if base is None:
            return math.log(x)
        elif base <= 0:
            return float('nan')
        else:
            return math.log(x, base)
    
    def _implement_sin(self, x):
        """
        Implementasi fungsi sinus.
        
        Args:
            x: Angka
            
        Returns:
            Hasil sinus
        """
        import math
        return math.sin(x)
    
    def _implement_cos(self, x):
        """
        Implementasi fungsi kosinus.
        
        Args:
            x: Angka
            
        Returns:
            Hasil kosinus
        """
        import math
        return math.cos(x)
    
    def _implement_tan(self, x):
        """
        Implementasi fungsi tangen.
        
        Args:
            x: Angka
            
        Returns:
            Hasil tangen
        """
        import math
        return math.tan(x)
    
    def _implement_factorial(self, x):
        """
        Implementasi fungsi faktorial.
        
        Args:
            x: Angka
            
        Returns:
            Hasil faktorial
        """
        if x < 0:
            return float('nan')
        if x == 0 or x == 1:
            return 1
        result = 1
        for i in range(2, int(x) + 1):
            result *= i
        return result
    
    def _implement_gcd(self, x, y):
        """
        Implementasi fungsi GCD (Greatest Common Divisor).
        
        Args:
            x: Angka pertama
            y: Angka kedua
            
        Returns:
            GCD
        """
        import math
        return math.gcd(int(x), int(y))
    
    def _implement_lcm(self, x, y):
        """
        Implementasi fungsi LCM (Least Common Multiple).
        
        Args:
            x: Angka pertama
            y: Angka kedua
            
        Returns:
            LCM
        """
        import math
        return abs(x * y) // math.gcd(int(x), int(y)) if x != 0 and y != 0 else 0
    
    def _implement_mod(self, x, y):
        """
        Implementasi fungsi modulo.
        
        Args:
            x: Angka pertama
            y: Angka kedua
            
        Returns:
            Hasil modulo
        """
        return x % y if y != 0 else float('nan')
    
    def _implement_floor(self, x):
        """
        Implementasi fungsi floor.
        
        Args:
            x: Angka
            
        Returns:
            Hasil floor
        """
        import math
        return math.floor(x)
    
    def _implement_ceil(self, x):
        """
        Implementasi fungsi ceiling.
        
        Args:
            x: Angka
            
        Returns:
            Hasil ceiling
        """
        import math
        return math.ceil(x)
    
    def _implement_round(self, x, digits=0):
        """
        Implementasi fungsi pembulatan.
        
        Args:
            x: Angka
            digits: Jumlah digit desimal
            
        Returns:
            Hasil pembulatan
        """
        return round(x, int(digits))
    
    def _implement_abs(self, x):
        """
        Implementasi fungsi nilai absolut.
        
        Args:
            x: Angka
            
        Returns:
            Nilai absolut
        """
        return abs(x)
    
    def _implement_max(self, *args):
        """
        Implementasi fungsi maksimum.
        
        Args:
            *args: Daftar angka
            
        Returns:
            Nilai maksimum
        """
        return max(args)
    
    def _implement_min(self, *args):
        """
        Implementasi fungsi minimum.
        
        Args:
            *args: Daftar angka
            
        Returns:
            Nilai minimum
        """
        return min(args)
    
    def _implement_sum(self, *args):
        """
        Implementasi fungsi jumlah.
        
        Args:
            *args: Daftar angka
            
        Returns:
            Jumlah
        """
        return sum(args)
    
    def _implement_mean(self, *args):
        """
        Implementasi fungsi rata-rata.
        
        Args:
            *args: Daftar angka
            
        Returns:
            Rata-rata
        """
        return sum(args) / len(args) if args else 0
    
    def _implement_median(self, *args):
        """
        Implementasi fungsi median.
        
        Args:
            *args: Daftar angka
            
        Returns:
            Median
        """
        if not args:
            return 0
        sorted_args = sorted(args)
        n = len(sorted_args)
        if n % 2 == 0:
            return (sorted_args[n//2 - 1] + sorted_args[n//2]) / 2
        else:
            return sorted_args[n//2]
    
    def _implement_variance(self, *args):
        """
        Implementasi fungsi varians.
        
        Args:
            *args: Daftar angka
            
        Returns:
            Varians
        """
        if not args:
            return 0
        mean = sum(args) / len(args)
        return sum((x - mean) ** 2 for x in args) / len(args)
    
    def _implement_std(self, *args):
        """
        Implementasi fungsi standar deviasi.
        
        Args:
            *args: Daftar angka
            
        Returns:
            Standar deviasi
        """
        if not args:
            return 0
        mean = sum(args) / len(args)
        variance = sum((x - mean) ** 2 for x in args) / len(args)
        return variance ** 0.5

class CodeGeneration:
    """
    Komponen pembuatan kode untuk DGM.
    
    Kelas ini menyediakan fungsionalitas untuk menghasilkan kode menggunakan LLM.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Inisialisasi komponen pembuatan kode.
        
        Args:
            llm_interface: Antarmuka LLM
        """
        self.llm_interface = llm_interface
    
    def generate_function(self, function_name: str, description: str, parameters: List[Dict[str, str]], return_type: str) -> str:
        """
        Hasilkan fungsi.
        
        Args:
            function_name: Nama fungsi
            description: Deskripsi fungsi
            parameters: Daftar parameter fungsi
            return_type: Tipe pengembalian fungsi
            
        Returns:
            Kode fungsi
        """
        # Buat deskripsi parameter
        params_desc = "\n".join([f"- {param['name']} ({param['type']}): {param['description']}" for param in parameters])
        
        task_description = f"""
        Generate a Python function with the following specifications:
        
        Function Name: {function_name}
        Description: {description}
        
        Parameters:
        {params_desc}
        
        Return Type: {return_type}
        
        The function should be well-documented with docstrings and comments where necessary.
        """
        
        return self.llm_interface.generate_code(task_description)
    
    def generate_class(self, class_name: str, description: str, base_classes: List[str], methods: List[Dict[str, Any]]) -> str:
        """
        Hasilkan kelas.
        
        Args:
            class_name: Nama kelas
            description: Deskripsi kelas
            base_classes: Daftar kelas dasar
            methods: Daftar metode kelas
            
        Returns:
            Kode kelas
        """
        # Buat deskripsi metode
        methods_desc = ""
        for method in methods:
            methods_desc += f"Method: {method['name']}\n"
            methods_desc += f"Description: {method['description']}\n"
            
            if "parameters" in method:
                methods_desc += "Parameters:\n"
                for param in method["parameters"]:
                    methods_desc += f"- {param['name']} ({param['type']}): {param['description']}\n"
            
            if "return_type" in method:
                methods_desc += f"Return Type: {method['return_type']}\n"
            
            methods_desc += "\n"
        
        task_description = f"""
        Generate a Python class with the following specifications:
        
        Class Name: {class_name}
        Description: {description}
        Base Classes: {', '.join(base_classes)}
        
        Methods:
        {methods_desc}
        
        The class should be well-documented with docstrings and comments where necessary.
        """
        
        return self.llm_interface.generate_code(task_description)
    
    def optimize_code(self, code: str, optimization_goal: str) -> str:
        """
        Optimalkan kode.
        
        Args:
            code: Kode yang akan dioptimalkan
            optimization_goal: Tujuan optimasi
            
        Returns:
            Kode yang dioptimalkan
        """
        task_description = f"""
        Optimize the following Python code for {optimization_goal}:
        
        ```python
        {code}
        ```
        
        Please provide the optimized code with explanations of the optimizations made.
        """
        
        return self.llm_interface.generate_code(task_description)
    
    def refactor_code(self, code: str, refactoring_type: str) -> str:
        """
        Refaktor kode.
        
        Args:
            code: Kode yang akan direfaktor
            refactoring_type: Jenis refaktor
            
        Returns:
            Kode yang direfaktor
        """
        task_description = f"""
        Refactor the following Python code using {refactoring_type}:
        
        ```python
        {code}
        ```
        
        Please provide the refactored code with explanations of the changes made.
        """
        
        return self.llm_interface.generate_code(task_description)

class ProblemSolving:
    """
    Komponen pemecahan masalah untuk DGM.
    
    Kelas ini menyediakan fungsionalitas untuk memecahkan masalah menggunakan LLM.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Inisialisasi komponen pemecahan masalah.
        
        Args:
            llm_interface: Antarmuka LLM
        """
        self.llm_interface = llm_interface
    
    def analyze_problem(self, problem: str) -> Dict[str, Any]:
        """
        Analisis masalah.
        
        Args:
            problem: Masalah yang akan dianalisis
            
        Returns:
            Hasil analisis
        """
        prompt = f"""
        Please analyze the following problem:
        {problem}
        
        Your analysis should include:
        1. Problem type
        2. Complexity
        3. Key concepts
        4. Potential approaches
        5. Potential challenges
        
        Your response should be in JSON format with the following structure:
        {{
            "problem_type": "type",
            "complexity": "complexity",
            "key_concepts": ["concept1", "concept2", ...],
            "approaches": [
                {{
                    "name": "approach_name",
                    "description": "approach_description",
                    "pros": ["pro1", "pro2", ...],
                    "cons": ["con1", "con2", ...]
                }}
            ],
            "challenges": ["challenge1", "challenge2", ...]
        }}
        """
        
        return self.llm_interface.query_json(prompt)
    
    def decompose_problem(self, problem: str) -> List[Dict[str, Any]]:
        """
        Dekomposisi masalah.
        
        Args:
            problem: Masalah yang akan didekomposisi
            
        Returns:
            Hasil dekomposisi
        """
        prompt = f"""
        Please decompose the following problem into smaller sub-problems:
        {problem}
        
        Your decomposition should include:
        1. Sub-problem ID
        2. Sub-problem description
        3. Dependencies (if any)
        
        Your response should be in JSON format with the following structure:
        {{
            "sub_problems": [
                {{
                    "id": "subproblem1",
                    "description": "description",
                    "dependencies": ["subproblem2", ...]
                }}
            ]
        }}
        """
        
        response = self.llm_interface.query_json(prompt)
        
        if "sub_problems" in response:
            return response["sub_problems"]
        else:
            return []
    
    def solve_problem(self, problem: str) -> str:
        """
        Selesaikan masalah.
        
        Args:
            problem: Masalah yang akan diselesaikan
            
        Returns:
            Solusi masalah
        """
        prompt = f"""
        Please solve the following problem:
        {problem}
        
        Your solution should be detailed and include:
        1. Your approach
        2. Step-by-step solution
        3. Final answer
        4. Verification of the answer
        """
        
        return self.llm_interface.query(prompt)
    
    def evaluate_solution(self, problem: str, solution: str) -> Dict[str, Any]:
        """
        Evaluasi solusi.
        
        Args:
            problem: Masalah
            solution: Solusi yang akan dievaluasi
            
        Returns:
            Hasil evaluasi
        """
        prompt = f"""
        Please evaluate the following solution to the given problem:
        
        Problem:
        {problem}
        
        Solution:
        {solution}
        
        Your evaluation should include:
        1. Correctness
        2. Efficiency
        3. Clarity
        4. Overall score
        5. Strengths
        6. Weaknesses
        7. Suggestions for improvement
        
        Your response should be in JSON format with the following structure:
        {{
            "correctness": score, // 0.0 to 1.0
            "efficiency": score, // 0.0 to 1.0
            "clarity": score, // 0.0 to 1.0
            "overall_score": score, // 0.0 to 1.0
            "strengths": ["strength1", "strength2", ...],
            "weaknesses": ["weakness1", "weakness2", ...],
            "suggestions": ["suggestion1", "suggestion2", ...]
        }}
        """
        
        return self.llm_interface.query_json(prompt)

class KnowledgeExtraction:
    """
    Komponen ekstraksi pengetahuan untuk DGM.
    
    Kelas ini menyediakan fungsionalitas untuk mengekstrak pengetahuan dari teks menggunakan LLM.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Inisialisasi komponen ekstraksi pengetahuan.
        
        Args:
            llm_interface: Antarmuka LLM
        """
        self.llm_interface = llm_interface
    
    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Ekstrak konsep dari teks.
        
        Args:
            text: Teks yang akan diekstrak
            
        Returns:
            Daftar konsep
        """
        prompt = f"""
        Please extract key concepts from the following text:
        {text}
        
        For each concept, provide:
        1. Concept name
        2. Definition
        3. Importance
        4. Related concepts
        
        Your response should be in JSON format with the following structure:
        {{
            "concepts": [
                {{
                    "name": "concept_name",
                    "definition": "definition",
                    "importance": "importance",
                    "related_concepts": ["concept1", "concept2", ...]
                }}
            ]
        }}
        """
        
        response = self.llm_interface.query_json(prompt)
        
        if "concepts" in response:
            return response["concepts"]
        else:
            return []
    
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Ekstrak hubungan dari teks.
        
        Args:
            text: Teks yang akan diekstrak
            
        Returns:
            Daftar hubungan
        """
        prompt = f"""
        Please extract relationships between concepts from the following text:
        {text}
        
        For each relationship, provide:
        1. Source concept
        2. Target concept
        3. Relationship type
        4. Description
        
        Your response should be in JSON format with the following structure:
        {{
            "relationships": [
                {{
                    "source": "source_concept",
                    "target": "target_concept",
                    "type": "relationship_type",
                    "description": "description"
                }}
            ]
        }}
        """
        
        response = self.llm_interface.query_json(prompt)
        
        if "relationships" in response:
            return response["relationships"]
        else:
            return []
    
    def extract_knowledge_graph(self, text: str) -> Dict[str, Any]:
        """
        Ekstrak grafik pengetahuan dari teks.
        
        Args:
            text: Teks yang akan diekstrak
            
        Returns:
            Grafik pengetahuan
        """
        prompt = f"""
        Please extract a knowledge graph from the following text:
        {text}
        
        The knowledge graph should include:
        1. Nodes (concepts)
        2. Edges (relationships between concepts)
        
        Your response should be in JSON format with the following structure:
        {{
            "nodes": [
                {{
                    "id": "node_id",
                    "label": "node_label",
                    "type": "node_type",
                    "properties": {{
                        "property1": "value1",
                        "property2": "value2"
                    }}
                }}
            ],
            "edges": [
                {{
                    "source": "source_node_id",
                    "target": "target_node_id",
                    "label": "edge_label",
                    "properties": {{
                        "property1": "value1",
                        "property2": "value2"
                    }}
                }}
            ]
        }}
        """
        
        return self.llm_interface.query_json(prompt)

class SelfModification:
    """
    Komponen modifikasi diri untuk DGM.
    
    Kelas ini menyediakan fungsionalitas untuk memodifikasi kode DGM menggunakan LLM.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Inisialisasi komponen modifikasi diri.
        
        Args:
            llm_interface: Antarmuka LLM
        """
        self.llm_interface = llm_interface
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analisis kode.
        
        Args:
            code: Kode yang akan dianalisis
            
        Returns:
            Hasil analisis
        """
        prompt = f"""
        Please analyze the following Python code:
        ```python
        {code}
        ```
        
        Your analysis should include:
        1. Code quality
        2. Complexity
        3. Potential issues
        4. Potential improvements
        
        Your response should be in JSON format with the following structure:
        {{
            "quality": {{
                "score": score, // 0.0 to 1.0
                "comments": ["comment1", "comment2", ...]
            }},
            "complexity": {{
                "score": score, // 0.0 to 1.0
                "comments": ["comment1", "comment2", ...]
            }},
            "issues": [
                {{
                    "type": "issue_type",
                    "description": "description",
                    "severity": "severity",
                    "location": "location"
                }}
            ],
            "improvements": [
                {{
                    "type": "improvement_type",
                    "description": "description",
                    "impact": "impact"
                }}
            ]
        }}
        """
        
        return self.llm_interface.query_json(prompt)
    
    def improve_code(self, code: str) -> str:
        """
        Tingkatkan kode.
        
        Args:
            code: Kode yang akan ditingkatkan
            
        Returns:
            Kode yang ditingkatkan
        """
        prompt = f"""
        Please improve the following Python code:
        ```python
        {code}
        ```
        
        Your improvements should focus on:
        1. Code quality
        2. Performance
        3. Readability
        4. Maintainability
        
        Please provide the improved code with explanations of the improvements made.
        """
        
        return self.llm_interface.generate_code(prompt)
    
    def add_feature(self, code: str, feature_description: str) -> str:
        """
        Tambahkan fitur ke kode.
        
        Args:
            code: Kode yang akan ditambahkan fitur
            feature_description: Deskripsi fitur
            
        Returns:
            Kode dengan fitur baru
        """
        prompt = f"""
        Please add the following feature to the Python code:
        
        Feature Description:
        {feature_description}
        
        Original Code:
        ```python
        {code}
        ```
        
        Please provide the modified code with the new feature implemented.
        """
        
        return self.llm_interface.generate_code(prompt)
    
    def refactor_code(self, code: str, refactoring_type: str) -> str:
        """
        Refaktor kode.
        
        Args:
            code: Kode yang akan direfaktor
            refactoring_type: Jenis refaktor
            
        Returns:
            Kode yang direfaktor
        """
        prompt = f"""
        Please refactor the following Python code using {refactoring_type}:
        ```python
        {code}
        ```
        
        Please provide the refactored code with explanations of the changes made.
        """
        
        return self.llm_interface.generate_code(prompt)