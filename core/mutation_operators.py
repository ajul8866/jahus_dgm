"""
Operator mutasi untuk Darwin-Gödel Machine.

Modul ini berisi implementasi berbagai operator mutasi yang digunakan oleh DGM
untuk memodifikasi agen-agen selama proses evolusi.
"""

import random
import copy
import inspect
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar, Generic, Union

from simple_dgm.agents.base_agent import BaseAgent, Tool

# Tipe generik untuk individu dalam populasi
T = TypeVar('T', bound=BaseAgent)


class MutationOperator(Generic[T]):
    """
    Kelas dasar untuk operator mutasi.
    
    Operator mutasi menentukan bagaimana individu dimodifikasi selama evolusi.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Inisialisasi operator mutasi.
        
        Args:
            mutation_rate: Probabilitas mutasi
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, individual: T) -> T:
        """
        Mutasi individu.
        
        Args:
            individual: Individu yang akan dimutasi
            
        Returns:
            Individu yang telah dimutasi
        """
        raise NotImplementedError("Subclass must implement abstract method")


class ParameterMutation(MutationOperator[T]):
    """
    Operator mutasi parameter.
    
    Operator ini memutasi parameter numerik individu.
    """
    
    def __init__(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1,
                parameter_names: Optional[List[str]] = None):
        """
        Inisialisasi operator mutasi parameter.
        
        Args:
            mutation_rate: Probabilitas mutasi
            mutation_strength: Kekuatan mutasi (faktor perubahan)
            parameter_names: Daftar nama parameter yang akan dimutasi
        """
        super().__init__(mutation_rate)
        self.mutation_strength = mutation_strength
        self.parameter_names = parameter_names or ["learning_rate", "exploration_rate", "memory_capacity"]
    
    def mutate(self, individual: T) -> T:
        """
        Mutasi parameter individu.
        
        Args:
            individual: Individu yang akan dimutasi
            
        Returns:
            Individu yang telah dimutasi
        """
        # Buat salinan individu
        mutated = copy.deepcopy(individual)
        
        # Mutasi setiap parameter dengan probabilitas mutation_rate
        for param_name in self.parameter_names:
            if hasattr(mutated, param_name) and random.random() < self.mutation_rate:
                param_value = getattr(mutated, param_name)
                
                # Mutasi berdasarkan tipe parameter
                if isinstance(param_value, (int, float)):
                    # Mutasi parameter numerik
                    if isinstance(param_value, int):
                        # Parameter integer
                        new_value = int(param_value * random.uniform(1 - self.mutation_strength, 1 + self.mutation_strength))
                        # Pastikan nilai minimal 1
                        new_value = max(1, new_value)
                    else:
                        # Parameter float
                        new_value = param_value * random.uniform(1 - self.mutation_strength, 1 + self.mutation_strength)
                        # Batasi nilai untuk parameter khusus
                        if param_name == "learning_rate":
                            new_value = max(0.001, min(0.1, new_value))
                        elif param_name == "exploration_rate":
                            new_value = max(0.01, min(0.5, new_value))
                    
                    # Tetapkan nilai baru
                    setattr(mutated, param_name, new_value)
                
                elif isinstance(param_value, str) and hasattr(mutated, f"{param_name}_options"):
                    # Mutasi parameter string dengan opsi
                    options = getattr(mutated, f"{param_name}_options")
                    if options:
                        new_value = random.choice(options)
                        setattr(mutated, param_name, new_value)
                
                elif isinstance(param_value, list):
                    # Mutasi parameter list
                    if param_value and all(isinstance(item, str) for item in param_value):
                        # List of strings
                        if hasattr(mutated, f"{param_name}_options"):
                            options = getattr(mutated, f"{param_name}_options")
                            if options:
                                # Tambah, hapus, atau ganti item
                                op = random.choice(["add", "remove", "replace"])
                                
                                if op == "add" and len(param_value) < len(options):
                                    # Tambah item baru
                                    available = [opt for opt in options if opt not in param_value]
                                    if available:
                                        param_value.append(random.choice(available))
                                
                                elif op == "remove" and len(param_value) > 1:
                                    # Hapus item acak
                                    param_value.pop(random.randrange(len(param_value)))
                                
                                elif op == "replace" and len(param_value) > 0:
                                    # Ganti item acak
                                    idx = random.randrange(len(param_value))
                                    available = [opt for opt in options if opt != param_value[idx]]
                                    if available:
                                        param_value[idx] = random.choice(available)
        
        return mutated


class StructuralMutation(MutationOperator[T]):
    """
    Operator mutasi struktural.
    
    Operator ini memutasi struktur individu, seperti menambah, menghapus,
    atau memodifikasi alat.
    """
    
    def __init__(self, mutation_rate: float = 0.1, tool_mutation_rate: float = 0.3,
                available_tools: Optional[List[Dict[str, Any]]] = None):
        """
        Inisialisasi operator mutasi struktural.
        
        Args:
            mutation_rate: Probabilitas mutasi
            tool_mutation_rate: Probabilitas mutasi alat
            available_tools: Daftar alat yang tersedia untuk ditambahkan
        """
        super().__init__(mutation_rate)
        self.tool_mutation_rate = tool_mutation_rate
        self.available_tools = available_tools or []
    
    def mutate(self, individual: T) -> T:
        """
        Mutasi struktur individu.
        
        Args:
            individual: Individu yang akan dimutasi
            
        Returns:
            Individu yang telah dimutasi
        """
        # Buat salinan individu
        mutated = copy.deepcopy(individual)
        
        # Mutasi alat dengan probabilitas mutation_rate
        if hasattr(mutated, "tools") and random.random() < self.mutation_rate:
            # Pilih operasi mutasi
            op = random.choices(["add", "remove", "modify"], 
                               weights=[0.4, 0.3, 0.3], k=1)[0]
            
            if op == "add" and self.available_tools:
                # Tambah alat baru
                tool_spec = random.choice(self.available_tools)
                
                # Periksa apakah alat sudah ada
                existing_names = [tool.name for tool in mutated.tools]
                if tool_spec["name"] not in existing_names:
                    # Buat alat baru
                    new_tool = Tool(
                        name=tool_spec["name"],
                        function=tool_spec["function"],
                        description=tool_spec.get("description", "")
                    )
                    
                    # Tambahkan alat ke individu
                    mutated.add_tool(new_tool)
            
            elif op == "remove" and mutated.tools:
                # Hapus alat acak
                tool_to_remove = random.choice(mutated.tools)
                mutated.remove_tool(tool_to_remove.name)
            
            elif op == "modify" and mutated.tools:
                # Modifikasi alat acak
                tool_to_modify = random.choice(mutated.tools)
                
                # Modifikasi deskripsi
                if random.random() < self.tool_mutation_rate:
                    # Tambahkan atau ubah deskripsi
                    descriptions = [
                        "Improved version of the tool",
                        "Enhanced tool with better performance",
                        "Optimized tool for faster execution",
                        "Modified tool with additional capabilities",
                        "Refined tool with better accuracy"
                    ]
                    tool_to_modify.description = random.choice(descriptions)
        
        # Mutasi atribut khusus berdasarkan jenis agen
        if hasattr(mutated, "__class__") and hasattr(mutated.__class__, "__name__"):
            class_name = mutated.__class__.__name__
            
            if class_name == "CodingAgent" and random.random() < self.mutation_rate:
                # Mutasi atribut CodingAgent
                if hasattr(mutated, "code_style") and random.random() < self.tool_mutation_rate:
                    styles = ["clean", "verbose", "compact"]
                    mutated.code_style = random.choice(styles)
                
                if hasattr(mutated, "preferred_language") and random.random() < self.tool_mutation_rate:
                    languages = ["python", "javascript", "java", "c++", "rust"]
                    mutated.preferred_language = random.choice(languages)
            
            elif class_name == "ProblemSolvingAgent" and random.random() < self.mutation_rate:
                # Mutasi atribut ProblemSolvingAgent
                if hasattr(mutated, "problem_types") and random.random() < self.tool_mutation_rate:
                    problem_types = ["optimization", "search", "planning", "classification"]
                    
                    # Pilih operasi
                    op = random.choice(["add", "remove", "replace"])
                    
                    if op == "add" and len(mutated.problem_types) < len(problem_types):
                        # Tambah tipe masalah
                        available = [pt for pt in problem_types if pt not in mutated.problem_types]
                        if available:
                            mutated.problem_types.append(random.choice(available))
                    
                    elif op == "remove" and len(mutated.problem_types) > 1:
                        # Hapus tipe masalah
                        mutated.problem_types.pop(random.randrange(len(mutated.problem_types)))
                    
                    elif op == "replace" and mutated.problem_types:
                        # Ganti tipe masalah
                        idx = random.randrange(len(mutated.problem_types))
                        available = [pt for pt in problem_types if pt != mutated.problem_types[idx]]
                        if available:
                            mutated.problem_types[idx] = random.choice(available)
            
            elif class_name == "MetaAgent" and random.random() < self.mutation_rate:
                # Mutasi atribut MetaAgent
                if hasattr(mutated, "agent_types") and random.random() < self.tool_mutation_rate:
                    agent_types = ["BaseAgent", "CodingAgent", "ProblemSolvingAgent", "MetaAgent"]
                    
                    # Pilih operasi
                    op = random.choice(["add", "remove", "replace"])
                    
                    if op == "add" and len(mutated.agent_types) < len(agent_types):
                        # Tambah tipe agen
                        available = [at for at in agent_types if at not in mutated.agent_types]
                        if available:
                            mutated.agent_types.append(random.choice(available))
                    
                    elif op == "remove" and len(mutated.agent_types) > 1:
                        # Hapus tipe agen
                        mutated.agent_types.pop(random.randrange(len(mutated.agent_types)))
                    
                    elif op == "replace" and mutated.agent_types:
                        # Ganti tipe agen
                        idx = random.randrange(len(mutated.agent_types))
                        available = [at for at in agent_types if at != mutated.agent_types[idx]]
                        if available:
                            mutated.agent_types[idx] = random.choice(available)
        
        return mutated


class FunctionalMutation(MutationOperator[T]):
    """
    Operator mutasi fungsional.
    
    Operator ini memutasi perilaku fungsional individu, seperti mengubah
    implementasi metode atau strategi pemecahan masalah.
    """
    
    def __init__(self, mutation_rate: float = 0.1, method_mutation_rate: float = 0.2,
                method_templates: Optional[Dict[str, List[str]]] = None):
        """
        Inisialisasi operator mutasi fungsional.
        
        Args:
            mutation_rate: Probabilitas mutasi
            method_mutation_rate: Probabilitas mutasi metode
            method_templates: Template untuk implementasi metode
        """
        super().__init__(mutation_rate)
        self.method_mutation_rate = method_mutation_rate
        
        # Template default jika tidak diberikan
        if method_templates is None:
            method_templates = {
                "solve": [
                    "# Implementasi dasar: pilih alat secara acak dan gunakan\n"
                    "if not self.tools:\n"
                    "    return None\n"
                    "selected_tool = random.choice(self.tools)\n"
                    "try:\n"
                    "    result = selected_tool(problem)\n"
                    "    self._add_to_memory({'problem': problem, 'tool': selected_tool.name, 'result': result})\n"
                    "    return result\n"
                    "except Exception as e:\n"
                    "    self._add_to_memory({'problem': problem, 'tool': selected_tool.name, 'error': str(e)})\n"
                    "    return None",
                    
                    "# Implementasi dengan eksplorasi: pilih alat berdasarkan eksplorasi\n"
                    "if not self.tools:\n"
                    "    return None\n"
                    "if random.random() < self.exploration_rate:\n"
                    "    selected_tool = random.choice(self.tools)\n"
                    "else:\n"
                    "    # Pilih alat berdasarkan performa sebelumnya\n"
                    "    tool_performance = {}\n"
                    "    for entry in self.memory:\n"
                    "        tool_name = entry.get('tool')\n"
                    "        if tool_name and 'error' not in entry:\n"
                    "            tool_performance[tool_name] = tool_performance.get(tool_name, 0) + 1\n"
                    "    if tool_performance:\n"
                    "        best_tool_name = max(tool_performance.items(), key=lambda x: x[1])[0]\n"
                    "        selected_tool = next((t for t in self.tools if t.name == best_tool_name), self.tools[0])\n"
                    "    else:\n"
                    "        selected_tool = self.tools[0]\n"
                    "try:\n"
                    "    result = selected_tool(problem)\n"
                    "    self._add_to_memory({'problem': problem, 'tool': selected_tool.name, 'result': result})\n"
                    "    return result\n"
                    "except Exception as e:\n"
                    "    self._add_to_memory({'problem': problem, 'tool': selected_tool.name, 'error': str(e)})\n"
                    "    return None",
                    
                    "# Implementasi dengan dekomposisi: dekomposisi masalah menjadi sub-masalah\n"
                    "if not self.tools:\n"
                    "    return None\n"
                    "# Coba dekomposisi masalah jika kompleks\n"
                    "if isinstance(problem, dict) and len(problem) > 3:\n"
                    "    sub_problems = []\n"
                    "    for key, value in problem.items():\n"
                    "        sub_problems.append({key: value})\n"
                    "    results = []\n"
                    "    for sub_problem in sub_problems:\n"
                    "        # Pilih alat yang sesuai untuk sub-masalah\n"
                    "        selected_tool = random.choice(self.tools)\n"
                    "        try:\n"
                    "            result = selected_tool(sub_problem)\n"
                    "            results.append(result)\n"
                    "        except Exception:\n"
                    "            pass\n"
                    "    if results:\n"
                    "        return results\n"
                    "# Jika dekomposisi gagal, gunakan pendekatan dasar\n"
                    "selected_tool = random.choice(self.tools)\n"
                    "try:\n"
                    "    result = selected_tool(problem)\n"
                    "    self._add_to_memory({'problem': problem, 'tool': selected_tool.name, 'result': result})\n"
                    "    return result\n"
                    "except Exception as e:\n"
                    "    self._add_to_memory({'problem': problem, 'tool': selected_tool.name, 'error': str(e)})\n"
                    "    return None"
                ]
            }
        
        self.method_templates = method_templates
    
    def mutate(self, individual: T) -> T:
        """
        Mutasi fungsional individu.
        
        Args:
            individual: Individu yang akan dimutasi
            
        Returns:
            Individu yang telah dimutasi
        """
        # Buat salinan individu
        mutated = copy.deepcopy(individual)
        
        # Mutasi metode dengan probabilitas mutation_rate
        if random.random() < self.mutation_rate:
            # Pilih metode untuk dimutasi
            method_name = random.choice(list(self.method_templates.keys()))
            
            if hasattr(mutated, method_name) and callable(getattr(mutated, method_name)):
                # Dapatkan template untuk metode ini
                templates = self.method_templates[method_name]
                
                if templates:
                    # Pilih template acak
                    template = random.choice(templates)
                    
                    # Dapatkan metode asli
                    original_method = getattr(mutated, method_name)
                    
                    # Dapatkan signature metode
                    sig = inspect.signature(original_method)
                    params = [p for p in sig.parameters.values()]
                    
                    # Buat metode baru
                    param_str = ", ".join(str(p) for p in params)
                    method_code = f"def {method_name}({param_str}):\n"
                    
                    # Tambahkan docstring jika ada
                    if original_method.__doc__:
                        method_code += f'    """{original_method.__doc__}"""\n'
                    
                    # Tambahkan implementasi
                    for line in template.split("\n"):
                        method_code += f"    {line}\n"
                    
                    # Kompilasi metode baru
                    namespace = {}
                    exec(method_code, globals(), namespace)
                    
                    # Tetapkan metode baru ke individu
                    setattr(mutated.__class__, method_name, namespace[method_name])
        
        return mutated


class AdaptiveMutation(MutationOperator[T]):
    """
    Operator mutasi adaptif.
    
    Operator ini menyesuaikan kekuatan mutasi berdasarkan kemajuan evolusi.
    """
    
    def __init__(self, initial_rate: float = 0.1, min_rate: float = 0.01, max_rate: float = 0.5,
                adaptation_rate: float = 0.1, operators: Optional[List[MutationOperator[T]]] = None):
        """
        Inisialisasi operator mutasi adaptif.
        
        Args:
            initial_rate: Tingkat mutasi awal
            min_rate: Tingkat mutasi minimum
            max_rate: Tingkat mutasi maksimum
            adaptation_rate: Tingkat adaptasi
            operators: Daftar operator mutasi yang digunakan
        """
        super().__init__(initial_rate)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adaptation_rate = adaptation_rate
        self.success_history = []  # Riwayat keberhasilan mutasi
        
        # Operator default jika tidak diberikan
        if operators is None:
            operators = [
                ParameterMutation(mutation_rate=initial_rate),
                StructuralMutation(mutation_rate=initial_rate),
                FunctionalMutation(mutation_rate=initial_rate)
            ]
        
        self.operators = operators
    
    def update_rate(self, success: bool):
        """
        Perbarui tingkat mutasi berdasarkan keberhasilan.
        
        Args:
            success: Apakah mutasi berhasil meningkatkan fitness
        """
        # Tambahkan hasil ke riwayat
        self.success_history.append(1 if success else 0)
        
        # Batasi panjang riwayat
        if len(self.success_history) > 10:
            self.success_history.pop(0)
        
        # Hitung tingkat keberhasilan
        if self.success_history:
            success_rate = sum(self.success_history) / len(self.success_history)
            
            # Sesuaikan tingkat mutasi
            if success_rate > 0.2:
                # Jika tingkat keberhasilan tinggi, kurangi tingkat mutasi
                self.mutation_rate = max(self.min_rate, 
                                        self.mutation_rate * (1 - self.adaptation_rate))
            else:
                # Jika tingkat keberhasilan rendah, tingkatkan tingkat mutasi
                self.mutation_rate = min(self.max_rate, 
                                        self.mutation_rate * (1 + self.adaptation_rate))
            
            # Perbarui tingkat mutasi operator
            for op in self.operators:
                op.mutation_rate = self.mutation_rate
    
    def mutate(self, individual: T) -> T:
        """
        Mutasi individu menggunakan operator yang dipilih secara adaptif.
        
        Args:
            individual: Individu yang akan dimutasi
            
        Returns:
            Individu yang telah dimutasi
        """
        # Pilih operator secara acak
        operator = random.choice(self.operators)
        
        # Mutasi individu
        return operator.mutate(individual)


class SelfAdaptiveMutation(MutationOperator[T]):
    """
    Operator mutasi self-adaptif.
    
    Operator ini menyertakan parameter mutasi dalam genom individu,
    sehingga parameter tersebut dapat berevolusi bersama individu.
    """
    
    def __init__(self, operators: Optional[List[MutationOperator[T]]] = None,
                initial_weights: Optional[List[float]] = None):
        """
        Inisialisasi operator mutasi self-adaptif.
        
        Args:
            operators: Daftar operator mutasi yang digunakan
            initial_weights: Bobot awal untuk setiap operator
        """
        super().__init__(0.1)  # Tingkat mutasi tidak digunakan langsung
        
        # Operator default jika tidak diberikan
        if operators is None:
            operators = [
                ParameterMutation(),
                StructuralMutation(),
                FunctionalMutation()
            ]
        
        self.operators = operators
        
        # Bobot awal jika tidak diberikan
        if initial_weights is None:
            initial_weights = [1.0 / len(operators)] * len(operators)
        
        self.initial_weights = initial_weights
    
    def _get_weights(self, individual: T) -> List[float]:
        """
        Dapatkan bobot operator dari individu.
        
        Args:
            individual: Individu
            
        Returns:
            Bobot untuk setiap operator
        """
        # Periksa apakah individu memiliki atribut mutation_weights
        if hasattr(individual, "mutation_weights"):
            return individual.mutation_weights
        else:
            # Jika tidak, gunakan bobot awal dan tambahkan atribut
            individual.mutation_weights = self.initial_weights.copy()
            return individual.mutation_weights
    
    def _mutate_weights(self, weights: List[float]) -> List[float]:
        """
        Mutasi bobot operator.
        
        Args:
            weights: Bobot saat ini
            
        Returns:
            Bobot yang telah dimutasi
        """
        # Buat salinan bobot
        new_weights = weights.copy()
        
        # Mutasi setiap bobot
        for i in range(len(new_weights)):
            # Tambahkan noise Gaussian
            new_weights[i] += random.gauss(0, 0.1)
            
            # Batasi nilai
            new_weights[i] = max(0.01, new_weights[i])
        
        # Normalisasi bobot
        total = sum(new_weights)
        new_weights = [w / total for w in new_weights]
        
        return new_weights
    
    def mutate(self, individual: T) -> T:
        """
        Mutasi individu menggunakan operator yang dipilih secara self-adaptif.
        
        Args:
            individual: Individu yang akan dimutasi
            
        Returns:
            Individu yang telah dimutasi
        """
        # Dapatkan bobot operator
        weights = self._get_weights(individual)
        
        # Mutasi bobot
        new_weights = self._mutate_weights(weights)
        
        # Buat salinan individu
        mutated = copy.deepcopy(individual)
        
        # Tetapkan bobot baru
        mutated.mutation_weights = new_weights
        
        # Pilih operator berdasarkan bobot
        operator_idx = random.choices(range(len(self.operators)), weights=new_weights, k=1)[0]
        operator = self.operators[operator_idx]
        
        # Mutasi individu
        mutated = operator.mutate(mutated)
        
        return mutated