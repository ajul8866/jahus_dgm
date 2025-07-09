"""
Utilitas evaluasi untuk Darwin-Gödel Machine.
"""

import time
import random
import json
import os
from typing import Dict, Any, List, Optional, Callable, Union, Tuple

from simple_dgm.agents.base_agent import BaseAgent


def evaluate_agent(agent: BaseAgent, task: Any) -> float:
    """
    Evaluasi agen pada tugas tertentu.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Skor evaluasi (0.0 - 1.0)
    """
    if isinstance(task, dict) and "type" in task:
        task_type = task["type"]
        
        if task_type == "coding":
            return evaluate_coding_agent(agent, task)
        elif task_type == "problem_solving":
            return evaluate_problem_solving_agent(agent, task)
        elif task_type == "meta":
            return evaluate_meta_agent(agent, task)
        else:
            return evaluate_general_agent(agent, task)
    else:
        return evaluate_general_agent(agent, task)


def evaluate_general_agent(agent: BaseAgent, task: Any) -> float:
    """
    Evaluasi agen umum.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Skor evaluasi (0.0 - 1.0)
    """
    # Implementasi sederhana: evaluasi berdasarkan jumlah alat
    tool_score = min(1.0, len(agent.tools) / 10.0)
    
    # Evaluasi berdasarkan parameter agen
    param_score = (
        (0.5 - abs(0.5 - agent.learning_rate * 10)) / 0.5 +  # Optimal learning_rate sekitar 0.05
        (0.5 - abs(0.5 - agent.exploration_rate * 2)) / 0.5   # Optimal exploration_rate sekitar 0.25
    ) / 2.0
    
    # Evaluasi berdasarkan kapasitas memori
    memory_score = min(1.0, agent.memory_capacity / 20.0)
    
    # Gabungkan skor
    return (tool_score + param_score + memory_score) / 3.0


def evaluate_coding_agent(agent: BaseAgent, task: Dict[str, Any]) -> float:
    """
    Evaluasi agen pengkodean.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas pengkodean
        
    Returns:
        Skor evaluasi (0.0 - 1.0)
    """
    from simple_dgm.agents.coding_agent import CodingAgent
    
    # Periksa apakah agen adalah CodingAgent
    if not isinstance(agent, CodingAgent):
        return 0.1  # Skor rendah untuk agen non-coding
    
    # Dapatkan subtask
    subtask = task.get("subtask", "analyze")
    code = task.get("code", "")
    
    if not code:
        return 0.0
    
    try:
        if subtask == "analyze":
            # Evaluasi kemampuan analisis kode
            result = agent._analyze_code(code)
            
            # Periksa apakah hasil analisis valid
            if not isinstance(result, dict) or "complexity" not in result or "bugs" not in result:
                return 0.2
            
            # Skor berdasarkan kelengkapan analisis
            return min(1.0, (len(result["complexity"]) + len(result["bugs"])) / 10.0)
        
        elif subtask == "fix_bugs":
            # Evaluasi kemampuan memperbaiki bug
            bugs = agent._analyze_code(code)["bugs"]
            fixed_code = agent._fix_bugs(code, bugs)
            
            # Periksa apakah kode yang diperbaiki berbeda dari kode asli
            if fixed_code == code:
                return 0.3
            
            # Analisis kode yang diperbaiki
            new_bugs = agent._analyze_code(fixed_code)["bugs"]
            
            # Skor berdasarkan pengurangan jumlah bug
            if not bugs:
                return 0.5  # Tidak ada bug untuk diperbaiki
            
            return min(1.0, max(0.0, 1.0 - len(new_bugs) / len(bugs)))
        
        elif subtask == "optimize":
            # Evaluasi kemampuan mengoptimalkan kode
            optimized_code = agent._optimize_code(code)
            
            # Periksa apakah kode yang dioptimalkan berbeda dari kode asli
            if optimized_code == code:
                return 0.3
            
            # Analisis kompleksitas kode yang dioptimalkan
            original_complexity = agent._analyze_code(code)["complexity"]
            optimized_complexity = agent._analyze_code(optimized_code)["complexity"]
            
            # Skor berdasarkan pengurangan kompleksitas
            if "cyclomatic_complexity" in original_complexity and "cyclomatic_complexity" in optimized_complexity:
                original_cc = original_complexity["cyclomatic_complexity"]
                optimized_cc = optimized_complexity["cyclomatic_complexity"]
                
                if original_cc <= optimized_cc:
                    return 0.4  # Tidak ada pengurangan kompleksitas
                
                return min(1.0, 0.5 + 0.5 * (original_cc - optimized_cc) / original_cc)
            
            return 0.5
        
        elif subtask == "generate":
            # Evaluasi kemampuan menghasilkan kode
            spec = task.get("spec", {})
            generated_code = agent._generate_code(spec)
            
            # Periksa apakah kode yang dihasilkan tidak kosong
            if not generated_code:
                return 0.0
            
            # Analisis kode yang dihasilkan
            analysis = agent._analyze_code(generated_code)
            
            # Periksa apakah kode valid
            if "error" in analysis.get("complexity", {}):
                return 0.2
            
            # Skor berdasarkan kompleksitas dan jumlah bug
            complexity_score = min(1.0, analysis["complexity"].get("loc", 0) / 50.0)
            bug_score = max(0.0, 1.0 - min(1.0, len(analysis["bugs"]) / 5.0))
            
            return (complexity_score + bug_score) / 2.0
        
        else:
            return 0.5  # Subtask tidak dikenal
    
    except Exception as e:
        # Penanganan kesalahan
        return 0.1


def evaluate_problem_solving_agent(agent: BaseAgent, task: Dict[str, Any]) -> float:
    """
    Evaluasi agen pemecahan masalah.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas pemecahan masalah
        
    Returns:
        Skor evaluasi (0.0 - 1.0)
    """
    from simple_dgm.agents.problem_solving_agent import ProblemSolvingAgent
    
    # Periksa apakah agen adalah ProblemSolvingAgent
    if not isinstance(agent, ProblemSolvingAgent):
        return 0.1  # Skor rendah untuk agen non-problem-solving
    
    # Dapatkan subtask
    subtask = task.get("subtask", "")
    problem = task.get("problem", {})
    
    if not problem:
        return 0.0
    
    try:
        # Selesaikan masalah
        start_time = time.time()
        solution = agent.solve(problem)
        solve_time = time.time() - start_time
        
        # Periksa apakah solusi valid
        if solution is None:
            return 0.0
        
        # Evaluasi solusi
        solution_score = agent._evaluate_solution(problem, solution)
        
        # Skor berdasarkan kualitas solusi dan waktu penyelesaian
        time_score = max(0.0, 1.0 - min(1.0, solve_time / agent.timeout))
        
        # Gabungkan skor
        return 0.7 * solution_score + 0.3 * time_score
    
    except Exception as e:
        # Penanganan kesalahan
        return 0.1


def evaluate_meta_agent(agent: BaseAgent, task: Dict[str, Any]) -> float:
    """
    Evaluasi agen meta.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas meta
        
    Returns:
        Skor evaluasi (0.0 - 1.0)
    """
    from simple_dgm.agents.meta_agent import MetaAgent
    
    # Periksa apakah agen adalah MetaAgent
    if not isinstance(agent, MetaAgent):
        return 0.1  # Skor rendah untuk agen non-meta
    
    # Dapatkan subtask
    subtask = task.get("subtask", "")
    
    if subtask == "create_agent":
        # Evaluasi kemampuan membuat agen
        agent_type = task.get("agent_type", "BaseAgent")
        params = task.get("params", {})
        
        try:
            # Buat agen
            new_agent = agent._create_agent(agent_type, params)
            
            # Periksa apakah agen berhasil dibuat
            if new_agent is None:
                return 0.0
            
            # Skor berdasarkan jenis agen dan jumlah alat
            type_score = 0.5 if agent_type == "BaseAgent" else (0.8 if agent_type == "CodingAgent" else 1.0)
            tool_score = min(1.0, len(new_agent.tools) / 5.0)
            
            return (type_score + tool_score) / 2.0
        
        except Exception as e:
            # Penanganan kesalahan
            return 0.1
    
    elif subtask == "modify_agent":
        # Evaluasi kemampuan memodifikasi agen
        from simple_dgm.agents.base_agent import BaseAgent
        
        # Buat agen sederhana untuk dimodifikasi
        test_agent = BaseAgent()
        modifications = task.get("modifications", {})
        
        try:
            # Modifikasi agen
            modified_agent = agent._modify_agent(test_agent, modifications)
            
            # Periksa apakah agen berhasil dimodifikasi
            if modified_agent is None:
                return 0.0
            
            # Skor berdasarkan jumlah modifikasi yang berhasil diterapkan
            param_mods = modifications.get("params", {})
            tool_mods = modifications.get("tools", [])
            
            param_score = sum(1 for param, value in param_mods.items() if hasattr(modified_agent, param) and getattr(modified_agent, param) == value)
            param_score = min(1.0, param_score / max(1, len(param_mods)))
            
            tool_score = min(1.0, len(modified_agent.tools) / max(1, len(tool_mods)))
            
            return (param_score + tool_score) / 2.0
        
        except Exception as e:
            # Penanganan kesalahan
            return 0.1
    
    elif subtask == "generate_agent_code":
        # Evaluasi kemampuan menghasilkan kode agen
        spec = task.get("spec", {})
        
        try:
            # Hasilkan kode agen
            code = agent._generate_agent_code(spec)
            
            # Periksa apakah kode valid
            if not code:
                return 0.0
            
            # Skor berdasarkan panjang kode dan kelengkapan
            length_score = min(1.0, len(code) / 1000.0)
            
            # Periksa kelengkapan kode
            completeness_score = 0.0
            required_elements = [
                "class", "def __init__", "def mutate", "def to_dict", "def from_dict"
            ]
            
            for element in required_elements:
                if element in code:
                    completeness_score += 1.0 / len(required_elements)
            
            return (length_score + completeness_score) / 2.0
        
        except Exception as e:
            # Penanganan kesalahan
            return 0.1
    
    else:
        # Subtask tidak dikenal
        return 0.5


def create_evaluation_task(task_type: str, difficulty: float = 0.5) -> Dict[str, Any]:
    """
    Buat tugas evaluasi.
    
    Args:
        task_type: Jenis tugas ("coding", "problem_solving", "meta")
        difficulty: Tingkat kesulitan (0.0 - 1.0)
        
    Returns:
        Tugas evaluasi
    """
    if task_type == "coding":
        return create_coding_task(difficulty)
    elif task_type == "problem_solving":
        return create_problem_solving_task(difficulty)
    elif task_type == "meta":
        return create_meta_task(difficulty)
    else:
        return {"type": "unknown"}


def create_coding_task(difficulty: float = 0.5) -> Dict[str, Any]:
    """
    Buat tugas pengkodean.
    
    Args:
        difficulty: Tingkat kesulitan (0.0 - 1.0)
        
    Returns:
        Tugas pengkodean
    """
    subtasks = ["analyze", "fix_bugs", "optimize", "generate"]
    subtask = random.choice(subtasks)
    
    if subtask == "analyze":
        # Buat kode untuk dianalisis
        code_length = int(50 + difficulty * 150)
        num_bugs = int(1 + difficulty * 5)
        
        code = f"def example_function(x, y):\n"
        code += f"    result = 0\n"
        
        # Tambahkan beberapa operasi
        for i in range(int(3 + difficulty * 7)):
            code += f"    result += x ** {i} * y\n"
        
        # Tambahkan beberapa bug
        for i in range(num_bugs):
            if random.random() < 0.5:
                # Bug variabel tidak didefinisikan
                code += f"    print(undefined_var_{i})\n"
            else:
                # Bug except tanpa spesifikasi
                code += f"    try:\n"
                code += f"        result /= (x - {i})\n"
                code += f"    except:\n"
                code += f"        pass\n"
        
        code += f"    return result\n"
        
        return {
            "type": "coding",
            "subtask": "analyze",
            "code": code,
            "difficulty": difficulty
        }
    
    elif subtask == "fix_bugs":
        # Buat kode dengan bug untuk diperbaiki
        code = f"def buggy_function(data):\n"
        code += f"    result = []\n"
        code += f"    for i in range(len(data)):\n"
        code += f"        try:\n"
        code += f"            value = data[i] / 0  # Bug: division by zero\n"
        code += f"            result.append(value)\n"
        code += f"        except:  # Bug: bare except\n"
        code += f"            pass\n"
        code += f"    \n"
        code += f"    # Bug: undefined variable\n"
        code += f"    for item in items:\n"
        code += f"        result.append(item)\n"
        code += f"    \n"
        code += f"    return result\n"
        
        return {
            "type": "coding",
            "subtask": "fix_bugs",
            "code": code,
            "difficulty": difficulty
        }
    
    elif subtask == "optimize":
        # Buat kode untuk dioptimalkan
        code = f"def inefficient_function(data):\n"
        code += f"    result = []\n"
        code += f"    # Inefficient: use list comprehension instead\n"
        code += f"    for i in range(len(data)):\n"
        code += f"        result.append(data[i] * 2)\n"
        code += f"    \n"
        code += f"    # Inefficient: use 'in' operator instead\n"
        code += f"    found = False\n"
        code += f"    for item in result:\n"
        code += f"        if item == 42:\n"
        code += f"            found = True\n"
        code += f"            break\n"
        code += f"    \n"
        code += f"    # Inefficient: use sum() instead\n"
        code += f"    total = 0\n"
        code += f"    for item in result:\n"
        code += f"        total += item\n"
        code += f"    \n"
        code += f"    return total, found\n"
        
        return {
            "type": "coding",
            "subtask": "optimize",
            "code": code,
            "difficulty": difficulty
        }
    
    elif subtask == "generate":
        # Buat spesifikasi untuk menghasilkan kode
        class_name = "DataProcessor"
        methods = ["process_data", "filter_data", "aggregate_data"]
        
        spec = {
            "imports": [
                {"import": ["math", "random"]},
                {"from": "typing", "import": ["List", "Dict", "Any"]}
            ],
            "classes": [
                {
                    "name": class_name,
                    "methods": [
                        {
                            "name": "__init__",
                            "params": ["data_source", "config"],
                            "body": [
                                "self.data_source = data_source",
                                "self.config = config",
                                "self.processed_data = None"
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Tambahkan metode berdasarkan tingkat kesulitan
        num_methods = int(1 + difficulty * len(methods))
        for i in range(min(num_methods, len(methods))):
            method_name = methods[i]
            params = ["data"] if i > 0 else []
            
            body = [
                f"# Process the data",
                f"result = []",
                f"for item in {'data' if i > 0 else 'self.data_source'}:",
                f"    if isinstance(item, dict):",
                f"        result.append(item.get('value', 0) * 2)",
                f"    else:",
                f"        result.append(item)",
                f"return result"
            ]
            
            spec["classes"][0]["methods"].append({
                "name": method_name,
                "params": params,
                "body": body
            })
        
        return {
            "type": "coding",
            "subtask": "generate",
            "spec": spec,
            "difficulty": difficulty
        }
    
    else:
        return {"type": "coding", "subtask": "unknown"}


def create_problem_solving_task(difficulty: float = 0.5) -> Dict[str, Any]:
    """
    Buat tugas pemecahan masalah.
    
    Args:
        difficulty: Tingkat kesulitan (0.0 - 1.0)
        
    Returns:
        Tugas pemecahan masalah
    """
    problem_types = ["optimization", "search", "planning", "classification"]
    problem_type = random.choice(problem_types)
    
    if problem_type == "optimization":
        # Buat masalah optimasi
        num_variables = int(2 + difficulty * 8)
        num_constraints = int(1 + difficulty * 5)
        
        variables = [f"x{i}" for i in range(num_variables)]
        
        constraints = []
        for i in range(num_constraints):
            # Buat batasan ketidaksetaraan acak
            constraint = {
                "type": "inequality",
                "lhs": lambda x, i=i: sum(x.get(f"x{j}", 0) * random.uniform(-1, 1) for j in range(num_variables)),
                "rhs": random.uniform(-10, 10),
                "is_linear": True,
                "variables": variables
            }
            constraints.append(constraint)
        
        # Buat fungsi objektif
        objective = {
            "type": "minimize" if random.random() < 0.5 else "maximize",
            "function": lambda x: sum(x.get(f"x{i}", 0) ** 2 for i in range(num_variables))
        }
        
        return {
            "type": "problem_solving",
            "subtask": "solve",
            "problem": {
                "type": "optimization",
                "variables": variables,
                "constraints": constraints,
                "objective": objective
            },
            "difficulty": difficulty
        }
    
    elif problem_type == "search":
        # Buat masalah pencarian
        space_size = int(10 ** (1 + difficulty * 4))
        
        # Buat ruang pencarian
        space = {
            "size": space_size,
            "is_discrete": random.random() < 0.5,
            "sample": lambda: random.uniform(0, 100) if space["is_discrete"] else random.randint(0, 100)
        }
        
        # Buat target
        target = random.uniform(0, 100) if space["is_discrete"] else random.randint(0, 100)
        
        # Buat heuristik
        heuristic = lambda x: -abs(x - target)
        
        return {
            "type": "problem_solving",
            "subtask": "solve",
            "problem": {
                "type": "search",
                "space": space,
                "target": target,
                "heuristic": heuristic
            },
            "difficulty": difficulty
        }
    
    elif problem_type == "planning":
        # Buat masalah perencanaan
        num_steps = int(3 + difficulty * 7)
        
        # Buat keadaan awal dan tujuan
        initial_state = {
            "position": 0,
            "has_key": False,
            "door_open": False
        }
        
        goal_state = {
            "position": num_steps,
            "has_key": True,
            "door_open": True
        }
        
        # Buat langkah-langkah
        steps = []
        for i in range(num_steps):
            if i == 1:
                # Langkah untuk mengambil kunci
                steps.append({
                    "name": f"pick_up_key",
                    "preconditions": {"position": 1, "has_key": False},
                    "effects": {"has_key": True}
                })
            elif i == num_steps - 2:
                # Langkah untuk membuka pintu
                steps.append({
                    "name": f"open_door",
                    "preconditions": {"position": num_steps - 2, "has_key": True, "door_open": False},
                    "effects": {"door_open": True}
                })
            else:
                # Langkah untuk bergerak
                steps.append({
                    "name": f"move_to_{i+1}",
                    "preconditions": {"position": i},
                    "effects": {"position": i + 1}
                })
        
        return {
            "type": "problem_solving",
            "subtask": "solve",
            "problem": {
                "type": "planning",
                "initial_state": initial_state,
                "goal_state": goal_state,
                "steps": steps
            },
            "difficulty": difficulty
        }
    
    elif problem_type == "classification":
        # Buat masalah klasifikasi
        num_features = int(2 + difficulty * 8)
        num_classes = int(2 + difficulty * 3)
        num_samples = int(10 + difficulty * 90)
        
        features = [f"feature_{i}" for i in range(num_features)]
        classes = [f"class_{i}" for i in range(num_classes)]
        
        # Buat data pelatihan
        training_data = []
        for i in range(num_samples):
            sample = {f: random.uniform(0, 1) for f in features}
            sample["class"] = random.choice(classes)
            training_data.append(sample)
        
        # Buat data pengujian
        test_data = []
        for i in range(int(num_samples * 0.2)):
            sample = {f: random.uniform(0, 1) for f in features}
            test_data.append(sample)
        
        return {
            "type": "problem_solving",
            "subtask": "solve",
            "problem": {
                "type": "classification",
                "features": features,
                "classes": classes,
                "training_data": training_data,
                "test_data": test_data
            },
            "difficulty": difficulty
        }
    
    else:
        return {"type": "problem_solving", "subtask": "unknown"}


def create_meta_task(difficulty: float = 0.5) -> Dict[str, Any]:
    """
    Buat tugas meta.
    
    Args:
        difficulty: Tingkat kesulitan (0.0 - 1.0)
        
    Returns:
        Tugas meta
    """
    subtasks = ["create_agent", "modify_agent", "generate_agent_code"]
    subtask = random.choice(subtasks)
    
    if subtask == "create_agent":
        # Buat tugas pembuatan agen
        agent_types = ["BaseAgent", "CodingAgent", "ProblemSolvingAgent"]
        agent_type = agent_types[int(difficulty * (len(agent_types) - 0.01))]
        
        params = {}
        if agent_type == "BaseAgent":
            params = {
                "memory_capacity": int(5 + difficulty * 15),
                "learning_rate": 0.01 + difficulty * 0.09,
                "exploration_rate": 0.05 + difficulty * 0.25
            }
        elif agent_type == "CodingAgent":
            params = {
                "memory_capacity": int(5 + difficulty * 15),
                "learning_rate": 0.01 + difficulty * 0.09,
                "exploration_rate": 0.05 + difficulty * 0.25,
                "code_style": random.choice(["clean", "verbose", "compact"]),
                "preferred_language": random.choice(["python", "javascript", "java"])
            }
        elif agent_type == "ProblemSolvingAgent":
            params = {
                "memory_capacity": int(5 + difficulty * 15),
                "learning_rate": 0.01 + difficulty * 0.09,
                "exploration_rate": 0.05 + difficulty * 0.25,
                "problem_types": random.sample(["optimization", "search", "planning", "classification"], int(1 + difficulty * 3)),
                "max_iterations": int(50 + difficulty * 150),
                "timeout": 10.0 + difficulty * 50.0
            }
        
        return {
            "type": "meta",
            "subtask": "create_agent",
            "agent_type": agent_type,
            "params": params,
            "difficulty": difficulty
        }
    
    elif subtask == "modify_agent":
        # Buat tugas modifikasi agen
        num_param_mods = int(1 + difficulty * 3)
        num_tool_mods = int(1 + difficulty * 2)
        
        # Buat modifikasi parameter
        param_mods = {}
        for i in range(num_param_mods):
            param_name = random.choice(["memory_capacity", "learning_rate", "exploration_rate"])
            if param_name == "memory_capacity":
                param_mods[param_name] = int(5 + difficulty * 15)
            elif param_name == "learning_rate":
                param_mods[param_name] = 0.01 + difficulty * 0.09
            elif param_name == "exploration_rate":
                param_mods[param_name] = 0.05 + difficulty * 0.25
        
        # Buat modifikasi alat
        tool_mods = []
        for i in range(num_tool_mods):
            action = random.choice(["add", "remove", "modify"])
            
            if action == "add":
                tool_mods.append({
                    "action": "add",
                    "tool": {
                        "name": f"new_tool_{i}",
                        "function": lambda x: x,
                        "description": f"New tool {i}"
                    }
                })
            elif action == "remove":
                tool_mods.append({
                    "action": "remove",
                    "name": f"tool_{i}"
                })
            elif action == "modify":
                tool_mods.append({
                    "action": "modify",
                    "name": f"tool_{i}",
                    "tool": {
                        "name": f"modified_tool_{i}",
                        "function": lambda x: x,
                        "description": f"Modified tool {i}"
                    }
                })
        
        return {
            "type": "meta",
            "subtask": "modify_agent",
            "modifications": {
                "params": param_mods,
                "tools": tool_mods
            },
            "difficulty": difficulty
        }
    
    elif subtask == "generate_agent_code":
        # Buat tugas pembuatan kode agen
        num_params = int(1 + difficulty * 4)
        num_methods = int(1 + difficulty * 3)
        
        # Buat spesifikasi agen
        params = []
        for i in range(num_params):
            param_type = random.choice(["str", "int", "float", "bool"])
            param_default = None
            
            if param_type == "str":
                param_default = f"default_{i}"
            elif param_type == "int":
                param_default = i * 10
            elif param_type == "float":
                param_default = i * 0.1
            elif param_type == "bool":
                param_default = random.choice([True, False])
            
            params.append({
                "name": f"param_{i}",
                "type": param_type,
                "default": param_default,
                "description": f"Parameter {i}"
            })
        
        # Buat metode
        methods = []
        for i in range(num_methods):
            method_params = []
            for j in range(int(1 + difficulty * 2)):
                method_params.append({
                    "name": f"arg_{j}",
                    "type": random.choice(["str", "int", "float", "bool", "Any"]),
                    "default": None,
                    "description": f"Argument {j}"
                })
            
            methods.append({
                "name": f"method_{i}",
                "params": method_params,
                "return": random.choice(["None", "bool", "int", "float", "str", "Dict[str, Any]", "List[Any]"]),
                "return_desc": f"Result of method {i}",
                "description": f"Method {i}",
                "body": ["pass"]
            })
        
        spec = {
            "name": f"Custom{int(difficulty * 100)}Agent",
            "base_class": random.choice(["BaseAgent", "CodingAgent", "ProblemSolvingAgent"]),
            "description": f"Custom agent with difficulty {difficulty}",
            "params": params,
            "methods": methods
        }
        
        return {
            "type": "meta",
            "subtask": "generate_agent_code",
            "spec": spec,
            "difficulty": difficulty
        }
    
    else:
        return {"type": "meta", "subtask": "unknown"}