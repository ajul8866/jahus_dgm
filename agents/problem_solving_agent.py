"""
Implementasi agen pemecahan masalah untuk Darwin-Gödel Machine.
"""

import random
import math
import time
from typing import Dict, Any, List, Optional, Callable, Union, Tuple

from simple_dgm.agents.base_agent import BaseAgent, Tool


class ProblemSolvingAgent(BaseAgent):
    """
    Agen khusus untuk pemecahan masalah umum.
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None, 
                 memory_capacity: int = 10,
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.1,
                 problem_types: Optional[List[str]] = None,
                 max_iterations: int = 100,
                 timeout: float = 30.0):
        """
        Inisialisasi agen pemecahan masalah.
        
        Args:
            tools: Daftar alat yang tersedia untuk agen
            memory_capacity: Kapasitas memori agen
            learning_rate: Tingkat pembelajaran agen
            exploration_rate: Tingkat eksplorasi agen
            problem_types: Jenis masalah yang dapat ditangani
            max_iterations: Jumlah maksimum iterasi untuk pemecahan masalah
            timeout: Batas waktu untuk pemecahan masalah (detik)
        """
        super().__init__(tools, memory_capacity, learning_rate, exploration_rate)
        
        self.problem_types = problem_types or ["optimization", "search", "planning", "classification"]
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.strategies = self._initialize_strategies()
        
        # Tambahkan alat khusus untuk pemecahan masalah
        self._add_problem_solving_tools()
        
        # Tambahkan metadata khusus
        self.metadata.update({
            "type": "ProblemSolvingAgent",
            "problem_types": self.problem_types,
            "max_iterations": max_iterations,
            "timeout": timeout
        })
    
    def _initialize_strategies(self) -> Dict[str, Callable]:
        """
        Inisialisasi strategi pemecahan masalah.
        
        Returns:
            Dictionary strategi pemecahan masalah
        """
        return {
            "optimization": self._solve_optimization,
            "search": self._solve_search,
            "planning": self._solve_planning,
            "classification": self._solve_classification
        }
    
    def _add_problem_solving_tools(self) -> None:
        """
        Tambahkan alat khusus untuk pemecahan masalah.
        """
        # Alat untuk dekomposisi masalah
        self.add_tool(Tool(
            name="decompose_problem",
            function=self._decompose_problem,
            description="Dekomposisi masalah menjadi sub-masalah yang lebih kecil"
        ))
        
        # Alat untuk analisis masalah
        self.add_tool(Tool(
            name="analyze_problem",
            function=self._analyze_problem,
            description="Analisis masalah untuk mengidentifikasi karakteristik kunci"
        ))
        
        # Alat untuk pemilihan strategi
        self.add_tool(Tool(
            name="select_strategy",
            function=self._select_strategy,
            description="Pilih strategi pemecahan masalah yang sesuai"
        ))
        
        # Alat untuk evaluasi solusi
        self.add_tool(Tool(
            name="evaluate_solution",
            function=self._evaluate_solution,
            description="Evaluasi kualitas solusi"
        ))
    
    def _decompose_problem(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dekomposisi masalah menjadi sub-masalah.
        
        Args:
            problem: Masalah yang akan didekomposisi
            
        Returns:
            Daftar sub-masalah
        """
        problem_type = problem.get("type", "")
        
        if problem_type == "optimization":
            # Dekomposisi masalah optimasi
            constraints = problem.get("constraints", [])
            variables = problem.get("variables", [])
            
            # Bagi variabel menjadi kelompok-kelompok
            var_groups = []
            group_size = max(1, len(variables) // 3)
            
            for i in range(0, len(variables), group_size):
                var_groups.append(variables[i:i+group_size])
            
            # Buat sub-masalah untuk setiap kelompok variabel
            sub_problems = []
            for i, var_group in enumerate(var_groups):
                sub_problems.append({
                    "type": "optimization",
                    "id": f"sub_{i}",
                    "variables": var_group,
                    "constraints": [c for c in constraints if any(v in c.get("variables", []) for v in var_group)],
                    "objective": problem.get("objective", {})
                })
            
            return sub_problems
        
        elif problem_type == "planning":
            # Dekomposisi masalah perencanaan
            steps = problem.get("steps", [])
            
            # Bagi langkah-langkah menjadi kelompok-kelompok
            step_groups = []
            group_size = max(1, len(steps) // 3)
            
            for i in range(0, len(steps), group_size):
                step_groups.append(steps[i:i+group_size])
            
            # Buat sub-masalah untuk setiap kelompok langkah
            sub_problems = []
            for i, step_group in enumerate(step_groups):
                sub_problems.append({
                    "type": "planning",
                    "id": f"sub_{i}",
                    "steps": step_group,
                    "initial_state": problem.get("initial_state", {}) if i == 0 else None,
                    "goal_state": problem.get("goal_state", {}) if i == len(step_groups) - 1 else None
                })
            
            return sub_problems
        
        else:
            # Untuk jenis masalah lain, kembalikan masalah asli
            return [problem]
    
    def _analyze_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis masalah.
        
        Args:
            problem: Masalah yang akan dianalisis
            
        Returns:
            Hasil analisis
        """
        problem_type = problem.get("type", "")
        analysis = {"type": problem_type}
        
        if problem_type == "optimization":
            # Analisis masalah optimasi
            variables = problem.get("variables", [])
            constraints = problem.get("constraints", [])
            
            analysis.update({
                "num_variables": len(variables),
                "num_constraints": len(constraints),
                "is_linear": all(c.get("is_linear", False) for c in constraints),
                "is_convex": all(c.get("is_convex", False) for c in constraints),
                "suggested_method": "linear_programming" if analysis.get("is_linear", False) else "nonlinear_optimization"
            })
        
        elif problem_type == "search":
            # Analisis masalah pencarian
            space = problem.get("space", {})
            
            analysis.update({
                "space_size": space.get("size", 0),
                "is_discrete": space.get("is_discrete", True),
                "has_heuristic": "heuristic" in problem,
                "suggested_method": "a_star" if analysis.get("has_heuristic", False) else "bfs"
            })
        
        elif problem_type == "planning":
            # Analisis masalah perencanaan
            steps = problem.get("steps", [])
            
            analysis.update({
                "num_steps": len(steps),
                "has_dependencies": any("dependencies" in step for step in steps),
                "suggested_method": "hierarchical_planning" if analysis.get("has_dependencies", False) else "sequential_planning"
            })
        
        elif problem_type == "classification":
            # Analisis masalah klasifikasi
            features = problem.get("features", [])
            classes = problem.get("classes", [])
            
            analysis.update({
                "num_features": len(features),
                "num_classes": len(classes),
                "is_binary": len(classes) == 2,
                "suggested_method": "decision_tree" if len(features) < 10 else "neural_network"
            })
        
        return analysis
    
    def _select_strategy(self, problem: Dict[str, Any]) -> str:
        """
        Pilih strategi pemecahan masalah.
        
        Args:
            problem: Masalah yang akan diselesaikan
            
        Returns:
            Nama strategi yang dipilih
        """
        # Analisis masalah
        analysis = self._analyze_problem(problem)
        problem_type = analysis["type"]
        
        # Pilih strategi berdasarkan analisis
        if problem_type in self.strategies:
            return problem_type
        else:
            # Default ke strategi acak
            return random.choice(list(self.strategies.keys()))
    
    def _evaluate_solution(self, problem: Dict[str, Any], solution: Any) -> float:
        """
        Evaluasi kualitas solusi.
        
        Args:
            problem: Masalah yang diselesaikan
            solution: Solusi yang akan dievaluasi
            
        Returns:
            Skor kualitas solusi (0.0 - 1.0)
        """
        problem_type = problem.get("type", "")
        
        if problem_type == "optimization":
            # Evaluasi solusi optimasi
            objective = problem.get("objective", {})
            constraints = problem.get("constraints", [])
            
            # Periksa apakah solusi memenuhi semua batasan
            constraints_satisfied = all(self._check_constraint(c, solution) for c in constraints)
            
            if not constraints_satisfied:
                return 0.0
            
            # Hitung nilai objektif
            if objective.get("type") == "minimize":
                value = objective.get("function", lambda x: 0)(solution)
                # Normalisasi nilai (asumsi nilai yang lebih rendah lebih baik)
                normalized_value = 1.0 / (1.0 + value)
                return normalized_value
            else:  # maximize
                value = objective.get("function", lambda x: 0)(solution)
                # Normalisasi nilai (asumsi nilai yang lebih tinggi lebih baik)
                normalized_value = min(1.0, value / 100.0)  # Asumsi nilai maksimum 100
                return normalized_value
        
        elif problem_type == "search":
            # Evaluasi solusi pencarian
            target = problem.get("target", None)
            
            if target is None:
                return 0.5  # Tidak ada target yang jelas
            
            # Periksa apakah solusi sama dengan target
            if solution == target:
                return 1.0
            
            # Hitung kesamaan (sederhana)
            similarity = sum(1 for a, b in zip(str(solution), str(target)) if a == b) / max(len(str(solution)), len(str(target)))
            return similarity
        
        elif problem_type == "planning":
            # Evaluasi solusi perencanaan
            goal_state = problem.get("goal_state", {})
            
            if not goal_state:
                return 0.5  # Tidak ada keadaan tujuan yang jelas
            
            # Periksa apakah solusi mencapai keadaan tujuan
            if isinstance(solution, dict) and all(solution.get(k) == v for k, v in goal_state.items()):
                return 1.0
            
            # Hitung kesamaan (sederhana)
            if isinstance(solution, dict):
                common_keys = set(solution.keys()) & set(goal_state.keys())
                if not common_keys:
                    return 0.0
                
                similarity = sum(1 for k in common_keys if solution[k] == goal_state[k]) / len(common_keys)
                return similarity
            
            return 0.0
        
        elif problem_type == "classification":
            # Evaluasi solusi klasifikasi
            true_labels = problem.get("true_labels", [])
            
            if not true_labels or not isinstance(solution, list):
                return 0.5  # Tidak ada label yang benar atau solusi bukan daftar
            
            # Hitung akurasi
            correct = sum(1 for pred, true in zip(solution, true_labels) if pred == true)
            accuracy = correct / len(true_labels)
            return accuracy
        
        else:
            # Default
            return 0.5
    
    def _check_constraint(self, constraint: Dict[str, Any], solution: Any) -> bool:
        """
        Periksa apakah solusi memenuhi batasan.
        
        Args:
            constraint: Batasan yang akan diperiksa
            solution: Solusi yang akan dievaluasi
            
        Returns:
            True jika solusi memenuhi batasan, False jika tidak
        """
        constraint_type = constraint.get("type", "")
        
        if constraint_type == "equality":
            # Batasan kesetaraan: lhs == rhs
            lhs = constraint.get("lhs", lambda x: 0)(solution)
            rhs = constraint.get("rhs", 0)
            return abs(lhs - rhs) < 1e-6
        
        elif constraint_type == "inequality":
            # Batasan ketidaksetaraan: lhs <= rhs
            lhs = constraint.get("lhs", lambda x: 0)(solution)
            rhs = constraint.get("rhs", 0)
            return lhs <= rhs
        
        elif constraint_type == "range":
            # Batasan rentang: min <= value <= max
            value = constraint.get("value", lambda x: 0)(solution)
            min_val = constraint.get("min", float("-inf"))
            max_val = constraint.get("max", float("inf"))
            return min_val <= value <= max_val
        
        else:
            # Default
            return True
    
    def _solve_optimization(self, problem: Dict[str, Any]) -> Any:
        """
        Selesaikan masalah optimasi.
        
        Args:
            problem: Masalah optimasi
            
        Returns:
            Solusi optimal
        """
        variables = problem.get("variables", [])
        constraints = problem.get("constraints", [])
        objective = problem.get("objective", {})
        
        if not variables:
            return None
        
        # Implementasi sederhana: hill climbing
        # Inisialisasi solusi acak
        solution = {var: random.uniform(0, 1) for var in variables}
        best_solution = solution.copy()
        best_value = self._evaluate_solution(problem, best_solution)
        
        # Iterasi untuk meningkatkan solusi
        start_time = time.time()
        for _ in range(self.max_iterations):
            if time.time() - start_time > self.timeout:
                break
            
            # Buat solusi baru dengan mengubah satu variabel
            new_solution = best_solution.copy()
            var_to_change = random.choice(variables)
            new_solution[var_to_change] += random.uniform(-0.1, 0.1)
            
            # Evaluasi solusi baru
            new_value = self._evaluate_solution(problem, new_solution)
            
            # Jika solusi baru lebih baik, perbarui solusi terbaik
            if new_value > best_value:
                best_solution = new_solution
                best_value = new_value
        
        return best_solution
    
    def _solve_search(self, problem: Dict[str, Any]) -> Any:
        """
        Selesaikan masalah pencarian.
        
        Args:
            problem: Masalah pencarian
            
        Returns:
            Hasil pencarian
        """
        space = problem.get("space", {})
        target = problem.get("target", None)
        heuristic = problem.get("heuristic", None)
        
        if not space:
            return None
        
        # Implementasi sederhana: pencarian acak
        best_item = None
        best_score = float("-inf")
        
        # Iterasi untuk menemukan item terbaik
        start_time = time.time()
        for _ in range(self.max_iterations):
            if time.time() - start_time > self.timeout:
                break
            
            # Pilih item acak dari ruang pencarian
            item = space.get("sample", lambda: random.random())()
            
            # Evaluasi item
            if heuristic:
                score = heuristic(item)
            else:
                # Jika tidak ada heuristik, gunakan kesamaan dengan target
                if target is not None:
                    score = -abs(item - target)  # Semakin dekat dengan target, semakin baik
                else:
                    score = 0
            
            # Jika item ini lebih baik, perbarui item terbaik
            if score > best_score:
                best_item = item
                best_score = score
                
                # Jika menemukan target, hentikan pencarian
                if target is not None and abs(best_item - target) < 1e-6:
                    break
        
        return best_item
    
    def _solve_planning(self, problem: Dict[str, Any]) -> Any:
        """
        Selesaikan masalah perencanaan.
        
        Args:
            problem: Masalah perencanaan
            
        Returns:
            Rencana solusi
        """
        steps = problem.get("steps", [])
        initial_state = problem.get("initial_state", {})
        goal_state = problem.get("goal_state", {})
        
        if not steps:
            return None
        
        # Implementasi sederhana: perencanaan berurutan
        current_state = initial_state.copy()
        plan = []
        
        # Iterasi melalui langkah-langkah
        for step in steps:
            # Periksa apakah langkah dapat diterapkan
            preconditions = step.get("preconditions", {})
            if all(current_state.get(k) == v for k, v in preconditions.items()):
                # Terapkan langkah
                effects = step.get("effects", {})
                for k, v in effects.items():
                    current_state[k] = v
                
                # Tambahkan langkah ke rencana
                plan.append(step.get("name", "unnamed_step"))
                
                # Periksa apakah tujuan tercapai
                if all(current_state.get(k) == v for k, v in goal_state.items()):
                    break
        
        return {"plan": plan, "final_state": current_state}
    
    def _solve_classification(self, problem: Dict[str, Any]) -> Any:
        """
        Selesaikan masalah klasifikasi.
        
        Args:
            problem: Masalah klasifikasi
            
        Returns:
            Model klasifikasi atau prediksi
        """
        features = problem.get("features", [])
        classes = problem.get("classes", [])
        training_data = problem.get("training_data", [])
        test_data = problem.get("test_data", [])
        
        if not features or not classes or not training_data:
            return None
        
        # Implementasi sederhana: klasifikasi berdasarkan kesamaan
        predictions = []
        
        for test_item in test_data:
            # Hitung kesamaan dengan setiap item pelatihan
            similarities = []
            for train_item in training_data:
                # Hitung kesamaan fitur
                sim = sum(1 for f in features if test_item.get(f) == train_item.get(f)) / len(features)
                similarities.append((sim, train_item.get("class")))
            
            # Pilih kelas dengan kesamaan tertinggi
            if similarities:
                best_class = max(similarities, key=lambda x: x[0])[1]
                predictions.append(best_class)
            else:
                # Default ke kelas pertama
                predictions.append(classes[0] if classes else None)
        
        return predictions
    
    def solve(self, problem: Dict[str, Any]) -> Any:
        """
        Selesaikan masalah.
        
        Args:
            problem: Masalah yang akan diselesaikan
            
        Returns:
            Solusi untuk masalah
        """
        # Pilih strategi pemecahan masalah
        strategy_name = self._select_strategy(problem)
        
        # Gunakan strategi yang dipilih
        if strategy_name in self.strategies:
            strategy = self.strategies[strategy_name]
            solution = strategy(problem)
            
            # Simpan hasil ke memori
            self._add_to_memory({
                "problem": problem,
                "strategy": strategy_name,
                "solution": solution,
                "score": self._evaluate_solution(problem, solution)
            })
            
            return solution
        else:
            # Gunakan implementasi dasar
            return super().solve(problem)
    
    def mutate(self) -> None:
        """
        Mutasi agen pemecahan masalah.
        """
        super().mutate()  # Panggil mutasi dasar
        
        # Mutasi parameter khusus
        self.max_iterations = int(self.max_iterations * random.uniform(0.8, 1.2))
        self.timeout = self.timeout * random.uniform(0.8, 1.2)
        
        # Batasi nilai parameter
        self.max_iterations = max(10, min(1000, self.max_iterations))
        self.timeout = max(1.0, min(60.0, self.timeout))
        
        # Update metadata
        self.metadata.update({
            "max_iterations": self.max_iterations,
            "timeout": self.timeout
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konversi agen pemecahan masalah ke dictionary.
        """
        data = super().to_dict()
        data.update({
            "problem_types": self.problem_types,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], function_map: Dict[str, Callable]) -> 'ProblemSolvingAgent':
        """
        Buat agen pemecahan masalah dari dictionary.
        """
        agent = super().from_dict(data, function_map)
        agent.problem_types = data["problem_types"]
        agent.max_iterations = data["max_iterations"]
        agent.timeout = data["timeout"]
        agent.strategies = agent._initialize_strategies()
        return agent