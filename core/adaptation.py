"""
Mesin adaptasi untuk Darwin-Gödel Machine.

Modul ini berisi implementasi mesin adaptasi yang digunakan oleh DGM
untuk beradaptasi dengan lingkungan dan tugas yang berubah.
"""

import random
import numpy as np
import time
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar, Generic, Union, Set

from simple_dgm.agents.base_agent import BaseAgent

# Tipe generik untuk individu dalam populasi
T = TypeVar('T', bound=BaseAgent)


class AdaptationEngine:
    """
    Mesin adaptasi untuk DGM.
    
    Mesin ini memungkinkan DGM beradaptasi dengan lingkungan dan tugas yang berubah.
    """
    
    def __init__(self):
        """
        Inisialisasi mesin adaptasi.
        """
        self.environment_models = {}
        self.task_models = {}
        self.resource_manager = ResourceManager()
        self.learning_rate_scheduler = LearningRateScheduler()
        self.adaptation_history = []
    
    def add_environment_model(self, name: str, model: 'EnvironmentModel'):
        """
        Tambahkan model lingkungan.
        
        Args:
            name: Nama model
            model: Model lingkungan
        """
        self.environment_models[name] = model
    
    def add_task_model(self, name: str, model: 'TaskModel'):
        """
        Tambahkan model tugas.
        
        Args:
            name: Nama model
            model: Model tugas
        """
        self.task_models[name] = model
    
    def adapt(self, dgm, environment: Dict[str, Any], task: Any) -> Any:
        """
        Adaptasi DGM dengan lingkungan dan tugas.
        
        Args:
            dgm: DGM yang akan diadaptasi
            environment: Lingkungan
            task: Tugas
            
        Returns:
            DGM yang telah diadaptasi
        """
        # Buat salinan DGM
        adapted_dgm = dgm
        
        # Analisis lingkungan
        environment_analysis = self._analyze_environment(environment)
        
        # Analisis tugas
        task_analysis = self._analyze_task(task)
        
        # Adaptasi parameter
        adapted_dgm = self._adapt_parameters(adapted_dgm, environment_analysis, task_analysis)
        
        # Adaptasi strategi
        adapted_dgm = self._adapt_strategy(adapted_dgm, environment_analysis, task_analysis)
        
        # Adaptasi sumber daya
        adapted_dgm = self._adapt_resources(adapted_dgm, environment_analysis, task_analysis)
        
        # Catat adaptasi
        self.adaptation_history.append({
            "environment": environment,
            "task": task,
            "environment_analysis": environment_analysis,
            "task_analysis": task_analysis
        })
        
        return adapted_dgm
    
    def _analyze_environment(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis lingkungan.
        
        Args:
            environment: Lingkungan
            
        Returns:
            Hasil analisis lingkungan
        """
        results = {}
        
        # Jalankan semua model lingkungan
        for name, model in self.environment_models.items():
            try:
                model_results = model.analyze(environment)
                results[name] = model_results
            except Exception as e:
                results[f"error_{name}"] = str(e)
        
        return results
    
    def _analyze_task(self, task: Any) -> Dict[str, Any]:
        """
        Analisis tugas.
        
        Args:
            task: Tugas
            
        Returns:
            Hasil analisis tugas
        """
        results = {}
        
        # Jalankan semua model tugas
        for name, model in self.task_models.items():
            try:
                model_results = model.analyze(task)
                results[name] = model_results
            except Exception as e:
                results[f"error_{name}"] = str(e)
        
        return results
    
    def _adapt_parameters(self, dgm, environment_analysis: Dict[str, Any], 
                         task_analysis: Dict[str, Any]) -> Any:
        """
        Adaptasi parameter DGM.
        
        Args:
            dgm: DGM yang akan diadaptasi
            environment_analysis: Hasil analisis lingkungan
            task_analysis: Hasil analisis tugas
            
        Returns:
            DGM dengan parameter yang diadaptasi
        """
        # Adaptasi tingkat pembelajaran
        if hasattr(dgm, "learning_rate"):
            new_learning_rate = self.learning_rate_scheduler.get_learning_rate(
                dgm.learning_rate, 
                environment_analysis, 
                task_analysis
            )
            dgm.learning_rate = new_learning_rate
        
        # Adaptasi tingkat eksplorasi
        if hasattr(dgm, "exploration_rate"):
            # Tingkatkan eksplorasi untuk tugas yang kompleks
            task_complexity = task_analysis.get("complexity", {}).get("value", 0.5)
            dgm.exploration_rate = 0.1 + 0.4 * task_complexity
        
        # Adaptasi ukuran populasi
        if hasattr(dgm, "population_size"):
            # Sesuaikan ukuran populasi berdasarkan sumber daya yang tersedia
            available_resources = environment_analysis.get("resources", {}).get("available", 1.0)
            min_population = 10
            max_population = 100
            dgm.population_size = int(min_population + (max_population - min_population) * available_resources)
        
        return dgm
    
    def _adapt_strategy(self, dgm, environment_analysis: Dict[str, Any], 
                       task_analysis: Dict[str, Any]) -> Any:
        """
        Adaptasi strategi DGM.
        
        Args:
            dgm: DGM yang akan diadaptasi
            environment_analysis: Hasil analisis lingkungan
            task_analysis: Hasil analisis tugas
            
        Returns:
            DGM dengan strategi yang diadaptasi
        """
        # Adaptasi strategi evolusi
        if hasattr(dgm, "evolution_strategy") and hasattr(dgm, "set_evolution_strategy"):
            # Pilih strategi berdasarkan karakteristik tugas
            task_type = task_analysis.get("type", "unknown")
            
            if task_type == "multi_objective":
                # Gunakan NSGA-II untuk tugas multi-objektif
                from simple_dgm.core.evolution_strategies import NSGA2Selection
                dgm.set_evolution_strategy(NSGA2Selection())
            
            elif task_type == "exploration":
                # Gunakan MAP-Elites untuk tugas eksplorasi
                from simple_dgm.core.evolution_strategies import MAP_ElitesStrategy
                dgm.set_evolution_strategy(MAP_ElitesStrategy())
            
            elif task_type == "optimization":
                # Gunakan CMA-ES untuk tugas optimasi
                from simple_dgm.core.evolution_strategies import CMA_ESStrategy
                dgm.set_evolution_strategy(CMA_ESStrategy())
            
            else:
                # Default ke seleksi turnamen
                from simple_dgm.core.evolution_strategies import TournamentSelection
                dgm.set_evolution_strategy(TournamentSelection())
        
        # Adaptasi operator mutasi
        if hasattr(dgm, "mutation_operator") and hasattr(dgm, "set_mutation_operator"):
            # Pilih operator berdasarkan fase evolusi
            generation = getattr(dgm, "generation", 0)
            
            if generation < 10:
                # Fase awal: gunakan mutasi parameter untuk eksplorasi
                from simple_dgm.core.mutation_operators import ParameterMutation
                dgm.set_mutation_operator(ParameterMutation(mutation_rate=0.3))
            
            elif generation < 50:
                # Fase tengah: gunakan mutasi struktural
                from simple_dgm.core.mutation_operators import StructuralMutation
                dgm.set_mutation_operator(StructuralMutation(mutation_rate=0.2))
            
            else:
                # Fase akhir: gunakan mutasi adaptif
                from simple_dgm.core.mutation_operators import AdaptiveMutation
                dgm.set_mutation_operator(AdaptiveMutation())
        
        # Adaptasi operator crossover
        if hasattr(dgm, "crossover_operator") and hasattr(dgm, "set_crossover_operator"):
            # Pilih operator berdasarkan keragaman populasi
            diversity = environment_analysis.get("population", {}).get("diversity", 0.5)
            
            if diversity < 0.3:
                # Keragaman rendah: gunakan crossover blend untuk meningkatkan keragaman
                from simple_dgm.core.crossover_operators import BlendCrossover
                dgm.set_crossover_operator(BlendCrossover(alpha=0.5))
            
            else:
                # Keragaman tinggi: gunakan crossover seragam
                from simple_dgm.core.crossover_operators import UniformCrossover
                dgm.set_crossover_operator(UniformCrossover())
        
        return dgm
    
    def _adapt_resources(self, dgm, environment_analysis: Dict[str, Any], 
                        task_analysis: Dict[str, Any]) -> Any:
        """
        Adaptasi sumber daya DGM.
        
        Args:
            dgm: DGM yang akan diadaptasi
            environment_analysis: Hasil analisis lingkungan
            task_analysis: Hasil analisis tugas
            
        Returns:
            DGM dengan sumber daya yang diadaptasi
        """
        # Dapatkan sumber daya yang tersedia
        available_resources = environment_analysis.get("resources", {}).get("available", 1.0)
        
        # Alokasikan sumber daya
        resource_allocation = self.resource_manager.allocate_resources(
            available_resources, 
            task_analysis
        )
        
        # Terapkan alokasi sumber daya
        if hasattr(dgm, "set_resource_allocation"):
            dgm.set_resource_allocation(resource_allocation)
        
        return dgm


class EnvironmentModel:
    """
    Model lingkungan untuk mesin adaptasi.
    
    Model ini menganalisis lingkungan DGM.
    """
    
    def __init__(self):
        """
        Inisialisasi model lingkungan.
        """
        pass
    
    def analyze(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis lingkungan.
        
        Args:
            environment: Lingkungan
            
        Returns:
            Hasil analisis lingkungan
        """
        results = {}
        
        # Analisis sumber daya
        results["resources"] = self._analyze_resources(environment)
        
        # Analisis populasi
        results["population"] = self._analyze_population(environment)
        
        # Analisis waktu
        results["time"] = self._analyze_time(environment)
        
        return results
    
    def _analyze_resources(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis sumber daya lingkungan.
        
        Args:
            environment: Lingkungan
            
        Returns:
            Hasil analisis sumber daya
        """
        results = {}
        
        # Dapatkan sumber daya dari lingkungan
        resources = environment.get("resources", {})
        
        # Analisis CPU
        cpu = resources.get("cpu", 1.0)
        results["cpu"] = cpu
        
        # Analisis memori
        memory = resources.get("memory", 1.0)
        results["memory"] = memory
        
        # Analisis penyimpanan
        storage = resources.get("storage", 1.0)
        results["storage"] = storage
        
        # Hitung sumber daya yang tersedia
        available = min(cpu, memory, storage)
        results["available"] = available
        
        return results
    
    def _analyze_population(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis populasi lingkungan.
        
        Args:
            environment: Lingkungan
            
        Returns:
            Hasil analisis populasi
        """
        results = {}
        
        # Dapatkan populasi dari lingkungan
        population = environment.get("population", [])
        
        # Hitung ukuran populasi
        results["size"] = len(population)
        
        # Hitung keragaman populasi
        if population:
            # Hitung keragaman berdasarkan fitness
            fitness_values = [ind.get("fitness", 0.0) for ind in population]
            fitness_mean = sum(fitness_values) / len(fitness_values)
            fitness_var = sum((f - fitness_mean) ** 2 for f in fitness_values) / len(fitness_values)
            fitness_std = fitness_var ** 0.5
            
            # Normalisasi keragaman
            if fitness_mean > 0:
                diversity = fitness_std / fitness_mean
            else:
                diversity = 0.0
            
            results["diversity"] = diversity
        else:
            results["diversity"] = 0.0
        
        return results
    
    def _analyze_time(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisis waktu lingkungan.
        
        Args:
            environment: Lingkungan
            
        Returns:
            Hasil analisis waktu
        """
        results = {}
        
        # Dapatkan waktu dari lingkungan
        time_info = environment.get("time", {})
        
        # Analisis waktu eksekusi
        execution_time = time_info.get("execution_time", 0.0)
        results["execution_time"] = execution_time
        
        # Analisis batas waktu
        timeout = time_info.get("timeout", float('inf'))
        results["timeout"] = timeout
        
        # Hitung waktu yang tersisa
        remaining_time = max(0.0, timeout - execution_time)
        results["remaining_time"] = remaining_time
        
        # Hitung rasio waktu
        if timeout > 0:
            time_ratio = execution_time / timeout
        else:
            time_ratio = 1.0
        
        results["time_ratio"] = time_ratio
        
        return results


class TaskModel:
    """
    Model tugas untuk mesin adaptasi.
    
    Model ini menganalisis tugas DGM.
    """
    
    def __init__(self):
        """
        Inisialisasi model tugas.
        """
        pass
    
    def analyze(self, task: Any) -> Dict[str, Any]:
        """
        Analisis tugas.
        
        Args:
            task: Tugas
            
        Returns:
            Hasil analisis tugas
        """
        results = {}
        
        # Analisis tipe tugas
        results["type"] = self._analyze_type(task)
        
        # Analisis kompleksitas tugas
        results["complexity"] = self._analyze_complexity(task)
        
        # Analisis dimensi tugas
        results["dimensions"] = self._analyze_dimensions(task)
        
        return results
    
    def _analyze_type(self, task: Any) -> Dict[str, Any]:
        """
        Analisis tipe tugas.
        
        Args:
            task: Tugas
            
        Returns:
            Hasil analisis tipe tugas
        """
        results = {}
        
        # Tentukan tipe tugas
        if isinstance(task, dict):
            # Periksa kunci dalam dictionary
            if "objectives" in task and len(task.get("objectives", [])) > 1:
                task_type = "multi_objective"
            elif "search_space" in task:
                task_type = "exploration"
            elif "objective" in task:
                task_type = "optimization"
            elif "classification" in task:
                task_type = "classification"
            elif "regression" in task:
                task_type = "regression"
            else:
                task_type = "unknown"
        elif isinstance(task, list):
            # Periksa apakah ini adalah daftar tugas
            if all(isinstance(t, dict) for t in task):
                task_type = "multi_task"
            else:
                task_type = "list"
        else:
            task_type = "unknown"
        
        results["value"] = task_type
        
        return results
    
    def _analyze_complexity(self, task: Any) -> Dict[str, Any]:
        """
        Analisis kompleksitas tugas.
        
        Args:
            task: Tugas
            
        Returns:
            Hasil analisis kompleksitas tugas
        """
        results = {}
        
        # Tentukan kompleksitas tugas
        if isinstance(task, dict):
            # Hitung kompleksitas berdasarkan jumlah kunci dan kedalaman
            num_keys = len(task)
            depth = self._calculate_dict_depth(task)
            
            # Normalisasi kompleksitas
            complexity = min(1.0, (num_keys * depth) / 100.0)
        elif isinstance(task, list):
            # Hitung kompleksitas berdasarkan panjang daftar
            complexity = min(1.0, len(task) / 100.0)
        else:
            complexity = 0.1
        
        results["value"] = complexity
        
        return results
    
    def _calculate_dict_depth(self, d: Dict[str, Any], current_depth: int = 1) -> int:
        """
        Hitung kedalaman dictionary.
        
        Args:
            d: Dictionary
            current_depth: Kedalaman saat ini
            
        Returns:
            Kedalaman maksimum
        """
        if not isinstance(d, dict) or not d:
            return current_depth
        
        return max(self._calculate_dict_depth(v, current_depth + 1) if isinstance(v, dict) else current_depth
                  for k, v in d.items())
    
    def _analyze_dimensions(self, task: Any) -> Dict[str, Any]:
        """
        Analisis dimensi tugas.
        
        Args:
            task: Tugas
            
        Returns:
            Hasil analisis dimensi tugas
        """
        results = {}
        
        # Tentukan dimensi tugas
        if isinstance(task, dict):
            # Periksa dimensi dalam dictionary
            if "dimensions" in task:
                dimensions = task["dimensions"]
            elif "variables" in task:
                dimensions = len(task["variables"])
            elif "features" in task:
                dimensions = len(task["features"])
            else:
                dimensions = len(task)
        elif isinstance(task, list):
            # Dimensi adalah panjang daftar
            dimensions = len(task)
        else:
            dimensions = 1
        
        results["value"] = dimensions
        
        return results


class ResourceManager:
    """
    Pengelola sumber daya untuk mesin adaptasi.
    
    Pengelola ini mengalokasikan sumber daya untuk DGM.
    """
    
    def __init__(self):
        """
        Inisialisasi pengelola sumber daya.
        """
        pass
    
    def allocate_resources(self, available_resources: float, 
                          task_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Alokasikan sumber daya.
        
        Args:
            available_resources: Sumber daya yang tersedia
            task_analysis: Hasil analisis tugas
            
        Returns:
            Alokasi sumber daya
        """
        allocation = {}
        
        # Dapatkan kompleksitas tugas
        task_complexity = task_analysis.get("complexity", {}).get("value", 0.5)
        
        # Alokasikan sumber daya berdasarkan kompleksitas tugas
        if task_complexity < 0.3:
            # Tugas sederhana: alokasikan lebih banyak untuk eksplorasi
            allocation["exploration"] = 0.7 * available_resources
            allocation["exploitation"] = 0.3 * available_resources
        elif task_complexity < 0.7:
            # Tugas menengah: alokasikan seimbang
            allocation["exploration"] = 0.5 * available_resources
            allocation["exploitation"] = 0.5 * available_resources
        else:
            # Tugas kompleks: alokasikan lebih banyak untuk eksploitasi
            allocation["exploration"] = 0.3 * available_resources
            allocation["exploitation"] = 0.7 * available_resources
        
        # Alokasikan sumber daya untuk komponen lain
        allocation["evaluation"] = 0.4 * available_resources
        allocation["selection"] = 0.2 * available_resources
        allocation["mutation"] = 0.2 * available_resources
        allocation["crossover"] = 0.2 * available_resources
        
        return allocation


class LearningRateScheduler:
    """
    Penjadwal tingkat pembelajaran untuk mesin adaptasi.
    
    Penjadwal ini menyesuaikan tingkat pembelajaran DGM.
    """
    
    def __init__(self, initial_rate: float = 0.01, min_rate: float = 0.001, 
                max_rate: float = 0.1, decay_factor: float = 0.9):
        """
        Inisialisasi penjadwal tingkat pembelajaran.
        
        Args:
            initial_rate: Tingkat pembelajaran awal
            min_rate: Tingkat pembelajaran minimum
            max_rate: Tingkat pembelajaran maksimum
            decay_factor: Faktor peluruhan
        """
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.decay_factor = decay_factor
        self.current_rate = initial_rate
        self.step = 0
    
    def get_learning_rate(self, current_rate: float, 
                         environment_analysis: Dict[str, Any], 
                         task_analysis: Dict[str, Any]) -> float:
        """
        Dapatkan tingkat pembelajaran yang disesuaikan.
        
        Args:
            current_rate: Tingkat pembelajaran saat ini
            environment_analysis: Hasil analisis lingkungan
            task_analysis: Hasil analisis tugas
            
        Returns:
            Tingkat pembelajaran yang disesuaikan
        """
        # Tingkatkan langkah
        self.step += 1
        
        # Dapatkan kompleksitas tugas
        task_complexity = task_analysis.get("complexity", {}).get("value", 0.5)
        
        # Dapatkan rasio waktu
        time_ratio = environment_analysis.get("time", {}).get("time_ratio", 0.5)
        
        # Hitung tingkat pembelajaran berdasarkan jadwal
        if self.step < 10:
            # Fase awal: tingkat pembelajaran tinggi
            rate = self.max_rate
        elif self.step < 50:
            # Fase tengah: tingkat pembelajaran menengah
            rate = self.initial_rate
        else:
            # Fase akhir: tingkat pembelajaran rendah
            rate = self.min_rate + (self.initial_rate - self.min_rate) * (self.decay_factor ** (self.step - 50))
        
        # Sesuaikan berdasarkan kompleksitas tugas
        rate *= (1.0 - 0.5 * task_complexity)
        
        # Sesuaikan berdasarkan rasio waktu
        if time_ratio > 0.8:
            # Waktu hampir habis: tingkatkan tingkat pembelajaran
            rate *= 1.5
        
        # Batasi tingkat pembelajaran
        rate = max(self.min_rate, min(self.max_rate, rate))
        
        return rate