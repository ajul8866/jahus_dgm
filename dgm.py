"""
Implementasi utama dari Darwin-Gödel Machine (DGM).

DGM adalah sistem AI yang mampu meningkatkan dirinya sendiri dengan mengkombinasikan
evolusi Darwinian dan konsep perbaikan diri Gödelian. Sistem ini menggunakan
mekanisme evolusi kompleks dan kemampuan introspeksi untuk menghasilkan agen
yang semakin canggih tanpa intervensi manusia.
"""

import random
import copy
import uuid
import time
import math
import json
import os
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable, Tuple, Set, Union, TypeVar, Generic

from simple_dgm.agents.base_agent import BaseAgent
from simple_dgm.utils.evaluation import evaluate_agent
from simple_dgm.core.evolution_strategies import EvolutionStrategy, TournamentSelection, RouletteWheelSelection, NSGA2Selection
from simple_dgm.core.mutation_operators import MutationOperator, ParameterMutation, StructuralMutation, FunctionalMutation
from simple_dgm.core.crossover_operators import CrossoverOperator, UniformCrossover, SinglePointCrossover
from simple_dgm.core.fitness_functions import FitnessFunction, MultiObjectiveFitness
from simple_dgm.core.diversity_metrics import DiversityMetric, BehavioralDiversity, StructuralDiversity
from simple_dgm.core.archive_strategies import ArchiveStrategy, NoveltyArchive, EliteArchive, QualityDiversityArchive


class DGM:
    """
    Darwin-Gödel Machine (DGM) - Sistem AI yang mampu meningkatkan dirinya sendiri
    dengan mengkombinasikan evolusi Darwinian dan konsep perbaikan diri Gödelian.
    
    Implementasi canggih DGM dengan fitur-fitur:
    - Strategi evolusi kompleks
    - Operator mutasi dan crossover adaptif
    - Fungsi fitness multi-objektif
    - Metrik keragaman
    - Strategi arsip
    - Mesin introspeksi
    - Mesin adaptasi
    - Mesin kolaborasi
    - Integrasi LLM
    """
    
    def __init__(self, initial_agent: Optional[BaseAgent] = None, population_size: int = 100):
        """
        Inisialisasi DGM dengan agen awal.
        
        Args:
            initial_agent: Agen awal untuk memulai evolusi. Jika None, akan dibuat BaseAgent baru.
            population_size: Ukuran populasi
        """
        self.archive = {}  # Arsip agen: {agent_id: {"agent": agent, "score": score, "parent_id": parent_id}}
        self.evolution_history = []  # Riwayat evolusi: [(agent_id, parent_id, score, generation)]
        self.current_generation = 0
        self.population_size = population_size
        
        # Komponen evolusi canggih
        self.evolution_strategy = None
        self.mutation_operator = None
        self.crossover_operator = None
        self.fitness_function = None
        self.diversity_metric = None
        self.archive_strategy = None
        
        # Mesin canggih
        self.introspection_engine = None
        self.adaptation_engine = None
        self.collaboration_engine = None
        self.llm_interface = None
        
        # Statistik evolusi
        self.evolution_stats = {
            "best_fitness": [],
            "avg_fitness": [],
            "diversity": [],
            "execution_time": []
        }
        
        # Tambahkan agen awal ke arsip
        if initial_agent is None:
            initial_agent = BaseAgent()
        
        initial_agent_id = str(uuid.uuid4())
        self.archive[initial_agent_id] = {
            "agent": initial_agent,
            "score": 0.0,
            "parent_id": None,
            "fitness": [0.0],  # Fitness multi-objektif
            "generation": 0
        }
        self.evolution_history.append((initial_agent_id, None, 0.0, 0))
        
        # Inisialisasi populasi awal
        self._initialize_population(initial_agent)
    
    def _initialize_population(self, initial_agent: BaseAgent) -> None:
        """
        Inisialisasi populasi awal.
        
        Args:
            initial_agent: Agen awal
        """
        # Jika ukuran populasi > 1, buat agen tambahan dengan mutasi
        for _ in range(1, self.population_size):
            # Buat agen baru dengan mutasi
            new_agent = copy.deepcopy(initial_agent)
            new_agent.mutate()
            
            # Tambahkan agen baru ke arsip
            new_agent_id = str(uuid.uuid4())
            self.archive[new_agent_id] = {
                "agent": new_agent,
                "score": 0.0,
                "parent_id": None,
                "fitness": [0.0],
                "generation": 0
            }
    
    def set_evolution_strategy(self, strategy):
        """
        Tetapkan strategi evolusi.
        
        Args:
            strategy: Strategi evolusi
        """
        self.evolution_strategy = strategy
    
    def set_mutation_operator(self, operator):
        """
        Tetapkan operator mutasi.
        
        Args:
            operator: Operator mutasi
        """
        self.mutation_operator = operator
    
    def set_crossover_operator(self, operator):
        """
        Tetapkan operator crossover.
        
        Args:
            operator: Operator crossover
        """
        self.crossover_operator = operator
    
    def set_fitness_function(self, function):
        """
        Tetapkan fungsi fitness.
        
        Args:
            function: Fungsi fitness
        """
        self.fitness_function = function
    
    def set_diversity_metric(self, metric):
        """
        Tetapkan metrik keragaman.
        
        Args:
            metric: Metrik keragaman
        """
        self.diversity_metric = metric
    
    def set_archive_strategy(self, strategy):
        """
        Tetapkan strategi arsip.
        
        Args:
            strategy: Strategi arsip
        """
        self.archive_strategy = strategy
    
    def set_introspection_engine(self, engine):
        """
        Tetapkan mesin introspeksi.
        
        Args:
            engine: Mesin introspeksi
        """
        self.introspection_engine = engine
    
    def set_adaptation_engine(self, engine):
        """
        Tetapkan mesin adaptasi.
        
        Args:
            engine: Mesin adaptasi
        """
        self.adaptation_engine = engine
    
    def set_collaboration_engine(self, engine):
        """
        Tetapkan mesin kolaborasi.
        
        Args:
            engine: Mesin kolaborasi
        """
        self.collaboration_engine = engine
    
    def set_llm_interface(self, interface):
        """
        Tetapkan antarmuka LLM.
        
        Args:
            interface: Antarmuka LLM
        """
        self.llm_interface = interface
    
    def select_parent(self, selection_method: str = "tournament") -> str:
        """
        Pilih agen induk dari arsip untuk dimodifikasi.
        
        Args:
            selection_method: Metode seleksi ("tournament", "roulette", "random", "nsga2")
            
        Returns:
            ID agen yang dipilih
        """
        # Jika strategi evolusi tersedia, gunakan itu
        if self.evolution_strategy:
            # Konversi arsip ke format yang diharapkan oleh strategi evolusi
            population = [(info["agent"], info["score"]) for info in self.archive.values()]
            
            # Pilih induk
            parents = self.evolution_strategy.select_parents(population)
            
            # Temukan ID agen induk pertama
            for agent_id, info in self.archive.items():
                if info["agent"] is parents[0]:
                    return agent_id
            
            # Fallback ke metode default
            return random.choice(list(self.archive.keys()))
        
        # Metode seleksi default
        if selection_method == "tournament":
            # Tournament selection: Pilih k agen secara acak dan ambil yang terbaik
            k = min(3, len(self.archive))
            candidates = random.sample(list(self.archive.keys()), k)
            return max(candidates, key=lambda agent_id: self.archive[agent_id]["score"])
        
        elif selection_method == "roulette":
            # Roulette wheel selection: Probabilitas seleksi proporsional dengan skor
            total_score = sum(info["score"] + 1.0 for info in self.archive.values())  # +1 untuk menghindari skor 0
            r = random.uniform(0, total_score)
            cumulative = 0
            for agent_id, info in self.archive.items():
                cumulative += info["score"] + 1.0
                if cumulative >= r:
                    return agent_id
            return list(self.archive.keys())[-1]  # Fallback
        
        elif selection_method == "nsga2":
            # NSGA-II selection: Berdasarkan dominasi Pareto dan jarak kerumunan
            if not all("fitness" in info and isinstance(info["fitness"], list) for info in self.archive.values()):
                # Jika fitness multi-objektif tidak tersedia, fallback ke tournament
                return self.select_parent("tournament")
            
            # Implementasi sederhana NSGA-II
            # 1. Hitung front Pareto
            fronts = self._fast_non_dominated_sort()
            
            # 2. Pilih agen dari front pertama
            if fronts and fronts[0]:
                return random.choice(fronts[0])
            
            # Fallback
            return random.choice(list(self.archive.keys()))
        
        else:  # "random"
            # Random selection: Pilih agen secara acak
            return random.choice(list(self.archive.keys()))
    
    def _fast_non_dominated_sort(self) -> List[List[str]]:
        """
        Lakukan pengurutan non-dominasi cepat untuk NSGA-II.
        
        Returns:
            Daftar front Pareto (daftar ID agen)
        """
        # Inisialisasi
        domination_count = {agent_id: 0 for agent_id in self.archive}
        dominated_set = {agent_id: [] for agent_id in self.archive}
        fronts = [[]]
        
        # Untuk setiap agen
        for agent_id1 in self.archive:
            for agent_id2 in self.archive:
                if agent_id1 == agent_id2:
                    continue
                
                # Periksa dominasi
                fitness1 = self.archive[agent_id1]["fitness"]
                fitness2 = self.archive[agent_id2]["fitness"]
                
                if all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 > f2 for f1, f2 in zip(fitness1, fitness2)):
                    # agent_id1 mendominasi agent_id2
                    dominated_set[agent_id1].append(agent_id2)
                elif all(f2 >= f1 for f1, f2 in zip(fitness1, fitness2)) and any(f2 > f1 for f1, f2 in zip(fitness1, fitness2)):
                    # agent_id2 mendominasi agent_id1
                    domination_count[agent_id1] += 1
            
            # Jika agent_id1 tidak didominasi, tambahkan ke front pertama
            if domination_count[agent_id1] == 0:
                fronts[0].append(agent_id1)
        
        # Buat front berikutnya
        i = 0
        while fronts[i]:
            next_front = []
            
            for agent_id in fronts[i]:
                for dominated_id in dominated_set[agent_id]:
                    domination_count[dominated_id] -= 1
                    if domination_count[dominated_id] == 0:
                        next_front.append(dominated_id)
            
            i += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def modify_agent(self, parent_agent: BaseAgent) -> BaseAgent:
        """
        Modifikasi agen untuk menghasilkan agen baru.
        
        Args:
            parent_agent: Agen induk yang akan dimodifikasi
            
        Returns:
            Agen baru hasil modifikasi
        """
        # Jika operator mutasi tersedia, gunakan itu
        if self.mutation_operator:
            return self.mutation_operator.mutate(parent_agent)
        
        # Metode default
        child_agent = copy.deepcopy(parent_agent)
        child_agent.mutate()
        return child_agent
    
    def crossover(self, parent1: BaseAgent, parent2: BaseAgent) -> BaseAgent:
        """
        Lakukan crossover antara dua agen induk.
        
        Args:
            parent1: Agen induk pertama
            parent2: Agen induk kedua
            
        Returns:
            Agen anak hasil crossover
        """
        # Jika operator crossover tersedia, gunakan itu
        if self.crossover_operator:
            return self.crossover_operator.crossover(parent1, parent2)
        
        # Metode default: salin parent1
        return copy.deepcopy(parent1)
    
    def evaluate_agent(self, agent: BaseAgent, task: Any) -> Union[float, List[float]]:
        """
        Evaluasi agen.
        
        Args:
            agent: Agen yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Nilai fitness (tunggal atau multi-objektif)
        """
        # Jika fungsi fitness tersedia, gunakan itu
        if self.fitness_function:
            return self.fitness_function.evaluate(agent, task)
        
        # Metode default
        return evaluate_agent(agent, task)
    
    def evolve(self, generations: int = 10, task: Any = None, 
               parallel: bool = False, num_workers: int = 4) -> None:
        """
        Jalankan proses evolusi DGM untuk beberapa generasi.
        
        Args:
            generations: Jumlah generasi untuk evolusi
            task: Tugas yang akan digunakan untuk evaluasi
            parallel: Apakah akan menggunakan evaluasi paralel
            num_workers: Jumlah worker untuk evaluasi paralel
        """
        start_time = time.time()
        
        # Evaluasi populasi awal
        self._evaluate_population(task, parallel, num_workers)
        
        # Jalankan evolusi
        for gen in range(generations):
            gen_start_time = time.time()
            self.current_generation += 1
            
            # Adaptasi DGM jika mesin adaptasi tersedia
            if self.adaptation_engine:
                environment = {
                    "resources": {
                        "cpu": 1.0,
                        "memory": 1.0,
                        "storage": 1.0
                    },
                    "population": [
                        {"id": agent_id, "fitness": info["score"]}
                        for agent_id, info in self.archive.items()
                    ],
                    "time": {
                        "execution_time": time.time() - start_time,
                        "timeout": float('inf')
                    }
                }
                self.adaptation_engine.adapt(self, environment, task)
            
            # Evolusi berdasarkan strategi
            if self.evolution_strategy:
                # Evolusi canggih
                self._evolve_advanced(task, parallel, num_workers)
            else:
                # Evolusi dasar
                self._evolve_basic(task)
            
            # Perbarui statistik
            self._update_statistics()
            
            # Catat waktu eksekusi
            gen_end_time = time.time()
            self.evolution_stats["execution_time"].append(gen_end_time - gen_start_time)
        
        end_time = time.time()
        print(f"Evolution completed in {end_time - start_time:.2f} seconds.")
        
        # Cetak informasi agen terbaik
        best_agent_id = self.get_best_agent_id()
        if best_agent_id:
            best_agent = self.archive[best_agent_id]["agent"]
            best_score = self.archive[best_agent_id]["score"]
            print(f"Best agent score: {best_score:.4f}")
            
            # Cetak informasi tambahan jika tersedia
            if hasattr(best_agent, "tools"):
                tool_names = [tool.name for tool in best_agent.tools]
                print(f"Best agent tools: {tool_names}")
            
            if hasattr(best_agent, "learning_rate") and hasattr(best_agent, "exploration_rate") and hasattr(best_agent, "memory_capacity"):
                print(f"Best agent parameters:")
                print(f"  - Memory capacity: {best_agent.memory_capacity}")
                print(f"  - Learning rate: {best_agent.learning_rate:.4f}")
                print(f"  - Exploration rate: {best_agent.exploration_rate:.4f}")
    
    def _evolve_basic(self, task: Any) -> None:
        """
        Jalankan evolusi dasar untuk satu generasi.
        
        Args:
            task: Tugas untuk evaluasi
        """
        # Pilih agen induk
        parent_id = self.select_parent()
        parent_agent = self.archive[parent_id]["agent"]
        
        # Modifikasi agen untuk menghasilkan agen baru
        child_agent = self.modify_agent(parent_agent)
        
        # Evaluasi agen baru
        fitness = self.evaluate_agent(child_agent, task)
        
        # Konversi fitness ke skor tunggal jika perlu
        if isinstance(fitness, list):
            score = sum(fitness) / len(fitness)
        else:
            score = fitness
            fitness = [fitness]
        
        # Tambahkan agen baru ke arsip
        child_id = str(uuid.uuid4())
        self.archive[child_id] = {
            "agent": child_agent,
            "score": score,
            "parent_id": parent_id,
            "fitness": fitness,
            "generation": self.current_generation
        }
        
        # Catat dalam riwayat evolusi
        self.evolution_history.append((child_id, parent_id, score, self.current_generation))
        
        print(f"Generation {self.current_generation}: New agent {child_id[:8]} (from {parent_id[:8]}) - Score: {score:.4f}")
    
    def _evolve_advanced(self, task: Any, parallel: bool = False, num_workers: int = 4) -> None:
        """
        Jalankan evolusi canggih untuk satu generasi.
        
        Args:
            task: Tugas untuk evaluasi
            parallel: Apakah akan menggunakan evaluasi paralel
            num_workers: Jumlah worker untuk evaluasi paralel
        """
        # Konversi arsip ke format yang diharapkan oleh strategi evolusi
        population = [(info["agent"], info["score"]) for info in self.archive.values()]
        
        # Hasilkan keturunan
        offspring = []
        offspring_parent_ids = []
        
        for _ in range(self.evolution_strategy.offspring_size):
            # Pilih induk
            parents = self.evolution_strategy.select_parents(population)
            parent_indices = []
            
            # Temukan ID agen induk
            for parent in parents:
                for i, (agent_id, info) in enumerate(self.archive.items()):
                    if info["agent"] is parent:
                        parent_indices.append(agent_id)
                        break
            
            # Jika tidak cukup induk ditemukan, gunakan induk acak
            while len(parent_indices) < 2:
                parent_indices.append(random.choice(list(self.archive.keys())))
            
            # Buat keturunan
            if self.crossover_operator and len(parents) >= 2:
                # Lakukan crossover
                child = self.crossover_operator.crossover(parents[0], parents[1])
                # Lakukan mutasi
                child = self.mutation_operator.mutate(child) if self.mutation_operator else child
            else:
                # Salin induk pertama
                child = copy.deepcopy(parents[0])
                # Lakukan mutasi
                child = self.mutation_operator.mutate(child) if self.mutation_operator else child.mutate()
            
            # Tambahkan ke daftar keturunan
            offspring.append(child)
            offspring_parent_ids.append(parent_indices[0])  # Gunakan induk pertama sebagai induk utama
        
        # Evaluasi keturunan
        if parallel:
            # Evaluasi paralel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Buat fungsi evaluasi
                def evaluate_agent_wrapper(agent):
                    return self.evaluate_agent(agent, task)
                
                # Evaluasi semua agen
                fitness_values = list(executor.map(evaluate_agent_wrapper, offspring))
        else:
            # Evaluasi sekuensial
            fitness_values = [self.evaluate_agent(agent, task) for agent in offspring]
        
        # Tambahkan keturunan ke arsip
        for i, (child, parent_id, fitness) in enumerate(zip(offspring, offspring_parent_ids, fitness_values)):
            # Konversi fitness ke skor tunggal jika perlu
            if isinstance(fitness, list):
                score = sum(fitness) / len(fitness)
            else:
                score = fitness
                fitness = [fitness]
            
            # Tambahkan agen baru ke arsip
            child_id = str(uuid.uuid4())
            self.archive[child_id] = {
                "agent": child,
                "score": score,
                "parent_id": parent_id,
                "fitness": fitness,
                "generation": self.current_generation
            }
            
            # Catat dalam riwayat evolusi
            self.evolution_history.append((child_id, parent_id, score, self.current_generation))
            
            # Cetak kemajuan untuk beberapa keturunan
            if i % max(1, len(offspring) // 10) == 0:
                print(f"Generation {self.current_generation}: New agent {child_id[:8]} (from {parent_id[:8]}) - Score: {score:.4f}")
        
        # Pilih individu yang akan bertahan
        if hasattr(self.evolution_strategy, "select_survivors"):
            # Konversi keturunan ke format yang diharapkan oleh strategi evolusi
            offspring_population = [(info["agent"], info["score"]) 
                                   for agent_id, info in self.archive.items() 
                                   if info["generation"] == self.current_generation]
            
            # Pilih individu yang akan bertahan
            new_population = self.evolution_strategy.select_survivors(population, offspring_population)
            
            # Perbarui arsip
            new_archive = {}
            for agent, score in new_population:
                # Temukan ID agen
                agent_id = None
                for aid, info in self.archive.items():
                    if info["agent"] is agent:
                        agent_id = aid
                        break
                
                if agent_id:
                    new_archive[agent_id] = self.archive[agent_id]
            
            self.archive = new_archive
        
        # Batasi ukuran arsip jika perlu
        if len(self.archive) > self.population_size * 2:
            # Pilih individu terbaik
            sorted_archive = sorted(self.archive.items(), key=lambda x: x[1]["score"], reverse=True)
            self.archive = {agent_id: info for agent_id, info in sorted_archive[:self.population_size]}
    
    def _evaluate_population(self, task: Any, parallel: bool = False, num_workers: int = 4) -> None:
        """
        Evaluasi seluruh populasi.
        
        Args:
            task: Tugas untuk evaluasi
            parallel: Apakah akan menggunakan evaluasi paralel
            num_workers: Jumlah worker untuk evaluasi paralel
        """
        # Dapatkan agen yang belum dievaluasi
        unevaluated_agents = {
            agent_id: info["agent"] 
            for agent_id, info in self.archive.items() 
            if info["score"] == 0.0
        }
        
        if not unevaluated_agents:
            return
        
        # Evaluasi agen
        if parallel:
            # Evaluasi paralel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Buat fungsi evaluasi
                def evaluate_agent_wrapper(agent_tuple):
                    agent_id, agent = agent_tuple
                    fitness = self.evaluate_agent(agent, task)
                    return agent_id, fitness
                
                # Evaluasi semua agen
                results = list(executor.map(
                    evaluate_agent_wrapper, 
                    [(agent_id, agent) for agent_id, agent in unevaluated_agents.items()]
                ))
                
                # Perbarui arsip
                for agent_id, fitness in results:
                    # Konversi fitness ke skor tunggal jika perlu
                    if isinstance(fitness, list):
                        score = sum(fitness) / len(fitness)
                    else:
                        score = fitness
                        fitness = [fitness]
                    
                    self.archive[agent_id]["score"] = score
                    self.archive[agent_id]["fitness"] = fitness
        else:
            # Evaluasi sekuensial
            for agent_id, agent in unevaluated_agents.items():
                fitness = self.evaluate_agent(agent, task)
                
                # Konversi fitness ke skor tunggal jika perlu
                if isinstance(fitness, list):
                    score = sum(fitness) / len(fitness)
                else:
                    score = fitness
                    fitness = [fitness]
                
                self.archive[agent_id]["score"] = score
                self.archive[agent_id]["fitness"] = fitness
    
    def _update_statistics(self) -> None:
        """
        Perbarui statistik evolusi.
        """
        # Hitung fitness terbaik dan rata-rata
        if self.archive:
            scores = [info["score"] for info in self.archive.values()]
            best_fitness = max(scores)
            avg_fitness = sum(scores) / len(scores)
            
            self.evolution_stats["best_fitness"].append(best_fitness)
            self.evolution_stats["avg_fitness"].append(avg_fitness)
        
        # Hitung keragaman jika metrik tersedia
        if self.diversity_metric:
            agents = [info["agent"] for info in self.archive.values()]
            diversity = self.diversity_metric.measure(agents)
            self.evolution_stats["diversity"].append(diversity)
    
    def get_best_agent(self) -> BaseAgent:
        """
        Dapatkan agen dengan skor tertinggi dari arsip.
        
        Returns:
            Agen terbaik
        """
        best_agent_id = self.get_best_agent_id()
        if best_agent_id:
            return self.archive[best_agent_id]["agent"]
        return None
    
    def get_best_agent_id(self) -> str:
        """
        Dapatkan ID agen dengan skor tertinggi dari arsip.
        
        Returns:
            ID agen terbaik
        """
        if not self.archive:
            return None
        return max(self.archive.keys(), key=lambda agent_id: self.archive[agent_id]["score"])
    
    def get_agent(self, agent_id: str) -> BaseAgent:
        """
        Dapatkan agen berdasarkan ID.
        
        Args:
            agent_id: ID agen
            
        Returns:
            Agen
        """
        if agent_id in self.archive:
            return self.archive[agent_id]["agent"]
        return None
    
    def get_agent_score(self, agent_id: str) -> float:
        """
        Dapatkan skor agen berdasarkan ID.
        
        Args:
            agent_id: ID agen
            
        Returns:
            Skor agen
        """
        if agent_id in self.archive:
            return self.archive[agent_id]["score"]
        return 0.0
    
    def get_agent_fitness(self, agent_id: str) -> List[float]:
        """
        Dapatkan fitness multi-objektif agen berdasarkan ID.
        
        Args:
            agent_id: ID agen
            
        Returns:
            Fitness multi-objektif agen
        """
        if agent_id in self.archive and "fitness" in self.archive[agent_id]:
            return self.archive[agent_id]["fitness"]
        return [0.0]
    
    def get_evolution_tree(self) -> Dict[str, Any]:
        """
        Dapatkan struktur pohon evolusi.
        
        Returns:
            Struktur pohon evolusi dalam format dictionary
        """
        tree = {}
        for agent_id, info in self.archive.items():
            parent_id = info["parent_id"]
            if parent_id is None:
                # Root node
                if "children" not in tree:
                    tree["children"] = []
                tree["children"].append({
                    "id": agent_id,
                    "score": info["score"],
                    "generation": info.get("generation", 0),
                    "children": []
                })
            else:
                # Find parent in tree and add this node as child
                def add_child_to_parent(node):
                    if node["id"] == parent_id:
                        if "children" not in node:
                            node["children"] = []
                        node["children"].append({
                            "id": agent_id,
                            "score": info["score"],
                            "generation": info.get("generation", self.current_generation),
                            "children": []
                        })
                        return True
                    if "children" in node:
                        for child in node["children"]:
                            if add_child_to_parent(child):
                                return True
                    return False
                
                if "children" in tree:
                    for root_child in tree["children"]:
                        if add_child_to_parent(root_child):
                            break
        
        return tree
    
    def get_statistics(self) -> Dict[str, List[float]]:
        """
        Dapatkan statistik evolusi.
        
        Returns:
            Statistik evolusi
        """
        return self.evolution_stats
    
    def introspect(self) -> Dict[str, Any]:
        """
        Introspeksi DGM.
        
        Returns:
            Hasil introspeksi
        """
        if self.introspection_engine:
            return self.introspection_engine.analyze(self)
        else:
            # Introspeksi dasar
            return {
                "num_agents": len(self.archive),
                "best_score": self.archive[self.get_best_agent_id()]["score"] if self.get_best_agent_id() else 0.0,
                "avg_score": sum(info["score"] for info in self.archive.values()) / len(self.archive) if self.archive else 0.0,
                "generation": self.current_generation
            }
    
    def improve(self) -> bool:
        """
        Tingkatkan DGM berdasarkan introspeksi.
        
        Returns:
            True jika berhasil, False jika tidak
        """
        if self.introspection_engine:
            # Analisis DGM
            analysis = self.introspection_engine.analyze(self)
            
            # Tingkatkan DGM
            improved_dgm = self.introspection_engine.improve(self, analysis)
            
            # Perbarui komponen
            if hasattr(improved_dgm, "evolution_strategy") and improved_dgm.evolution_strategy:
                self.evolution_strategy = improved_dgm.evolution_strategy
            
            if hasattr(improved_dgm, "mutation_operator") and improved_dgm.mutation_operator:
                self.mutation_operator = improved_dgm.mutation_operator
            
            if hasattr(improved_dgm, "crossover_operator") and improved_dgm.crossover_operator:
                self.crossover_operator = improved_dgm.crossover_operator
            
            return True
        
        return False
    
    def collaborate(self, task: Any) -> Any:
        """
        Kolaborasi antar agen untuk menyelesaikan tugas.
        
        Args:
            task: Tugas yang akan diselesaikan
            
        Returns:
            Solusi kolaboratif
        """
        if self.collaboration_engine:
            # Tambahkan semua agen ke mesin kolaborasi
            for agent_id, info in self.archive.items():
                self.collaboration_engine.add_agent(
                    info["agent"], 
                    agent_id, 
                    info["agent"].__class__.__name__
                )
            
            # Kolaborasi untuk menyelesaikan tugas
            return self.collaboration_engine.collaborate(task)
        else:
            # Gunakan agen terbaik
            best_agent = self.get_best_agent()
            if best_agent:
                return best_agent.solve(task)
            return None
    
    def save(self, filepath: str) -> None:
        """
        Simpan DGM ke file.
        
        Args:
            filepath: Jalur file
        """
        from simple_dgm.utils.serialization import save_dgm
        save_dgm(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'DGM':
        """
        Muat DGM dari file.
        
        Args:
            filepath: Jalur file
            
        Returns:
            DGM
        """
        from simple_dgm.utils.serialization import load_dgm
        return load_dgm(filepath)