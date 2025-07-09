"""
Contoh penggunaan lanjutan Darwin-Gödel Machine.

Contoh ini mendemonstrasikan penggunaan fitur-fitur lanjutan DGM seperti:
- Strategi evolusi
- Operator mutasi
- Operator crossover
- Fungsi fitness multi-objektif
- Metrik keragaman
- Strategi arsip
- Introspeksi dan adaptasi
- Kolaborasi
- Integrasi LLM
"""

import random
import numpy as np
from typing import List, Dict, Any, Tuple

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.core.evolution_strategies import TournamentSelection, RouletteWheelSelection
from simple_dgm.core.mutation_operators import ParameterMutation, StructuralMutation
from simple_dgm.core.crossover_operators import BlendCrossover
from simple_dgm.core.fitness_functions import MultiObjectiveFitness
from simple_dgm.core.diversity_metrics import BehavioralDiversity
from simple_dgm.core.archive_strategies import QualityDiversityArchive
from simple_dgm.core.introspection import IntrospectionEngine
from simple_dgm.core.adaptation import AdaptationEngine
from simple_dgm.core.collaboration import CollaborationEngine
from simple_dgm.core.llm_integration import LLMInterface

# Definisikan fungsi alat
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y if y != 0 else 0

def power(x, y):
    return x ** y

def sqrt(x):
    return x ** 0.5 if x >= 0 else 0

def log(x):
    return np.log(x) if x > 0 else 0

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

# Buat tugas matematika kompleks
def create_complex_math_task():
    """
    Buat tugas matematika kompleks.
    
    Returns:
        Tugas matematika
    """
    problems = []
    
    # Operasi aritmatika dasar
    for _ in range(5):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        operations = [
            (f"{a} + {b}", a + b),
            (f"{a} - {b}", a - b),
            (f"{a} * {b}", a * b),
            (f"{a} / {b}", a / b if b != 0 else 0)
        ]
        problems.extend([{"expression": expr, "result": res} for expr, res in operations])
    
    # Operasi matematika lanjutan
    for _ in range(3):
        a = random.randint(1, 5)
        b = random.randint(1, 3)
        operations = [
            (f"power {a} {b}", a ** b),
            (f"sqrt {a*a}", a),
            (f"log {a*10}", np.log(a*10)),
            (f"sin {a}", np.sin(a)),
            (f"cos {a}", np.cos(a))
        ]
        problems.extend([{"expression": expr, "result": res} for expr, res in operations])
    
    return {
        "type": "complex_arithmetic",
        "problems": problems
    }

# Fungsi evaluasi multi-objektif
def evaluate_agent_multi_objective(agent, task) -> List[float]:
    """
    Evaluasi agen pada tugas dengan beberapa objektif.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Daftar skor evaluasi [akurasi, efisiensi, keragaman_alat]
    """
    if "problems" not in task:
        return [0.0, 0.0, 0.0]
    
    correct = 0
    total = len(task["problems"])
    tools_used = set()
    
    for problem in task["problems"]:
        expression = problem["expression"]
        expected = problem["result"]
        
        try:
            # Parse expression
            parts = expression.split()
            if len(parts) >= 2:
                # Tentukan operasi dan parameter
                if parts[0] in ["add", "subtract", "multiply", "divide", "power"]:
                    if len(parts) == 3:
                        op = parts[0]
                        a = float(parts[1])
                        b = float(parts[2])
                        
                        # Cari alat yang sesuai
                        tool = None
                        for t in agent.tools:
                            if t.name == op:
                                tool = t
                                tools_used.add(op)
                                break
                        
                        if tool:
                            # Gunakan alat
                            result = tool.function(a, b)
                            
                            # Periksa apakah hasil mendekati yang diharapkan
                            if isinstance(result, (int, float)) and abs(result - expected) < 0.001:
                                correct += 1
                
                elif parts[0] in ["sqrt", "log", "sin", "cos"]:
                    if len(parts) == 2:
                        op = parts[0]
                        a = float(parts[1])
                        
                        # Cari alat yang sesuai
                        tool = None
                        for t in agent.tools:
                            if t.name == op:
                                tool = t
                                tools_used.add(op)
                                break
                        
                        if tool:
                            # Gunakan alat
                            result = tool.function(a)
                            
                            # Periksa apakah hasil mendekati yang diharapkan
                            if isinstance(result, (int, float)) and abs(result - expected) < 0.001:
                                correct += 1
                
                else:
                    # Format tradisional (e.g., "2 + 3")
                    if len(parts) == 3:
                        a = float(parts[0])
                        op = parts[1]
                        b = float(parts[2])
                        
                        # Pilih alat berdasarkan operator
                        if op == "+":
                            tool_name = "add"
                        elif op == "-":
                            tool_name = "subtract"
                        elif op == "*":
                            tool_name = "multiply"
                        elif op == "/":
                            tool_name = "divide"
                        else:
                            continue
                        
                        # Cari alat yang sesuai
                        tool = None
                        for t in agent.tools:
                            if t.name == tool_name:
                                tool = t
                                tools_used.add(tool_name)
                                break
                        
                        if tool:
                            # Gunakan alat
                            result = tool.function(a, b)
                            
                            # Periksa apakah hasil mendekati yang diharapkan
                            if isinstance(result, (int, float)) and abs(result - expected) < 0.001:
                                correct += 1
            
        except Exception as e:
            pass
    
    # Hitung skor
    accuracy = correct / total if total > 0 else 0.0
    efficiency = 1.0 / (1.0 + len(agent.tools))  # Lebih sedikit alat = lebih efisien
    tool_diversity = len(tools_used) / 9.0  # 9 alat yang mungkin
    
    return [accuracy, efficiency, tool_diversity]

# Fungsi deskriptor perilaku
def behavior_descriptor(agent) -> List[float]:
    """
    Deskripsikan perilaku agen.
    
    Args:
        agent: Agen
        
    Returns:
        Deskriptor perilaku
    """
    # Deskriptor sederhana berdasarkan parameter agen
    descriptor = [
        agent.learning_rate,
        agent.exploration_rate,
        len(agent.tools) / 9.0  # Normalisasi jumlah alat
    ]
    return descriptor

def main():
    print("=== Darwin-Gödel Machine: Advanced Example ===")
    
    # Buat agen dasar dengan beberapa alat
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    agent.add_tool(Tool(name="add", function=add, description="Tambahkan dua angka"))
    agent.add_tool(Tool(name="subtract", function=subtract, description="Kurangkan dua angka"))
    
    # Inisialisasi komponen lanjutan
    evolution_strategy = TournamentSelection(tournament_size=3, offspring_size=5)
    mutation_operator = ParameterMutation(mutation_rate=0.2, mutation_strength=0.1)
    crossover_operator = BlendCrossover(alpha=0.5)
    
    # Definisikan fungsi objektif
    def accuracy_objective(agent, task):
        scores = evaluate_agent_multi_objective(agent, task)
        return scores[0]  # Akurasi
    
    def efficiency_objective(agent, task):
        scores = evaluate_agent_multi_objective(agent, task)
        return scores[1]  # Efisiensi
    
    def diversity_objective(agent, task):
        scores = evaluate_agent_multi_objective(agent, task)
        return scores[2]  # Keragaman alat
    
    # Buat fungsi fitness multi-objektif
    fitness_function = MultiObjectiveFitness(
        objectives=[accuracy_objective, efficiency_objective, diversity_objective],
        weights=[0.6, 0.2, 0.2]  # Prioritaskan akurasi
    )
    
    diversity_metric = BehavioralDiversity(behavior_descriptor=behavior_descriptor)
    archive_strategy = QualityDiversityArchive(
        capacity=20,
        feature_descriptor=behavior_descriptor,
        feature_dimensions=[(0.0, 1.0, 5), (0.0, 1.0, 5), (0.0, 1.0, 5)]
    )
    introspection_engine = IntrospectionEngine()
    adaptation_engine = AdaptationEngine()
    collaboration_engine = CollaborationEngine()
    llm_interface = LLMInterface()
    
    # Inisialisasi DGM
    dgm = DGM(initial_agent=agent, population_size=10)
    
    # Tetapkan komponen lanjutan
    dgm.set_evolution_strategy(evolution_strategy)
    dgm.set_mutation_operator(mutation_operator)
    dgm.set_crossover_operator(crossover_operator)
    dgm.set_fitness_function(fitness_function)
    dgm.set_diversity_metric(diversity_metric)
    dgm.set_archive_strategy(archive_strategy)
    
    # Tetapkan mesin introspeksi, adaptasi, kolaborasi, dan antarmuka LLM
    dgm.set_introspection_engine(introspection_engine)
    dgm.set_adaptation_engine(adaptation_engine)
    dgm.set_collaboration_engine(collaboration_engine)
    dgm.set_llm_interface(llm_interface)
    
    # Buat tugas
    task = create_complex_math_task()
    
    # Evaluasi agen awal
    initial_scores = evaluate_agent_multi_objective(agent, task)
    print(f"Initial agent scores: Accuracy={initial_scores[0]:.4f}, Efficiency={initial_scores[1]:.4f}, Tool Diversity={initial_scores[2]:.4f}")
    
    # Jalankan evolusi
    print("\nEvolving for 10 generations...")
    dgm.evolve(generations=10, task=task)
    
    # Dapatkan agen terbaik
    best_agent = dgm.get_best_agent()
    
    # Evaluasi agen terbaik
    best_scores = evaluate_agent_multi_objective(best_agent, task)
    print(f"\nBest agent scores: Accuracy={best_scores[0]:.4f}, Efficiency={best_scores[1]:.4f}, Tool Diversity={best_scores[2]:.4f}")
    
    # Cetak alat agen terbaik
    print("\nBest agent tools:")
    for tool in best_agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen terbaik
    print("\nBest agent parameters:")
    print(f"  - Memory capacity: {best_agent.memory_capacity}")
    print(f"  - Learning rate: {best_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {best_agent.exploration_rate:.4f}")
    
    # Lakukan introspeksi
    print("\nPerforming introspection...")
    introspection_results = dgm.introspect()
    print(f"Introspection results: {introspection_results}")
    
    # Tingkatkan DGM
    print("\nImproving DGM based on introspection...")
    improved = dgm.improve()
    print(f"DGM improved: {improved}")
    
    # Jalankan evolusi lagi setelah peningkatan
    print("\nEvolving for 5 more generations after improvement...")
    dgm.evolve(generations=5, task=task)
    
    # Dapatkan agen terbaik setelah peningkatan
    improved_best_agent = dgm.get_best_agent()
    
    # Evaluasi agen terbaik setelah peningkatan
    improved_best_scores = evaluate_agent_multi_objective(improved_best_agent, task)
    print(f"\nImproved best agent scores: Accuracy={improved_best_scores[0]:.4f}, Efficiency={improved_best_scores[1]:.4f}, Tool Diversity={improved_best_scores[2]:.4f}")
    
    # Kolaborasi
    print("\nPerforming collaboration...")
    collaborative_solution = dgm.collaborate(task)
    print(f"Collaborative solution quality: {collaborative_solution if collaborative_solution else 'None'}")

if __name__ == "__main__":
    main()