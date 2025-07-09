"""
Contoh evolusi canggih untuk Darwin-Gödel Machine.

Contoh ini mendemonstrasikan penggunaan fitur-fitur canggih DGM seperti:
- Strategi evolusi NSGA-II
- Operator mutasi adaptif
- Operator crossover blend
- Fungsi fitness multi-objektif
- Metrik keragaman perilaku
- Strategi arsip kualitas-keragaman
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.core.evolution_strategies import NSGA2Selection
from simple_dgm.core.mutation_operators import AdaptiveMutation, ParameterMutation, StructuralMutation
from simple_dgm.core.crossover_operators import BlendCrossover
from simple_dgm.core.fitness_functions import MultiObjectiveFitness
from simple_dgm.core.diversity_metrics import BehavioralDiversity
from simple_dgm.core.archive_strategies import QualityDiversityArchive
from simple_dgm.utils.visualization import visualize_evolution_tree, visualize_performance


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
    try:
        return x ** y
    except:
        return 0

def log(x):
    try:
        return math.log(x) if x > 0 else 0
    except:
        return 0

def sin(x):
    try:
        return math.sin(x)
    except:
        return 0

def cos(x):
    try:
        return math.cos(x)
    except:
        return 0


# Definisikan tugas
def generate_math_task(complexity: str = "medium") -> Dict[str, Any]:
    """
    Hasilkan tugas matematika.
    
    Args:
        complexity: Kompleksitas tugas ("easy", "medium", "hard")
        
    Returns:
        Tugas matematika
    """
    if complexity == "easy":
        # Tugas aritmatika sederhana
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        operations = [
            (f"{a} + {b}", a + b),
            (f"{a} - {b}", a - b),
            (f"{a} * {b}", a * b)
        ]
        return {
            "type": "arithmetic",
            "problems": [{"expression": expr, "result": res} for expr, res in operations],
            "complexity": 0.3
        }
    
    elif complexity == "hard":
        # Tugas matematika kompleks
        problems = []
        for _ in range(5):
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            c = random.randint(1, 10)
            operations = [
                (f"{a} * {b} + {c}", a * b + c),
                (f"{a} * ({b} + {c})", a * (b + c)),
                (f"{a} ** 2 + {b}", a ** 2 + b),
                (f"({a} + {b}) / {c if c > 0 else 1}", (a + b) / (c if c > 0 else 1))
            ]
            expr, res = random.choice(operations)
            problems.append({"expression": expr, "result": res})
        
        return {
            "type": "complex_math",
            "problems": problems,
            "complexity": 0.8
        }
    
    else:  # "medium"
        # Tugas matematika menengah
        problems = []
        for _ in range(3):
            a = random.randint(1, 15)
            b = random.randint(1, 15)
            operations = [
                (f"{a} + {b} * 2", a + b * 2),
                (f"{a} * {b} - {a}", a * b - a),
                (f"({a} + {b}) * 2", (a + b) * 2)
            ]
            expr, res = random.choice(operations)
            problems.append({"expression": expr, "result": res})
        
        return {
            "type": "intermediate_math",
            "problems": problems,
            "complexity": 0.5
        }


# Definisikan fungsi objektif
def accuracy_objective(agent: BaseAgent, task: Dict[str, Any]) -> float:
    """
    Objektif akurasi: seberapa akurat agen menyelesaikan masalah.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Nilai akurasi (0.0 - 1.0)
    """
    if "problems" not in task:
        return 0.0
    
    correct = 0
    total = len(task["problems"])
    
    for problem in task["problems"]:
        expression = problem["expression"]
        expected = problem["result"]
        
        try:
            # Coba evaluasi ekspresi menggunakan agen
            result = agent.solve(expression)
            
            # Periksa apakah hasil mendekati yang diharapkan
            if isinstance(result, (int, float)) and abs(result - expected) < 0.001:
                correct += 1
        except:
            pass
    
    return correct / total if total > 0 else 0.0


def efficiency_objective(agent: BaseAgent, task: Dict[str, Any]) -> float:
    """
    Objektif efisiensi: seberapa efisien agen menyelesaikan masalah.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Nilai efisiensi (0.0 - 1.0)
    """
    if "problems" not in task:
        return 0.0
    
    total_time = 0
    total = len(task["problems"])
    
    for problem in task["problems"]:
        expression = problem["expression"]
        
        try:
            # Ukur waktu eksekusi
            start_time = time.time()
            agent.solve(expression)
            end_time = time.time()
            
            # Tambahkan waktu eksekusi
            total_time += end_time - start_time
        except:
            # Jika gagal, tambahkan penalti waktu
            total_time += 1.0
    
    # Konversi waktu ke efisiensi (semakin cepat, semakin efisien)
    avg_time = total_time / total if total > 0 else 1.0
    efficiency = 1.0 / (1.0 + avg_time)  # Nilai antara 0 dan 1
    
    return efficiency


def complexity_objective(agent: BaseAgent, task: Dict[str, Any]) -> float:
    """
    Objektif kompleksitas: seberapa kompleks agen.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Nilai kompleksitas (0.0 - 1.0)
    """
    # Hitung kompleksitas berdasarkan jumlah alat
    tool_count = len(agent.tools) if hasattr(agent, "tools") else 0
    
    # Normalisasi kompleksitas
    complexity = min(1.0, tool_count / 10.0)
    
    return complexity


# Definisikan deskriptor perilaku
def behavior_descriptor(agent: BaseAgent) -> List[float]:
    """
    Deskripsikan perilaku agen.
    
    Args:
        agent: Agen yang akan dideskripsikan
        
    Returns:
        Deskriptor perilaku
    """
    # Deskriptor berdasarkan parameter agen
    descriptor = [
        agent.learning_rate if hasattr(agent, "learning_rate") else 0.01,
        agent.exploration_rate if hasattr(agent, "exploration_rate") else 0.1
    ]
    
    # Tambahkan informasi alat
    if hasattr(agent, "tools"):
        tool_types = {
            "arithmetic": ["add", "subtract", "multiply", "divide"],
            "power": ["power", "log"],
            "trigonometric": ["sin", "cos"]
        }
        
        # Hitung proporsi setiap tipe alat
        tool_names = [tool.name for tool in agent.tools]
        for category, names in tool_types.items():
            count = sum(1 for name in tool_names if name in names)
            descriptor.append(count / max(1, len(tool_names)))
    
    return descriptor


def main():
    print("=== Darwin-Gödel Machine: Advanced Evolution ===")
    
    # Buat agen dasar dengan beberapa alat
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    agent.add_tool(Tool(name="add", function=add, description="Tambahkan dua angka"))
    agent.add_tool(Tool(name="subtract", function=subtract, description="Kurangkan dua angka"))
    agent.add_tool(Tool(name="multiply", function=multiply, description="Kalikan dua angka"))
    
    # Inisialisasi DGM dengan ukuran populasi yang lebih besar
    dgm = DGM(initial_agent=agent, population_size=50)
    
    # Tetapkan strategi evolusi NSGA-II
    dgm.set_evolution_strategy(NSGA2Selection(population_size=50, offspring_size=30))
    
    # Tetapkan operator mutasi adaptif
    parameter_mutation = ParameterMutation(mutation_rate=0.3, mutation_strength=0.2)
    structural_mutation = StructuralMutation(
        mutation_rate=0.2,
        available_tools=[
            {"name": "divide", "function": divide, "description": "Bagi dua angka"},
            {"name": "power", "function": power, "description": "Pangkatkan angka pertama dengan angka kedua"},
            {"name": "log", "function": log, "description": "Logaritma natural dari angka"},
            {"name": "sin", "function": sin, "description": "Sinus dari angka"},
            {"name": "cos", "function": cos, "description": "Kosinus dari angka"}
        ]
    )
    dgm.set_mutation_operator(AdaptiveMutation(
        initial_rate=0.3,
        operators=[parameter_mutation, structural_mutation]
    ))
    
    # Tetapkan operator crossover blend
    dgm.set_crossover_operator(BlendCrossover(crossover_rate=0.7, alpha=0.5))
    
    # Tetapkan fungsi fitness multi-objektif
    dgm.set_fitness_function(MultiObjectiveFitness(
        objectives=[accuracy_objective, efficiency_objective, complexity_objective],
        weights=[0.6, 0.3, 0.1]
    ))
    
    # Tetapkan metrik keragaman perilaku
    dgm.set_diversity_metric(BehavioralDiversity(
        behavior_descriptor=behavior_descriptor,
        distance_metric="euclidean"
    ))
    
    # Tetapkan strategi arsip kualitas-keragaman
    dgm.set_archive_strategy(QualityDiversityArchive(
        capacity=100,
        feature_descriptor=behavior_descriptor,
        feature_dimensions=[
            (0.0, 0.1, 10),  # learning_rate
            (0.0, 0.5, 10),  # exploration_rate
            (0.0, 1.0, 5),   # proporsi alat aritmatika
            (0.0, 1.0, 5),   # proporsi alat pangkat
            (0.0, 1.0, 5)    # proporsi alat trigonometri
        ]
    ))
    
    # Hasilkan tugas matematika
    task = generate_math_task("medium")
    
    # Jalankan evolusi
    generations = 20
    print(f"Evolving for {generations} generations...")
    dgm.evolve(generations=generations, task=task, parallel=True, num_workers=4)
    
    # Dapatkan agen terbaik
    best_agent = dgm.get_best_agent()
    
    # Uji agen terbaik pada tugas baru
    print("\nTesting best agent on new tasks...")
    
    for complexity in ["easy", "medium", "hard"]:
        test_task = generate_math_task(complexity)
        accuracy = accuracy_objective(best_agent, test_task)
        efficiency = efficiency_objective(best_agent, test_task)
        
        print(f"Task complexity: {complexity}")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Efficiency: {efficiency:.4f}")
    
    # Hasilkan visualisasi
    print("\nGenerating visualizations...")
    visualize_evolution_tree(dgm, output_path="advanced_evolution_tree.png", show_plot=False)
    
    # Visualisasi kinerja
    stats = dgm.get_statistics()
    plt.figure(figsize=(12, 8))
    
    # Plot fitness terbaik dan rata-rata
    plt.subplot(2, 2, 1)
    plt.plot(stats["best_fitness"], label="Best Fitness")
    plt.plot(stats["avg_fitness"], label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Evolution")
    plt.legend()
    
    # Plot keragaman
    if "diversity" in stats and stats["diversity"]:
        plt.subplot(2, 2, 2)
        plt.plot(stats["diversity"])
        plt.xlabel("Generation")
        plt.ylabel("Diversity")
        plt.title("Population Diversity")
    
    # Plot waktu eksekusi
    if "execution_time" in stats and stats["execution_time"]:
        plt.subplot(2, 2, 3)
        plt.plot(stats["execution_time"])
        plt.xlabel("Generation")
        plt.ylabel("Execution Time (s)")
        plt.title("Generation Execution Time")
    
    # Simpan visualisasi
    plt.tight_layout()
    plt.savefig("advanced_performance.png")
    print("Visualizations saved to 'advanced_evolution_tree.png' and 'advanced_performance.png'.")


if __name__ == "__main__":
    main()