"""
Contoh penggunaan metrik keragaman dalam Darwin-Gödel Machine.

Contoh ini mendemonstrasikan bagaimana metrik keragaman dapat digunakan
untuk mempertahankan keragaman populasi selama evolusi.
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.core.diversity_metrics import BehavioralDiversity, GenotypeDiversity
from simple_dgm.core.evolution_strategies import TournamentSelection
from simple_dgm.core.mutation_operators import ParameterMutation
from simple_dgm.core.crossover_operators import BlendCrossover
from simple_dgm.core.fitness_functions import FitnessFunction

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
    return float('nan') if x <= 0 else (x.log() if hasattr(x, 'log') else __import__('math').log(x))

def sin(x):
    return __import__('math').sin(x)

def cos(x):
    return __import__('math').cos(x)

# Fungsi deskriptor perilaku
def behavior_descriptor(agent):
    """
    Deskriptor perilaku agen.
    
    Args:
        agent: Agen yang akan dideskripsikan
        
    Returns:
        Deskriptor perilaku
    """
    # Hitung proporsi jenis alat
    tool_types = {
        "arithmetic": ["add", "subtract", "multiply", "divide"],
        "power": ["power", "sqrt"],
        "trigonometric": ["sin", "cos"],
        "logarithmic": ["log"]
    }
    
    # Inisialisasi proporsi
    proportions = []
    
    # Hitung proporsi untuk setiap jenis
    for type_name, tool_names in tool_types.items():
        count = sum(1 for tool in agent.tools if tool.name in tool_names)
        proportion = count / max(1, len(agent.tools))
        proportions.append(proportion)
    
    # Tambahkan parameter agen yang dinormalisasi
    proportions.append(agent.learning_rate)
    proportions.append(agent.exploration_rate)
    proportions.append(agent.memory_capacity / 10.0)  # Normalisasi
    
    return proportions

# Fungsi evaluasi
def evaluate_agent(agent, task):
    """
    Evaluasi agen pada tugas.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Skor evaluasi
    """
    if "problems" not in task:
        return 0.0
    
    correct = 0
    total = len(task["problems"])
    
    for problem in task["problems"]:
        expression = problem["expression"]
        expected = problem["result"]
        
        try:
            # Parse expression
            parts = expression.split()
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
                elif op == "^":
                    tool_name = "power"
                else:
                    continue
                
                # Cari alat yang sesuai
                tool = None
                for t in agent.tools:
                    if t.name == tool_name:
                        tool = t
                        break
                
                if tool:
                    # Gunakan alat
                    result = tool.function(a, b)
                    
                    # Periksa apakah hasil mendekati yang diharapkan
                    if isinstance(result, (int, float)) and abs(result - expected) < 0.001:
                        correct += 1
            
            # Format "op a" untuk operasi unary
            elif len(parts) == 2:
                op = parts[0]
                a = float(parts[1])
                
                # Pilih alat berdasarkan operator
                if op == "sqrt":
                    tool_name = "sqrt"
                elif op == "log":
                    tool_name = "log"
                elif op == "sin":
                    tool_name = "sin"
                elif op == "cos":
                    tool_name = "cos"
                else:
                    continue
                
                # Cari alat yang sesuai
                tool = None
                for t in agent.tools:
                    if t.name == tool_name:
                        tool = t
                        break
                
                if tool:
                    # Gunakan alat
                    result = tool.function(a)
                    
                    # Periksa apakah hasil mendekati yang diharapkan
                    if isinstance(result, (int, float)) and abs(result - expected) < 0.001:
                        correct += 1
            
        except Exception as e:
            print(f"Error evaluating {expression}: {e}")
    
    return correct / total if total > 0 else 0.0

# Buat tugas matematika
def create_math_task():
    """
    Buat tugas matematika.
    
    Returns:
        Tugas matematika
    """
    problems = []
    
    # Operasi aritmatika
    for _ in range(3):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        operations = [
            (f"{a} + {b}", a + b),
            (f"{a} - {b}", a - b),
            (f"{a} * {b}", a * b),
            (f"{a} / {b}", a / b if b != 0 else 0),
            (f"{a} ^ {b}", a ** b)
        ]
        problems.extend([{"expression": expr, "result": res} for expr, res in operations])
    
    # Operasi unary
    for _ in range(2):
        a = random.randint(1, 5)
        operations = [
            (f"sqrt {a*a}", a),
            (f"log {a}", __import__('math').log(a)),
            (f"sin {0}", 0),
            (f"cos {0}", 1)
        ]
        problems.extend([{"expression": expr, "result": res} for expr, res in operations])
    
    return {
        "type": "arithmetic",
        "problems": problems
    }

# Fungsi fitness kustom
class CustomFitness(FitnessFunction):
    """
    Fungsi fitness kustom yang menggabungkan akurasi dan keragaman.
    """
    
    def __init__(self, diversity_metric, diversity_weight=0.3):
        """
        Inisialisasi fungsi fitness kustom.
        
        Args:
            diversity_metric: Metrik keragaman
            diversity_weight: Bobot keragaman dalam fitness
        """
        super().__init__()
        self.diversity_metric = diversity_metric
        self.diversity_weight = diversity_weight
        self.population = []
        self.accuracy_scores = {}
    
    def set_population(self, population):
        """
        Tetapkan populasi untuk perhitungan keragaman.
        
        Args:
            population: Populasi agen
        """
        self.population = population
    
    def _evaluate(self, individual, task):
        """
        Evaluasi individu.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Nilai fitness
        """
        # Hitung akurasi
        accuracy = evaluate_agent(individual, task)
        self.accuracy_scores[id(individual)] = accuracy
        
        # Jika populasi kosong, kembalikan hanya akurasi
        if not self.population:
            return accuracy
        
        # Hitung keragaman
        # Buat salinan populasi tanpa individu saat ini
        temp_population = [agent for agent in self.population if id(agent) != id(individual)]
        
        # Jika populasi hanya berisi individu saat ini, kembalikan hanya akurasi
        if not temp_population:
            return accuracy
        
        # Tambahkan individu saat ini
        temp_population.append(individual)
        
        # Hitung keragaman
        diversity = self.diversity_metric.measure(temp_population)
        
        # Gabungkan akurasi dan keragaman
        fitness = (1 - self.diversity_weight) * accuracy + self.diversity_weight * diversity
        
        return fitness

def main():
    print("=== Darwin-Gödel Machine: Diversity Metrics Example ===")
    
    # Buat agen dasar dengan beberapa alat
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    agent.add_tool(Tool(name="add", function=add, description="Tambahkan dua angka"))
    agent.add_tool(Tool(name="subtract", function=subtract, description="Kurangkan dua angka"))
    
    # Buat tugas
    task = create_math_task()
    
    # Evaluasi agen awal
    initial_score = evaluate_agent(agent, task)
    print(f"Initial agent score: {initial_score:.4f}")
    
    # Inisialisasi metrik keragaman
    behavioral_diversity = BehavioralDiversity(behavior_descriptor=behavior_descriptor)
    
    # Fungsi ekstraksi genotipe
    def genotype_extractor(agent):
        genotype = [
            agent.learning_rate,
            agent.exploration_rate,
            agent.memory_capacity / 10.0  # Normalisasi
        ]
        
        # Tambahkan informasi alat (0 jika tidak ada, 1 jika ada)
        tool_names = ["add", "subtract", "multiply", "divide", "power", "sqrt", "log", "sin", "cos"]
        for name in tool_names:
            genotype.append(1.0 if any(t.name == name for t in agent.tools) else 0.0)
        
        return genotype
    
    genotype_diversity = GenotypeDiversity(genotype_extractor=genotype_extractor)
    
    # Jalankan evolusi dengan dan tanpa metrik keragaman
    print("\nRunning evolution without diversity metrics...")
    dgm_without_diversity = DGM(initial_agent=agent, population_size=20)
    
    # Tetapkan strategi evolusi
    dgm_without_diversity.set_evolution_strategy(TournamentSelection(tournament_size=3, offspring_size=10))
    dgm_without_diversity.set_mutation_operator(ParameterMutation(mutation_rate=0.2, mutation_strength=0.1))
    dgm_without_diversity.set_crossover_operator(BlendCrossover(alpha=0.5))
    
    # Jalankan evolusi
    start_time = time.time()
    dgm_without_diversity.evolve(generations=20, task=task)
    time_without_diversity = time.time() - start_time
    
    # Dapatkan agen terbaik
    best_agent_without_diversity = dgm_without_diversity.get_best_agent()
    best_score_without_diversity = evaluate_agent(best_agent_without_diversity, task)
    
    print(f"Best agent score without diversity: {best_score_without_diversity:.4f}")
    print(f"Evolution time without diversity: {time_without_diversity:.2f} seconds")
    
    # Hitung keragaman populasi
    population_without_diversity = [info["agent"] for info in dgm_without_diversity.archive.values()]
    behavioral_diversity_without = behavioral_diversity.measure(population_without_diversity)
    genotype_diversity_without = genotype_diversity.measure(population_without_diversity)
    
    print(f"Behavioral diversity without diversity metrics: {behavioral_diversity_without:.4f}")
    print(f"Genotype diversity without diversity metrics: {genotype_diversity_without:.4f}")
    
    # Jalankan evolusi dengan metrik keragaman
    print("\nRunning evolution with diversity metrics...")
    dgm_with_diversity = DGM(initial_agent=agent, population_size=20)
    
    # Tetapkan strategi evolusi
    dgm_with_diversity.set_evolution_strategy(TournamentSelection(tournament_size=3, offspring_size=10))
    dgm_with_diversity.set_mutation_operator(ParameterMutation(mutation_rate=0.2, mutation_strength=0.1))
    dgm_with_diversity.set_crossover_operator(BlendCrossover(alpha=0.5))
    
    # Tetapkan metrik keragaman
    dgm_with_diversity.set_diversity_metric(behavioral_diversity)
    
    # Tetapkan fungsi fitness kustom
    custom_fitness = CustomFitness(behavioral_diversity, diversity_weight=0.3)
    dgm_with_diversity.set_fitness_function(custom_fitness)
    
    # Jalankan evolusi
    start_time = time.time()
    
    # Simpan metrik keragaman selama evolusi
    diversity_history = []
    
    for generation in range(20):
        # Perbarui populasi dalam fungsi fitness
        population = [info["agent"] for info in dgm_with_diversity.archive.values()]
        custom_fitness.set_population(population)
        
        # Jalankan satu generasi
        dgm_with_diversity.evolve(generations=1, task=task)
        
        # Hitung dan simpan keragaman
        population = [info["agent"] for info in dgm_with_diversity.archive.values()]
        behavioral_div = behavioral_diversity.measure(population)
        genotype_div = genotype_diversity.measure(population)
        
        diversity_history.append((behavioral_div, genotype_div))
    
    time_with_diversity = time.time() - start_time
    
    # Dapatkan agen terbaik
    best_agent_with_diversity = dgm_with_diversity.get_best_agent()
    best_score_with_diversity = evaluate_agent(best_agent_with_diversity, task)
    
    print(f"Best agent score with diversity: {best_score_with_diversity:.4f}")
    print(f"Evolution time with diversity: {time_with_diversity:.2f} seconds")
    
    # Hitung keragaman populasi
    population_with_diversity = [info["agent"] for info in dgm_with_diversity.archive.values()]
    behavioral_diversity_with = behavioral_diversity.measure(population_with_diversity)
    genotype_diversity_with = genotype_diversity.measure(population_with_diversity)
    
    print(f"Behavioral diversity with diversity metrics: {behavioral_diversity_with:.4f}")
    print(f"Genotype diversity with diversity metrics: {genotype_diversity_with:.4f}")
    
    # Bandingkan alat agen terbaik
    print("\nBest agent tools without diversity:")
    for tool in best_agent_without_diversity.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    print("\nBest agent tools with diversity:")
    for tool in best_agent_with_diversity.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Bandingkan parameter agen terbaik
    print("\nBest agent parameters without diversity:")
    print(f"  - Memory capacity: {best_agent_without_diversity.memory_capacity}")
    print(f"  - Learning rate: {best_agent_without_diversity.learning_rate:.4f}")
    print(f"  - Exploration rate: {best_agent_without_diversity.exploration_rate:.4f}")
    
    print("\nBest agent parameters with diversity:")
    print(f"  - Memory capacity: {best_agent_with_diversity.memory_capacity}")
    print(f"  - Learning rate: {best_agent_with_diversity.learning_rate:.4f}")
    print(f"  - Exploration rate: {best_agent_with_diversity.exploration_rate:.4f}")
    
    # Plot keragaman selama evolusi
    plt.figure(figsize=(10, 6))
    
    generations = range(1, len(diversity_history) + 1)
    behavioral_divs = [d[0] for d in diversity_history]
    genotype_divs = [d[1] for d in diversity_history]
    
    plt.plot(generations, behavioral_divs, 'b-', label='Behavioral Diversity')
    plt.plot(generations, genotype_divs, 'r-', label='Genotype Diversity')
    
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Population Diversity During Evolution')
    plt.legend()
    plt.grid(True)
    
    # Simpan plot
    plt.savefig('diversity_evolution.png')
    print("\nDiversity evolution plot saved as 'diversity_evolution.png'")

if __name__ == "__main__":
    main()