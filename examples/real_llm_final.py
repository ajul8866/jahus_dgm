"""
Contoh penggunaan integrasi LLM yang sebenarnya dengan Darwin-Gödel Machine.

Contoh ini mendemonstrasikan penggunaan integrasi LLM yang sebenarnya untuk:
- Peningkatan agen
- Pembuatan kode
- Pemecahan masalah
- Ekstraksi pengetahuan
- Modifikasi diri
"""

import os
import random
import time
import json
from typing import List, Dict, Any, Tuple

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.core.llm_integration_real import (
    LLMInterface, CodeGeneration, ProblemSolving, 
    KnowledgeExtraction, SelfModification
)

# Definisikan fungsi alat
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y if y != 0 else 0

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
    
    return {
        "type": "arithmetic",
        "problems": problems
    }

def main():
    print("=== Darwin-Gödel Machine: Real LLM Integration Example ===")
    
    # Dapatkan API key dari environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return
    
    # Inisialisasi antarmuka LLM
    llm_interface = LLMInterface(api_key=api_key, model="gpt-3.5-turbo")
    
    # Inisialisasi komponen LLM
    code_generation = CodeGeneration(llm_interface)
    problem_solving = ProblemSolving(llm_interface)
    knowledge_extraction = KnowledgeExtraction(llm_interface)
    self_modification = SelfModification(llm_interface)
    
    # Buat agen dasar dengan beberapa alat
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    agent.add_tool(Tool(name="add", function=add, description="Tambahkan dua angka"))
    agent.add_tool(Tool(name="subtract", function=subtract, description="Kurangkan dua angka"))
    
    # Buat tugas
    task = create_math_task()
    task_description = "Solve arithmetic problems involving addition, subtraction, multiplication, and division."
    
    # Evaluasi agen awal
    initial_score = evaluate_agent(agent, task)
    print(f"Initial agent score: {initial_score:.4f}")
    
    # Cetak alat agen awal
    print("\nInitial agent tools:")
    for tool in agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen awal
    print("\nInitial agent parameters:")
    print(f"  - Memory capacity: {agent.memory_capacity}")
    print(f"  - Learning rate: {agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {agent.exploration_rate:.4f}")
    
    # Demonstrasi peningkatan agen
    print("\n=== Agent Improvement Demo ===")
    improved_agent = llm_interface.improve_agent(agent, task_description)
    
    # Evaluasi agen yang ditingkatkan
    improved_score = evaluate_agent(improved_agent, task)
    print(f"\nImproved agent score: {improved_score:.4f}")
    
    # Cetak alat agen yang ditingkatkan
    print("\nImproved agent tools:")
    for tool in improved_agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen yang ditingkatkan
    print("\nImproved agent parameters:")
    print(f"  - Memory capacity: {improved_agent.memory_capacity}")
    print(f"  - Learning rate: {improved_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {improved_agent.exploration_rate:.4f}")
    
    # Demonstrasi pembuatan kode
    print("\n=== Code Generation Demo ===")
    function_code = code_generation.generate_function(
        function_name="calculate_fibonacci",
        description="Calculate the nth Fibonacci number efficiently using dynamic programming",
        parameters=[
            {"name": "n", "type": "int", "description": "The position in the Fibonacci sequence (0-indexed)"}
        ],
        return_type="int"
    )
    
    print("Generated function:")
    print(function_code)
    
    # Demonstrasi pemecahan masalah
    print("\n=== Problem Solving Demo ===")
    problem = "Find the maximum subarray sum in an array of integers."
    
    print(f"Analyzing problem: {problem}")
    analysis = problem_solving.analyze_problem(problem)
    
    print("Problem analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Demonstrasi ekstraksi pengetahuan
    print("\n=== Knowledge Extraction Demo ===")
    text = """
    Darwin-Gödel Machine (DGM) is a self-improving AI system that combines Darwinian evolution
    and Gödelian self-reference. It uses evolutionary algorithms to evolve a population of agents,
    and self-reference to modify its own code. The system has several components:
    
    1. Agents: The basic units that solve problems
    2. Evolution: The process of selecting and modifying agents
    3. Introspection: The ability to analyze and understand its own code
    4. Adaptation: The ability to adapt to changing environments and tasks
    5. Collaboration: The ability to collaborate with other agents
    
    DGM can be applied to various domains, including software development, scientific research,
    and creative tasks.
    """
    
    print("Extracting concepts...")
    concepts = knowledge_extraction.extract_concepts(text)
    
    print("Extracted concepts:")
    print(json.dumps(concepts[:2], indent=2))  # Tampilkan 2 konsep pertama
    
    # Demonstrasi modifikasi diri
    print("\n=== Self Modification Demo ===")
    code = """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
    """
    
    print("Analyzing code...")
    analysis = self_modification.analyze_code(code)
    
    print("Code analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Demonstrasi generasi strategi evolusi
    print("\n=== Evolution Strategy Generation Demo ===")
    strategy_info = llm_interface.generate_evolution_strategy(task_description)
    
    print("Generated evolution strategy:")
    print(json.dumps(strategy_info, indent=2))
    
    # Demonstrasi generasi operator mutasi
    print("\n=== Mutation Operator Generation Demo ===")
    mutation_info = llm_interface.generate_mutation_operator(task_description)
    
    print("Generated mutation operator:")
    print(json.dumps(mutation_info, indent=2))
    
    # Demonstrasi generasi operator crossover
    print("\n=== Crossover Operator Generation Demo ===")
    crossover_info = llm_interface.generate_crossover_operator(task_description)
    
    print("Generated crossover operator:")
    print(json.dumps(crossover_info, indent=2))
    
    # Analisis hasil evolusi
    print("\n=== Evolution Results Analysis ===")
    evolution_results = {
        "initial_score": initial_score,
        "improved_score": improved_score,
        "best_score": improved_score,  # Gunakan improved_score sebagai best_score
        "evolution_time": 0.5,  # Nilai contoh
        "generations": 5,
        "population_size": 10,
        "best_agent_tools": [tool.name for tool in improved_agent.tools],
        "best_agent_parameters": {
            "memory_capacity": improved_agent.memory_capacity,
            "learning_rate": improved_agent.learning_rate,
            "exploration_rate": improved_agent.exploration_rate
        }
    }
    
    analysis = llm_interface.analyze_evolution_results(evolution_results)
    
    print("Evolution results analysis:")
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main()