"""
Contoh penggunaan dasar Darwin-Gödel Machine.

Contoh ini mendemonstrasikan penggunaan dasar DGM untuk menyelesaikan
tugas matematika sederhana.
"""

import random
from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool

# Definisikan fungsi alat
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y if y != 0 else 0

# Buat tugas matematika
def create_math_task():
    """
    Buat tugas matematika sederhana.
    
    Returns:
        Tugas matematika
    """
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    operations = [
        (f"{a} + {b}", a + b),
        (f"{a} - {b}", a - b),
        (f"{a} * {b}", a * b),
        (f"{a} / {b}", a / b if b != 0 else 0)
    ]
    return {
        "type": "arithmetic",
        "problems": [{"expression": expr, "result": res} for expr, res in operations]
    }

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

def main():
    print("=== Darwin-Gödel Machine: Basic Example ===")
    
    # Buat agen dasar dengan beberapa alat
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    agent.add_tool(Tool(name="add", function=add, description="Tambahkan dua angka"))
    agent.add_tool(Tool(name="subtract", function=subtract, description="Kurangkan dua angka"))
    
    # Inisialisasi DGM
    dgm = DGM(initial_agent=agent, population_size=10)
    
    # Buat tugas
    task = create_math_task()
    
    # Evaluasi agen awal
    initial_score = evaluate_agent(agent, task)
    print(f"Initial agent score: {initial_score:.4f}")
    
    # Jalankan evolusi
    print("\nEvolving for 10 generations...")
    dgm.evolve(generations=10, task=task)
    
    # Dapatkan agen terbaik
    best_agent = dgm.get_best_agent()
    
    # Evaluasi agen terbaik
    best_score = evaluate_agent(best_agent, task)
    print(f"Best agent score: {best_score:.4f}")
    
    # Cetak alat agen terbaik
    print("\nBest agent tools:")
    for tool in best_agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen terbaik
    print("\nBest agent parameters:")
    print(f"  - Memory capacity: {best_agent.memory_capacity}")
    print(f"  - Learning rate: {best_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {best_agent.exploration_rate:.4f}")
    
    # Uji agen terbaik pada tugas baru
    print("\nTesting best agent on a new task...")
    new_task = create_math_task()
    
    print("New task problems:")
    for problem in new_task["problems"]:
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
                    print(f"  - {expression} = Unknown operator (expected: {expected:.4f})")
                    continue
                
                # Cari alat yang sesuai
                tool = None
                for t in best_agent.tools:
                    if t.name == tool_name:
                        tool = t
                        break
                
                if tool:
                    # Gunakan alat
                    result = tool.function(a, b)
                    print(f"  - {expression} = {result:.4f} (expected: {expected:.4f})")
                else:
                    print(f"  - {expression} = Tool not available (expected: {expected:.4f})")
            else:
                print(f"  - {expression} = Invalid format (expected: {expected:.4f})")
        except Exception as e:
            print(f"  - {expression} = Error: {e} (expected: {expected:.4f})")

if __name__ == "__main__":
    main()