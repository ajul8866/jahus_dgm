"""
Contoh introspeksi dan adaptasi dalam Darwin-Gödel Machine.

Contoh ini mendemonstrasikan bagaimana DGM dapat melakukan introspeksi
dan beradaptasi dengan lingkungan yang berubah.
"""

import random
import time
from typing import List, Dict, Any, Tuple

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.core.introspection import IntrospectionEngine
from simple_dgm.core.adaptation import AdaptationEngine

# Definisikan fungsi alat
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y if y != 0 else 0

# Implementasi mesin introspeksi kustom
class CustomIntrospectionEngine(IntrospectionEngine):
    """
    Mesin introspeksi kustom.
    """
    
    def analyze(self, dgm: DGM) -> Dict[str, Any]:
        """
        Analisis DGM.
        
        Args:
            dgm: DGM yang akan dianalisis
            
        Returns:
            Hasil analisis
        """
        # Dapatkan statistik dasar
        num_agents = len(dgm.archive)
        best_agent_id = dgm.get_best_agent_id()
        best_score = dgm.archive[best_agent_id]["score"] if best_agent_id else 0.0
        avg_score = sum(info["score"] for info in dgm.archive.values()) / num_agents if num_agents > 0 else 0.0
        
        # Analisis keragaman alat
        tool_counts = {}
        for agent_id, info in dgm.archive.items():
            agent = info["agent"]
            for tool in agent.tools:
                if tool.name not in tool_counts:
                    tool_counts[tool.name] = 0
                tool_counts[tool.name] += 1
        
        # Analisis parameter agen
        learning_rates = [info["agent"].learning_rate for info in dgm.archive.values()]
        exploration_rates = [info["agent"].exploration_rate for info in dgm.archive.values()]
        memory_capacities = [info["agent"].memory_capacity for info in dgm.archive.values()]
        
        avg_learning_rate = sum(learning_rates) / len(learning_rates) if learning_rates else 0.0
        avg_exploration_rate = sum(exploration_rates) / len(exploration_rates) if exploration_rates else 0.0
        avg_memory_capacity = sum(memory_capacities) / len(memory_capacities) if memory_capacities else 0.0
        
        # Analisis performa evolusi
        if hasattr(dgm, "evolution_stats") and "execution_time" in dgm.evolution_stats:
            avg_execution_time = sum(dgm.evolution_stats["execution_time"]) / len(dgm.evolution_stats["execution_time"]) if dgm.evolution_stats["execution_time"] else 0.0
        else:
            avg_execution_time = 0.0
        
        # Hasil analisis
        return {
            "num_agents": num_agents,
            "best_score": best_score,
            "avg_score": avg_score,
            "generation": dgm.current_generation,
            "tool_diversity": tool_counts,
            "avg_learning_rate": avg_learning_rate,
            "avg_exploration_rate": avg_exploration_rate,
            "avg_memory_capacity": avg_memory_capacity,
            "avg_execution_time": avg_execution_time,
            "timestamp": time.time()
        }
    
    def improve(self, dgm: DGM, analysis: Dict[str, Any]) -> DGM:
        """
        Tingkatkan DGM berdasarkan analisis.
        
        Args:
            dgm: DGM yang akan ditingkatkan
            analysis: Hasil analisis
            
        Returns:
            DGM yang ditingkatkan
        """
        # Salin DGM
        improved_dgm = dgm
        
        # Tingkatkan berdasarkan analisis
        if "tool_diversity" in analysis:
            # Jika keragaman alat rendah, tambahkan alat yang kurang
            tool_counts = analysis["tool_diversity"]
            
            # Periksa alat yang kurang
            missing_tools = []
            if "add" not in tool_counts or tool_counts["add"] < 3:
                missing_tools.append(("add", add, "Tambahkan dua angka"))
            if "subtract" not in tool_counts or tool_counts["subtract"] < 3:
                missing_tools.append(("subtract", subtract, "Kurangkan dua angka"))
            if "multiply" not in tool_counts or tool_counts["multiply"] < 3:
                missing_tools.append(("multiply", multiply, "Kalikan dua angka"))
            if "divide" not in tool_counts or tool_counts["divide"] < 3:
                missing_tools.append(("divide", divide, "Bagi dua angka"))
            
            # Jika ada alat yang kurang, tambahkan ke agen baru
            if missing_tools:
                # Buat agen baru dengan alat yang kurang
                new_agent = BaseAgent(
                    memory_capacity=int(analysis["avg_memory_capacity"]),
                    learning_rate=analysis["avg_learning_rate"],
                    exploration_rate=analysis["avg_exploration_rate"]
                )
                
                # Tambahkan alat yang kurang
                for name, function, description in missing_tools:
                    new_agent.add_tool(Tool(name=name, function=function, description=description))
                
                # Evaluasi dan tambahkan agen baru ke DGM
                score = 0.5  # Skor default
                agent_id = f"agent_{time.time()}"
                improved_dgm.archive[agent_id] = {
                    "agent": new_agent,
                    "score": score,
                    "parent_id": None
                }
        
        # Tingkatkan parameter evolusi
        if "avg_execution_time" in analysis and analysis["avg_execution_time"] > 0.1:
            # Jika waktu eksekusi terlalu lama, kurangi ukuran populasi
            if improved_dgm.population_size > 5:
                improved_dgm.population_size = max(5, improved_dgm.population_size - 5)
        
        return improved_dgm

# Implementasi mesin adaptasi kustom
class CustomAdaptationEngine(AdaptationEngine):
    """
    Mesin adaptasi kustom.
    """
    
    def adapt(self, dgm: DGM, environment: Dict[str, Any], task: Any) -> None:
        """
        Adaptasi DGM ke lingkungan.
        
        Args:
            dgm: DGM yang akan diadaptasi
            environment: Lingkungan
            task: Tugas
        """
        # Adaptasi berdasarkan sumber daya
        if "resources" in environment:
            resources = environment["resources"]
            
            # Jika sumber daya terbatas, kurangi ukuran populasi
            if resources.get("cpu", 1.0) < 0.5 or resources.get("memory", 1.0) < 0.5:
                dgm.population_size = max(5, dgm.population_size // 2)
            
            # Jika sumber daya berlimpah, tingkatkan ukuran populasi
            if resources.get("cpu", 1.0) > 0.8 and resources.get("memory", 1.0) > 0.8:
                dgm.population_size = min(100, dgm.population_size * 2)
        
        # Adaptasi berdasarkan waktu
        if "time" in environment:
            time_info = environment["time"]
            
            # Jika waktu eksekusi mendekati batas waktu, kurangi generasi
            if time_info.get("execution_time", 0) > 0.5 * time_info.get("timeout", float('inf')):
                # Tidak ada tindakan langsung yang dapat diambil, tetapi dapat memengaruhi keputusan evolusi
                pass
        
        # Adaptasi berdasarkan tugas
        if task and isinstance(task, dict) and "difficulty" in task:
            difficulty = task["difficulty"]
            
            # Jika tugas sulit, tingkatkan eksplorasi
            if difficulty == "hard":
                # Tingkatkan eksplorasi untuk semua agen
                for agent_id, info in dgm.archive.items():
                    agent = info["agent"]
                    agent.exploration_rate = min(0.5, agent.exploration_rate * 1.5)
            
            # Jika tugas mudah, kurangi eksplorasi
            elif difficulty == "easy":
                # Kurangi eksplorasi untuk semua agen
                for agent_id, info in dgm.archive.items():
                    agent = info["agent"]
                    agent.exploration_rate = max(0.01, agent.exploration_rate * 0.8)

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
def create_math_task(difficulty="medium"):
    """
    Buat tugas matematika.
    
    Args:
        difficulty: Tingkat kesulitan tugas ("easy", "medium", "hard")
        
    Returns:
        Tugas matematika
    """
    problems = []
    
    # Tentukan rentang angka berdasarkan kesulitan
    if difficulty == "easy":
        num_range = (1, 10)
        operations = ["+", "-"]
    elif difficulty == "medium":
        num_range = (1, 20)
        operations = ["+", "-", "*"]
    else:  # hard
        num_range = (1, 50)
        operations = ["+", "-", "*", "/"]
    
    # Buat masalah
    for _ in range(10):
        a = random.randint(*num_range)
        b = random.randint(*num_range)
        
        # Pastikan tidak ada pembagian dengan nol
        if "/" in operations:
            while b == 0:
                b = random.randint(*num_range)
        
        # Pilih operasi secara acak
        op = random.choice(operations)
        
        # Hitung hasil
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        elif op == "/":
            result = a / b
        
        # Tambahkan masalah
        problems.append({"expression": f"{a} {op} {b}", "result": result})
    
    return {
        "type": "arithmetic",
        "difficulty": difficulty,
        "problems": problems
    }

def main():
    print("=== Darwin-Gödel Machine: Introspection and Adaptation Example ===")
    
    # Buat agen dasar dengan beberapa alat
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    agent.add_tool(Tool(name="add", function=add, description="Tambahkan dua angka"))
    agent.add_tool(Tool(name="subtract", function=subtract, description="Kurangkan dua angka"))
    
    # Inisialisasi DGM
    dgm = DGM(initial_agent=agent, population_size=10)
    
    # Inisialisasi mesin introspeksi dan adaptasi
    introspection_engine = CustomIntrospectionEngine()
    adaptation_engine = CustomAdaptationEngine()
    
    # Tetapkan mesin introspeksi dan adaptasi ke DGM
    dgm.set_introspection_engine(introspection_engine)
    dgm.set_adaptation_engine(adaptation_engine)
    
    # Buat tugas awal (mudah)
    easy_task = create_math_task(difficulty="easy")
    
    # Evaluasi agen awal
    initial_score = evaluate_agent(agent, easy_task)
    print(f"Initial agent score on easy task: {initial_score:.4f}")
    
    # Jalankan evolusi pada tugas mudah
    print("\nEvolving on easy task...")
    
    # Buat lingkungan dengan sumber daya berlimpah
    environment = {
        "resources": {
            "cpu": 0.9,
            "memory": 0.9,
            "storage": 0.9
        },
        "time": {
            "execution_time": 0.0,
            "timeout": 60.0
        }
    }
    
    # Adaptasi ke lingkungan
    print("Adapting to environment...")
    dgm.adaptation_engine.adapt(dgm, environment, easy_task)
    
    # Jalankan evolusi
    dgm.evolve(generations=5, task=easy_task)
    
    # Lakukan introspeksi
    print("\nPerforming introspection...")
    analysis = dgm.introspect()
    print("Introspection results:")
    for key, value in analysis.items():
        if key != "tool_diversity":
            print(f"  - {key}: {value}")
        else:
            print(f"  - {key}:")
            for tool, count in value.items():
                print(f"    - {tool}: {count}")
    
    # Tingkatkan DGM berdasarkan introspeksi
    print("\nImproving DGM based on introspection...")
    dgm.improve()
    
    # Buat tugas baru (sedang)
    medium_task = create_math_task(difficulty="medium")
    
    # Evaluasi agen terbaik pada tugas sedang
    best_agent = dgm.get_best_agent()
    medium_score = evaluate_agent(best_agent, medium_task)
    print(f"\nBest agent score on medium task: {medium_score:.4f}")
    
    # Jalankan evolusi pada tugas sedang
    print("\nEvolving on medium task...")
    
    # Perbarui lingkungan dengan sumber daya terbatas
    environment["resources"]["cpu"] = 0.5
    environment["resources"]["memory"] = 0.5
    
    # Adaptasi ke lingkungan baru
    print("Adapting to new environment...")
    dgm.adaptation_engine.adapt(dgm, environment, medium_task)
    
    # Jalankan evolusi
    dgm.evolve(generations=5, task=medium_task)
    
    # Lakukan introspeksi lagi
    print("\nPerforming introspection again...")
    analysis = dgm.introspect()
    print("Introspection results:")
    for key, value in analysis.items():
        if key != "tool_diversity":
            print(f"  - {key}: {value}")
        else:
            print(f"  - {key}:")
            for tool, count in value.items():
                print(f"    - {tool}: {count}")
    
    # Tingkatkan DGM berdasarkan introspeksi
    print("\nImproving DGM based on introspection...")
    dgm.improve()
    
    # Buat tugas baru (sulit)
    hard_task = create_math_task(difficulty="hard")
    
    # Evaluasi agen terbaik pada tugas sulit
    best_agent = dgm.get_best_agent()
    hard_score = evaluate_agent(best_agent, hard_task)
    print(f"\nBest agent score on hard task: {hard_score:.4f}")
    
    # Jalankan evolusi pada tugas sulit
    print("\nEvolving on hard task...")
    
    # Perbarui lingkungan dengan sumber daya sangat terbatas
    environment["resources"]["cpu"] = 0.2
    environment["resources"]["memory"] = 0.2
    environment["time"]["execution_time"] = 30.0
    
    # Adaptasi ke lingkungan baru
    print("Adapting to new environment...")
    dgm.adaptation_engine.adapt(dgm, environment, hard_task)
    
    # Jalankan evolusi
    dgm.evolve(generations=5, task=hard_task)
    
    # Dapatkan agen terbaik
    best_agent = dgm.get_best_agent()
    
    # Evaluasi agen terbaik pada semua tugas
    final_easy_score = evaluate_agent(best_agent, easy_task)
    final_medium_score = evaluate_agent(best_agent, medium_task)
    final_hard_score = evaluate_agent(best_agent, hard_task)
    
    print("\nFinal best agent scores:")
    print(f"  - Easy task: {final_easy_score:.4f}")
    print(f"  - Medium task: {final_medium_score:.4f}")
    print(f"  - Hard task: {final_hard_score:.4f}")
    
    # Cetak alat agen terbaik
    print("\nBest agent tools:")
    for tool in best_agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen terbaik
    print("\nBest agent parameters:")
    print(f"  - Memory capacity: {best_agent.memory_capacity}")
    print(f"  - Learning rate: {best_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {best_agent.exploration_rate:.4f}")

if __name__ == "__main__":
    main()