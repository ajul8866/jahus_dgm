"""
Contoh DGM yang meningkatkan diri sendiri.

Contoh ini mendemonstrasikan penggunaan mesin introspeksi dan adaptasi untuk:
- Menganalisis kode DGM
- Meningkatkan kinerja DGM
- Beradaptasi dengan lingkungan dan tugas yang berubah
"""

import random
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.core.introspection import (
    IntrospectionEngine, CodeAnalyzer, PerformanceProfiler,
    BehaviorTracker, SelfImprovementEngine
)
from simple_dgm.core.adaptation import (
    AdaptationEngine, EnvironmentModel, TaskModel,
    ResourceManager, LearningRateScheduler
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


def main():
    print("=== Darwin-Gödel Machine: Self-Improving Example ===")
    
    # Buat agen dasar
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    agent.add_tool(Tool(name="add", function=add, description="Tambahkan dua angka"))
    agent.add_tool(Tool(name="subtract", function=subtract, description="Kurangkan dua angka"))
    
    # Inisialisasi DGM
    dgm = DGM(initial_agent=agent, population_size=20)
    
    # Buat mesin introspeksi
    introspection_engine = create_introspection_engine()
    dgm.set_introspection_engine(introspection_engine)
    
    # Buat mesin adaptasi
    adaptation_engine = create_adaptation_engine()
    dgm.set_adaptation_engine(adaptation_engine)
    
    # Jalankan evolusi dengan introspeksi dan adaptasi
    print("\nRunning evolution with introspection and adaptation...")
    run_evolution_with_introspection(dgm)
    
    # Jalankan evolusi dengan tugas yang berubah
    print("\nRunning evolution with changing tasks...")
    run_evolution_with_changing_tasks(dgm)
    
    # Visualisasi hasil
    print("\nVisualizing results...")
    visualize_results(dgm)


def create_introspection_engine() -> IntrospectionEngine:
    """
    Buat mesin introspeksi.
    
    Returns:
        Mesin introspeksi
    """
    # Buat mesin introspeksi
    introspection_engine = IntrospectionEngine()
    
    # Tambahkan penganalisis kode
    code_analyzer = CodeAnalyzer()
    introspection_engine.add_analyzer(lambda target: {
        "code_analysis": code_analyzer.analyze_class(target.__class__)
    })
    
    # Tambahkan profiler kinerja
    performance_profiler = PerformanceProfiler()
    introspection_engine.add_analyzer(lambda target: {
        "performance": {
            "execution_time": target.evolution_stats.get("execution_time", []),
            "best_fitness": target.evolution_stats.get("best_fitness", []),
            "avg_fitness": target.evolution_stats.get("avg_fitness", [])
        }
    })
    
    # Tambahkan pelacak perilaku
    behavior_tracker = BehaviorTracker()
    introspection_engine.add_analyzer(lambda target: {
        "behavior": {
            "population_size": len(target.archive),
            "generation": target.current_generation,
            "diversity": target.evolution_stats.get("diversity", [])
        }
    })
    
    # Tambahkan strategi peningkatan
    improvement_engine = SelfImprovementEngine()
    
    # Strategi peningkatan parameter
    def improve_parameters(target, analysis):
        # Jika kinerja menurun, tingkatkan eksplorasi
        performance = analysis.get("performance", {})
        best_fitness = performance.get("best_fitness", [])
        
        if len(best_fitness) >= 3 and best_fitness[-1] <= best_fitness[-2] <= best_fitness[-3]:
            # Kinerja stagnan atau menurun, tingkatkan eksplorasi
            for agent_id, info in target.archive.items():
                agent = info["agent"]
                if hasattr(agent, "exploration_rate"):
                    agent.exploration_rate = min(0.5, agent.exploration_rate * 1.5)
            
            print("  Increased exploration rate due to stagnant performance")
        
        return target
    
    # Strategi peningkatan alat
    def improve_tools(target, analysis):
        # Tambahkan alat yang hilang ke agen terbaik
        best_agent_id = target.get_best_agent_id()
        if best_agent_id:
            best_agent = target.archive[best_agent_id]["agent"]
            
            # Periksa alat yang dimiliki
            tool_names = [tool.name for tool in best_agent.tools]
            
            # Tambahkan alat yang hilang
            if "multiply" not in tool_names:
                best_agent.add_tool(Tool(name="multiply", function=multiply, description="Kalikan dua angka"))
                print("  Added 'multiply' tool to best agent")
            
            if "divide" not in tool_names:
                best_agent.add_tool(Tool(name="divide", function=divide, description="Bagi dua angka"))
                print("  Added 'divide' tool to best agent")
        
        return target
    
    # Tambahkan strategi peningkatan
    introspection_engine.add_improvement_strategy(improve_parameters)
    introspection_engine.add_improvement_strategy(improve_tools)
    
    print("Created introspection engine with code analyzer, performance profiler, and behavior tracker")
    return introspection_engine


def create_adaptation_engine() -> AdaptationEngine:
    """
    Buat mesin adaptasi.
    
    Returns:
        Mesin adaptasi
    """
    # Buat mesin adaptasi
    adaptation_engine = AdaptationEngine()
    
    # Tambahkan model lingkungan
    environment_model = EnvironmentModel()
    adaptation_engine.add_environment_model("default", environment_model)
    
    # Tambahkan model tugas
    task_model = TaskModel()
    adaptation_engine.add_task_model("default", task_model)
    
    # Tambahkan pengelola sumber daya
    resource_manager = ResourceManager()
    adaptation_engine.resource_manager = resource_manager
    
    # Tambahkan penjadwal tingkat pembelajaran
    learning_rate_scheduler = LearningRateScheduler(
        initial_rate=0.01,
        min_rate=0.001,
        max_rate=0.1,
        decay_factor=0.9
    )
    adaptation_engine.learning_rate_scheduler = learning_rate_scheduler
    
    print("Created adaptation engine with environment model, task model, resource manager, and learning rate scheduler")
    return adaptation_engine


def run_evolution_with_introspection(dgm: DGM) -> None:
    """
    Jalankan evolusi dengan introspeksi.
    
    Args:
        dgm: DGM yang akan dijalankan
    """
    # Buat tugas matematika
    task = create_math_task()
    
    # Jalankan evolusi dengan introspeksi
    for i in range(5):
        print(f"\nEvolution cycle {i+1}:")
        
        # Jalankan evolusi
        dgm.evolve(generations=5, task=task)
        
        # Introspeksi DGM
        print("Performing introspection...")
        analysis = dgm.introspect()
        
        # Cetak beberapa hasil analisis
        if "code_analysis" in analysis:
            code_analysis = analysis["code_analysis"]
            complexity = code_analysis.get("complexity", {})
            print(f"  Code complexity: {complexity.get('cyclomatic_complexity', 'N/A')}")
        
        if "performance" in analysis:
            performance = analysis["performance"]
            best_fitness = performance.get("best_fitness", [])
            if best_fitness:
                print(f"  Best fitness: {best_fitness[-1]:.4f}")
        
        if "behavior" in analysis:
            behavior = analysis["behavior"]
            print(f"  Population size: {behavior.get('population_size', 'N/A')}")
            print(f"  Generation: {behavior.get('generation', 'N/A')}")
        
        # Tingkatkan DGM
        print("Improving DGM...")
        dgm.improve()


def run_evolution_with_changing_tasks(dgm: DGM) -> None:
    """
    Jalankan evolusi dengan tugas yang berubah.
    
    Args:
        dgm: DGM yang akan dijalankan
    """
    # Jalankan evolusi dengan tugas yang berubah
    for i in range(3):
        print(f"\nTask cycle {i+1}:")
        
        # Buat tugas dengan kompleksitas yang meningkat
        complexity = ["easy", "medium", "hard"][i]
        task = create_math_task(complexity)
        
        print(f"Task complexity: {complexity}")
        
        # Jalankan evolusi
        dgm.evolve(generations=5, task=task)
        
        # Evaluasi agen terbaik
        best_agent = dgm.get_best_agent()
        if best_agent:
            score = evaluate_agent(best_agent, task)
            print(f"Best agent score: {score:.4f}")


def create_math_task(complexity: str = "medium") -> Dict[str, Any]:
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


def evaluate_agent(agent: BaseAgent, task: Dict[str, Any]) -> float:
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
            # Coba evaluasi ekspresi menggunakan agen
            result = agent.solve(expression)
            
            # Periksa apakah hasil mendekati yang diharapkan
            if isinstance(result, (int, float)) and abs(result - expected) < 0.001:
                correct += 1
        except:
            pass
    
    return correct / total if total > 0 else 0.0


def visualize_results(dgm: DGM) -> None:
    """
    Visualisasi hasil evolusi.
    
    Args:
        dgm: DGM yang akan divisualisasikan
    """
    # Dapatkan statistik
    stats = dgm.get_statistics()
    
    # Buat plot
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
    plt.savefig("self_improving_results.png")
    print("Results visualization saved to 'self_improving_results.png'")


if __name__ == "__main__":
    main()