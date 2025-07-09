"""
Contoh evolusi agen pengkodean menggunakan Darwin-Gödel Machine.
"""

import os
import sys
import random
import time
from typing import Dict, Any

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simple_dgm.dgm import DGM
from simple_dgm.agents.coding_agent import CodingAgent
from simple_dgm.utils.evaluation import evaluate_agent, create_evaluation_task


def create_initial_coding_agent() -> CodingAgent:
    """
    Buat agen pengkodean awal.
    
    Returns:
        Agen pengkodean
    """
    return CodingAgent(
        memory_capacity=10,
        learning_rate=0.01,
        exploration_rate=0.1,
        code_style="clean",
        preferred_language="python"
    )


def main():
    """
    Fungsi utama.
    """
    print("=== Darwin-Gödel Machine: Coding Agent Evolution ===")
    
    # Buat agen pengkodean awal
    initial_agent = create_initial_coding_agent()
    
    # Inisialisasi DGM
    dgm = DGM(initial_agent=initial_agent)
    
    # Buat tugas evaluasi
    tasks = [
        create_evaluation_task("coding", difficulty=0.3),
        create_evaluation_task("coding", difficulty=0.5),
        create_evaluation_task("coding", difficulty=0.7)
    ]
    
    # Fungsi evaluasi kustom
    def custom_evaluation(agent: CodingAgent, tasks: list) -> float:
        """
        Evaluasi agen pada beberapa tugas pengkodean.
        
        Args:
            agent: Agen yang akan dievaluasi
            tasks: Daftar tugas pengkodean
            
        Returns:
            Skor evaluasi (0.0 - 1.0)
        """
        if not isinstance(agent, CodingAgent):
            return 0.1
        
        scores = []
        
        for task in tasks:
            try:
                score = evaluate_agent(agent, task)
                scores.append(score)
            except Exception as e:
                print(f"Error evaluating task: {e}")
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    # Jalankan evolusi
    num_generations = 20
    print(f"Evolving for {num_generations} generations...")
    
    start_time = time.time()
    dgm.evolve(generations=num_generations, task=tasks, evaluation_fn=custom_evaluation)
    end_time = time.time()
    
    print(f"Evolution completed in {end_time - start_time:.2f} seconds.")
    
    # Dapatkan agen terbaik
    best_agent = dgm.get_best_agent()
    best_score = custom_evaluation(best_agent, tasks)
    
    print(f"Best agent score: {best_score:.4f}")
    print(f"Best agent parameters:")
    print(f"  - Code style: {best_agent.code_style}")
    print(f"  - Preferred language: {best_agent.preferred_language}")
    print(f"  - Memory capacity: {best_agent.memory_capacity}")
    print(f"  - Learning rate: {best_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {best_agent.exploration_rate:.4f}")
    
    # Uji agen terbaik pada tugas baru
    print("\nTesting best agent on a new task...")
    
    test_task = create_evaluation_task("coding", difficulty=0.6)
    
    if test_task["subtask"] == "analyze":
        print(f"Task: Analyze code with {len(test_task['code'].split('\\n'))} lines")
        result = best_agent._analyze_code(test_task["code"])
        print(f"Analysis result:")
        print(f"  - Complexity: {result['complexity']}")
        print(f"  - Bugs found: {len(result['bugs'])}")
    
    elif test_task["subtask"] == "fix_bugs":
        print(f"Task: Fix bugs in code with {len(test_task['code'].split('\\n'))} lines")
        bugs = best_agent._analyze_code(test_task["code"])["bugs"]
        fixed_code = best_agent._fix_bugs(test_task["code"], bugs)
        print(f"Fixed code:")
        print(fixed_code[:200] + "..." if len(fixed_code) > 200 else fixed_code)
    
    elif test_task["subtask"] == "optimize":
        print(f"Task: Optimize code with {len(test_task['code'].split('\\n'))} lines")
        optimized_code = best_agent._optimize_code(test_task["code"])
        print(f"Optimized code:")
        print(optimized_code[:200] + "..." if len(optimized_code) > 200 else optimized_code)
    
    elif test_task["subtask"] == "generate":
        print(f"Task: Generate code for {test_task['spec']['classes'][0]['name']} class")
        generated_code = best_agent._generate_code(test_task["spec"])
        print(f"Generated code:")
        print(generated_code[:200] + "..." if len(generated_code) > 200 else generated_code)
    
    # Visualisasi pohon evolusi jika matplotlib tersedia
    try:
        from simple_dgm.utils.visualization import visualize_evolution_tree, visualize_performance_history
        
        print("\nGenerating visualizations...")
        visualize_evolution_tree(dgm, output_path="coding_evolution_tree.png", show_plot=False)
        visualize_performance_history(dgm, output_path="coding_performance.png", show_plot=False)
        print("Visualizations saved to 'coding_evolution_tree.png' and 'coding_performance.png'.")
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")


if __name__ == "__main__":
    main()