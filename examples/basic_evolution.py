"""
Contoh evolusi dasar menggunakan Darwin-Gödel Machine.
"""

import os
import sys
import random
import time
from typing import Dict, Any

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.utils.evaluation import evaluate_agent, create_evaluation_task


def create_basic_agent() -> BaseAgent:
    """
    Buat agen dasar dengan beberapa alat.
    
    Returns:
        Agen dasar
    """
    agent = BaseAgent(
        memory_capacity=5,
        learning_rate=0.01,
        exploration_rate=0.1
    )
    
    # Tambahkan beberapa alat dasar
    agent.add_tool(Tool(
        name="add",
        function=lambda x, y: x + y,
        description="Tambahkan dua angka"
    ))
    
    agent.add_tool(Tool(
        name="multiply",
        function=lambda x, y: x * y,
        description="Kalikan dua angka"
    ))
    
    agent.add_tool(Tool(
        name="power",
        function=lambda x, y: x ** y,
        description="Pangkatkan angka pertama dengan angka kedua"
    ))
    
    return agent


def main():
    """
    Fungsi utama.
    """
    print("=== Darwin-Gödel Machine: Basic Evolution ===")
    
    # Buat agen dasar
    initial_agent = create_basic_agent()
    
    # Inisialisasi DGM
    dgm = DGM(initial_agent=initial_agent)
    
    # Buat tugas evaluasi
    task = {
        "type": "math",
        "operations": [
            {"op": "add", "args": [5, 3]},
            {"op": "multiply", "args": [4, 7]},
            {"op": "power", "args": [2, 3]}
        ]
    }
    
    # Fungsi evaluasi kustom
    def custom_evaluation(agent: BaseAgent, task: Dict[str, Any]) -> float:
        """
        Evaluasi agen pada tugas matematika.
        
        Args:
            agent: Agen yang akan dievaluasi
            task: Tugas matematika
            
        Returns:
            Skor evaluasi (0.0 - 1.0)
        """
        if task["type"] != "math":
            return 0.0
        
        operations = task["operations"]
        correct = 0
        
        for op in operations:
            op_name = op["op"]
            args = op["args"]
            
            # Dapatkan alat yang sesuai
            tool = agent.get_tool(op_name)
            
            if tool is None:
                continue
            
            # Hitung hasil yang diharapkan
            if op_name == "add":
                expected = args[0] + args[1]
            elif op_name == "multiply":
                expected = args[0] * args[1]
            elif op_name == "power":
                expected = args[0] ** args[1]
            else:
                continue
            
            # Hitung hasil aktual
            try:
                actual = tool(*args)
                
                # Periksa apakah hasil benar
                if abs(actual - expected) < 1e-6:
                    correct += 1
            except Exception:
                pass
        
        # Hitung skor
        return correct / len(operations) if operations else 0.0
    
    # Jalankan evolusi
    num_generations = 10
    print(f"Evolving for {num_generations} generations...")
    
    start_time = time.time()
    dgm.evolve(generations=num_generations, task=task, evaluation_fn=custom_evaluation)
    end_time = time.time()
    
    print(f"Evolution completed in {end_time - start_time:.2f} seconds.")
    
    # Dapatkan agen terbaik
    best_agent = dgm.get_best_agent()
    best_score = custom_evaluation(best_agent, task)
    
    print(f"Best agent score: {best_score:.4f}")
    print(f"Best agent tools: {[tool.name for tool in best_agent.tools]}")
    print(f"Best agent parameters:")
    print(f"  - Memory capacity: {best_agent.memory_capacity}")
    print(f"  - Learning rate: {best_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {best_agent.exploration_rate:.4f}")
    
    # Visualisasi pohon evolusi jika matplotlib tersedia
    try:
        from simple_dgm.utils.visualization import visualize_evolution_tree
        
        print("Generating evolution tree visualization...")
        visualize_evolution_tree(dgm, output_path="evolution_tree.png", show_plot=False)
        print("Visualization saved to 'evolution_tree.png'.")
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")


if __name__ == "__main__":
    main()