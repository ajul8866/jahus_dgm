"""
Contoh penggunaan agen meta untuk Darwin-Gödel Machine.
"""

import os
import sys
import random
import time
from typing import Dict, Any

# Tambahkan path root ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simple_dgm.agents.base_agent import BaseAgent
from simple_dgm.agents.coding_agent import CodingAgent
from simple_dgm.agents.problem_solving_agent import ProblemSolvingAgent
from simple_dgm.agents.meta_agent import MetaAgent
from simple_dgm.utils.evaluation import evaluate_agent, create_evaluation_task
from simple_dgm.utils.serialization import save_agent, load_agent, export_agent_code


def main():
    """
    Fungsi utama.
    """
    print("=== Darwin-Gödel Machine: Meta Agent Example ===")
    
    # Buat agen meta
    meta_agent = MetaAgent(
        memory_capacity=20,
        learning_rate=0.01,
        exploration_rate=0.1,
        agent_types=["BaseAgent", "CodingAgent", "ProblemSolvingAgent"],
        code_generation_capacity=0.7
    )
    
    print("Meta agent created.")
    print(f"Agent types: {meta_agent.agent_types}")
    print(f"Code generation capacity: {meta_agent.code_generation_capacity}")
    
    # Buat agen baru menggunakan agen meta
    print("\nCreating a new coding agent...")
    
    coding_agent_spec = {
        "type": "create_agent",
        "agent_type": "CodingAgent",
        "params": {
            "memory_capacity": 15,
            "learning_rate": 0.02,
            "exploration_rate": 0.15,
            "code_style": "verbose",
            "preferred_language": "python"
        }
    }
    
    coding_agent = meta_agent.solve(coding_agent_spec)
    
    print("Coding agent created.")
    print(f"Code style: {coding_agent.code_style}")
    print(f"Preferred language: {coding_agent.preferred_language}")
    
    # Modifikasi agen
    print("\nModifying the coding agent...")
    
    modifications = {
        "type": "modify_agent",
        "agent": coding_agent,
        "modifications": {
            "params": {
                "code_style": "compact",
                "learning_rate": 0.03
            },
            "tools": [
                {
                    "action": "add",
                    "tool": {
                        "name": "custom_tool",
                        "function": lambda x: x * 2,
                        "description": "Custom tool that doubles the input"
                    }
                }
            ]
        }
    }
    
    modified_agent = meta_agent.solve(modifications)
    
    print("Coding agent modified.")
    print(f"New code style: {modified_agent.code_style}")
    print(f"New learning rate: {modified_agent.learning_rate}")
    print(f"Tools: {[tool.name for tool in modified_agent.tools]}")
    
    # Buat agen pemecahan masalah
    print("\nCreating a problem solving agent...")
    
    problem_solving_spec = {
        "type": "create_agent",
        "agent_type": "ProblemSolvingAgent",
        "params": {
            "memory_capacity": 15,
            "learning_rate": 0.02,
            "exploration_rate": 0.15,
            "problem_types": ["optimization", "search"],
            "max_iterations": 200,
            "timeout": 20.0
        }
    }
    
    problem_solving_agent = meta_agent.solve(problem_solving_spec)
    
    print("Problem solving agent created.")
    print(f"Problem types: {problem_solving_agent.problem_types}")
    print(f"Max iterations: {problem_solving_agent.max_iterations}")
    
    # Gabungkan agen
    print("\nMerging coding and problem solving agents...")
    
    merge_spec = {
        "type": "merge_agents",
        "agent1": coding_agent,
        "agent2": problem_solving_agent
    }
    
    merged_agent = meta_agent.solve(merge_spec)
    
    print("Agents merged.")
    print(f"Merged agent type: {merged_agent.__class__.__name__}")
    print(f"Tools: {[tool.name for tool in merged_agent.tools]}")
    
    # Analisis agen
    print("\nAnalyzing the merged agent...")
    
    analysis_spec = {
        "type": "analyze_agent",
        "agent": merged_agent
    }
    
    analysis = meta_agent.solve(analysis_spec)
    
    print("Analysis completed.")
    print(f"Agent type: {analysis['type']}")
    print(f"Number of tools: {analysis['num_tools']}")
    print(f"Capabilities: {analysis.get('capabilities', [])}")
    
    # Hasilkan kode agen baru
    print("\nGenerating code for a new agent...")
    
    code_spec = {
        "type": "generate_agent_code",
        "spec": {
            "name": "CustomOptimizationAgent",
            "base_class": "ProblemSolvingAgent",
            "description": "Custom agent specialized for optimization problems",
            "params": [
                {
                    "name": "optimization_method",
                    "type": "str",
                    "default": "hill_climbing",
                    "description": "Method used for optimization"
                },
                {
                    "name": "convergence_threshold",
                    "type": "float",
                    "default": 0.001,
                    "description": "Threshold for convergence"
                }
            ],
            "methods": [
                {
                    "name": "optimize",
                    "params": [
                        {
                            "name": "problem",
                            "type": "Dict[str, Any]",
                            "description": "Optimization problem"
                        }
                    ],
                    "return": "Dict[str, Any]",
                    "return_desc": "Optimized solution",
                    "description": "Optimize the given problem",
                    "body": [
                        "# Implement optimization logic here",
                        "if self.optimization_method == 'hill_climbing':",
                        "    # Hill climbing implementation",
                        "    return {'solution': 'hill_climbing_result'}",
                        "elif self.optimization_method == 'simulated_annealing':",
                        "    # Simulated annealing implementation",
                        "    return {'solution': 'simulated_annealing_result'}",
                        "else:",
                        "    # Default implementation",
                        "    return {'solution': 'default_result'}"
                    ]
                }
            ]
        }
    }
    
    agent_code = meta_agent.solve(code_spec)
    
    print("Agent code generated.")
    print("Code preview:")
    print("\n".join(agent_code.split("\n")[:20]) + "\n...")
    
    # Simpan kode agen ke file
    with open("custom_optimization_agent.py", "w") as f:
        f.write(agent_code)
    
    print("Agent code saved to 'custom_optimization_agent.py'.")
    
    # Simpan agen
    print("\nSaving agents...")
    
    save_agent(coding_agent, "coding_agent.json")
    save_agent(problem_solving_agent, "problem_solving_agent.json")
    save_agent(merged_agent, "merged_agent.json")
    
    print("Agents saved to JSON files.")
    
    # Ekspor kode agen
    print("\nExporting agent code...")
    
    export_agent_code(merged_agent, "exported_merged_agent.py")
    
    print("Agent code exported to 'exported_merged_agent.py'.")


if __name__ == "__main__":
    main()