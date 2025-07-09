"""
Contoh agen kolaboratif untuk Darwin-Gödel Machine.

Contoh ini mendemonstrasikan penggunaan mesin kolaborasi untuk:
- Membangun jaringan agen
- Berbagi pengetahuan antar agen
- Dekomposisi tugas
- Pembangunan konsensus
"""

import random
import time
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.agents.coding_agent import CodingAgent
from simple_dgm.agents.problem_solving_agent import ProblemSolvingAgent
from simple_dgm.core.collaboration import (
    CollaborationEngine, AgentNetwork, KnowledgeSharing,
    TaskDecomposition, ConsensusBuilder
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

def power(x, y):
    try:
        return x ** y
    except:
        return 0

def sqrt(x):
    try:
        return x ** 0.5 if x >= 0 else 0
    except:
        return 0

def analyze_code(code):
    """
    Analisis kode sederhana.
    
    Args:
        code: Kode yang akan dianalisis
        
    Returns:
        Hasil analisis
    """
    lines = code.strip().split("\n")
    num_lines = len(lines)
    
    # Hitung komentar
    num_comments = sum(1 for line in lines if line.strip().startswith("#"))
    
    # Hitung fungsi
    num_functions = sum(1 for line in lines if line.strip().startswith("def "))
    
    # Hitung kelas
    num_classes = sum(1 for line in lines if line.strip().startswith("class "))
    
    return {
        "num_lines": num_lines,
        "num_comments": num_comments,
        "num_functions": num_functions,
        "num_classes": num_classes,
        "comment_ratio": num_comments / num_lines if num_lines > 0 else 0
    }

def fix_bugs(code):
    """
    Perbaiki bug sederhana dalam kode.
    
    Args:
        code: Kode yang akan diperbaiki
        
    Returns:
            Kode yang diperbaiki
    """
    # Perbaiki beberapa bug umum
    fixed_code = code.replace("print x", "print(x)")
    fixed_code = fixed_code.replace("except Exception, e:", "except Exception as e:")
    fixed_code = fixed_code.replace("xrange", "range")
    
    return fixed_code

def optimize_code(code):
    """
    Optimasi kode sederhana.
    
    Args:
        code: Kode yang akan dioptimasi
        
    Returns:
        Kode yang dioptimasi
    """
    # Optimasi sederhana
    optimized_code = code.replace("for i in range(len(arr)):", "for i, item in enumerate(arr):")
    optimized_code = optimized_code.replace("for i in range(0, len(arr)):", "for i, item in enumerate(arr):")
    
    return optimized_code

def search(query, data):
    """
    Pencarian sederhana.
    
    Args:
        query: Kueri pencarian
        data: Data yang akan dicari
        
    Returns:
        Hasil pencarian
    """
    if isinstance(data, list):
        return [item for item in data if query in str(item)]
    elif isinstance(data, dict):
        return {k: v for k, v in data.items() if query in str(k) or query in str(v)}
    else:
        return None

def plan(goal, steps):
    """
    Perencanaan sederhana.
    
    Args:
        goal: Tujuan
        steps: Langkah-langkah yang tersedia
        
    Returns:
        Rencana
    """
    # Pilih langkah-langkah yang relevan
    relevant_steps = [step for step in steps if any(keyword in step for keyword in goal.split())]
    
    # Urutkan langkah-langkah
    plan = sorted(relevant_steps, key=lambda step: len(set(step.split()) & set(goal.split())), reverse=True)
    
    return plan


def main():
    print("=== Darwin-Gödel Machine: Collaborative Agents Example ===")
    
    # Buat mesin kolaborasi
    collaboration_engine = CollaborationEngine()
    
    # Buat agen-agen
    print("\nCreating agents...")
    agents = create_agents()
    
    # Tambahkan agen-agen ke mesin kolaborasi
    for agent_id, agent in agents.items():
        collaboration_engine.add_agent(agent, agent_id, agent.__class__.__name__)
    
    # Hubungkan agen-agen
    print("\nConnecting agents...")
    connect_agents(collaboration_engine)
    
    # Visualisasi jaringan agen
    print("\nVisualizing agent network...")
    visualize_agent_network(collaboration_engine.agent_network)
    
    # Berbagi pengetahuan antar agen
    print("\nSharing knowledge between agents...")
    share_knowledge(collaboration_engine)
    
    # Dekomposisi tugas
    print("\nDecomposing task...")
    task = create_complex_task()
    print(f"Complex task: {task['description']}")
    
    agent_ids = list(agents.keys())
    subtasks = collaboration_engine.decompose_task(task, agent_ids)
    
    print("Subtasks:")
    for agent_id, subtask in subtasks.items():
        print(f"  Agent {agent_id}: {subtask.get('description', '')}")
    
    # Kolaborasi untuk menyelesaikan tugas
    print("\nCollaborating to solve the task...")
    solution = collaboration_engine.collaborate(task)
    
    print("Collaborative solution:")
    if isinstance(solution, dict):
        for key, value in solution.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {solution}")
    
    # Dapatkan statistik berbagi pengetahuan
    print("\nKnowledge sharing statistics:")
    stats = collaboration_engine.knowledge_sharing.get_sharing_statistics()
    
    print(f"  Total shares: {stats['total_shares']}")
    print("  By type:")
    for type_name, count in stats.get("by_type", {}).items():
        print(f"    {type_name}: {count}")


def create_agents() -> Dict[str, BaseAgent]:
    """
    Buat agen-agen untuk kolaborasi.
    
    Returns:
        Dictionary agen
    """
    agents = {}
    
    # Buat agen matematika
    math_agent = BaseAgent(memory_capacity=10, learning_rate=0.01, exploration_rate=0.1)
    math_agent.add_tool(Tool(name="add", function=add, description="Tambahkan dua angka"))
    math_agent.add_tool(Tool(name="subtract", function=subtract, description="Kurangkan dua angka"))
    math_agent.add_tool(Tool(name="multiply", function=multiply, description="Kalikan dua angka"))
    math_agent.add_tool(Tool(name="divide", function=divide, description="Bagi dua angka"))
    math_agent.add_tool(Tool(name="power", function=power, description="Pangkatkan angka pertama dengan angka kedua"))
    math_agent.add_tool(Tool(name="sqrt", function=sqrt, description="Akar kuadrat dari angka"))
    
    agents["math_agent"] = math_agent
    print("  Created math agent with 6 tools")
    
    # Buat agen pengkodean
    coding_agent = CodingAgent(
        code_style="clean",
        preferred_language="python",
        memory_capacity=10,
        learning_rate=0.01,
        exploration_rate=0.1
    )
    coding_agent.add_tool(Tool(name="analyze_code", function=analyze_code, description="Analisis kode"))
    coding_agent.add_tool(Tool(name="fix_bugs", function=fix_bugs, description="Perbaiki bug dalam kode"))
    coding_agent.add_tool(Tool(name="optimize_code", function=optimize_code, description="Optimasi kode"))
    
    agents["coding_agent"] = coding_agent
    print("  Created coding agent with 3 tools")
    
    # Buat agen pemecahan masalah
    problem_solving_agent = ProblemSolvingAgent(
        problem_types=["search", "planning"],
        memory_capacity=10,
        learning_rate=0.01,
        exploration_rate=0.1
    )
    problem_solving_agent.add_tool(Tool(name="search", function=search, description="Cari dalam data"))
    problem_solving_agent.add_tool(Tool(name="plan", function=plan, description="Buat rencana"))
    
    agents["problem_solving_agent"] = problem_solving_agent
    print("  Created problem solving agent with 2 tools")
    
    return agents


def connect_agents(collaboration_engine: CollaborationEngine) -> None:
    """
    Hubungkan agen-agen dalam jaringan kolaborasi.
    
    Args:
        collaboration_engine: Mesin kolaborasi
    """
    # Hubungkan agen matematika dengan agen pengkodean
    collaboration_engine.connect_agents("math_agent", "coding_agent", weight=0.7)
    print("  Connected math agent to coding agent (weight: 0.7)")
    
    # Hubungkan agen matematika dengan agen pemecahan masalah
    collaboration_engine.connect_agents("math_agent", "problem_solving_agent", weight=0.5)
    print("  Connected math agent to problem solving agent (weight: 0.5)")
    
    # Hubungkan agen pengkodean dengan agen pemecahan masalah
    collaboration_engine.connect_agents("coding_agent", "problem_solving_agent", weight=0.8)
    print("  Connected coding agent to problem solving agent (weight: 0.8)")


def visualize_agent_network(agent_network: AgentNetwork) -> None:
    """
    Visualisasi jaringan agen.
    
    Args:
        agent_network: Jaringan agen
    """
    # Buat grafik dari jaringan agen
    G = agent_network.graph
    
    # Tetapkan warna berdasarkan tipe agen
    node_colors = []
    for node in G.nodes():
        agent_type = agent_network.get_agent_type(node)
        if agent_type == "BaseAgent":
            node_colors.append("skyblue")
        elif agent_type == "CodingAgent":
            node_colors.append("lightgreen")
        elif agent_type == "ProblemSolvingAgent":
            node_colors.append("salmon")
        else:
            node_colors.append("gray")
    
    # Tetapkan label
    labels = {node: node for node in G.nodes()}
    
    # Tetapkan ketebalan edge berdasarkan bobot
    edge_widths = [G[u][v]["weight"] * 3 for u, v in G.edges()]
    
    # Buat plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    
    # Tambahkan label edge
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("Agent Collaboration Network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("agent_network.png")
    print("  Agent network visualization saved to 'agent_network.png'")


def share_knowledge(collaboration_engine: CollaborationEngine) -> None:
    """
    Berbagi pengetahuan antar agen.
    
    Args:
        collaboration_engine: Mesin kolaborasi
    """
    # Berbagi alat dari agen matematika ke agen pengkodean
    collaboration_engine.share_knowledge(
        source_id="math_agent",
        target_id="coding_agent",
        knowledge_type="tool",
        knowledge="power"
    )
    print("  Shared 'power' tool from math agent to coding agent")
    
    # Berbagi alat dari agen pengkodean ke agen pemecahan masalah
    collaboration_engine.share_knowledge(
        source_id="coding_agent",
        target_id="problem_solving_agent",
        knowledge_type="tool",
        knowledge="analyze_code"
    )
    print("  Shared 'analyze_code' tool from coding agent to problem solving agent")
    
    # Berbagi parameter dari agen pemecahan masalah ke agen matematika
    collaboration_engine.share_knowledge(
        source_id="problem_solving_agent",
        target_id="math_agent",
        knowledge_type="parameter",
        knowledge={"exploration_rate": 0.2}
    )
    print("  Shared 'exploration_rate' parameter from problem solving agent to math agent")


def create_complex_task() -> Dict[str, Any]:
    """
    Buat tugas kompleks yang memerlukan kolaborasi.
    
    Returns:
        Tugas kompleks
    """
    return {
        "type": "complex",
        "description": "Analyze and optimize a mathematical algorithm",
        "parts": [
            {
                "type": "math",
                "description": "Calculate the complexity of the algorithm",
                "data": {
                    "operations": [
                        {"type": "addition", "count": 10},
                        {"type": "multiplication", "count": 5},
                        {"type": "power", "count": 2}
                    ]
                }
            },
            {
                "type": "code",
                "description": "Analyze and optimize the code",
                "data": {
                    "code": """
def calculate(x, n):
    result = 0
    for i in range(n):
        result = result + x
    for i in range(n // 2):
        result = result * 2
    result = result ** 2
    return result
"""
                }
            },
            {
                "type": "planning",
                "description": "Create a plan to further improve the algorithm",
                "data": {
                    "goal": "Improve algorithm efficiency",
                    "available_steps": [
                        "Analyze time complexity",
                        "Identify bottlenecks",
                        "Optimize loops",
                        "Use mathematical simplifications",
                        "Implement parallel processing",
                        "Reduce memory usage"
                    ]
                }
            }
        ]
    }


if __name__ == "__main__":
    main()