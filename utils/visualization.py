"""
Utilitas visualisasi untuk Darwin-Gödel Machine.
"""

import os
import json
import random
from typing import Dict, Any, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def visualize_evolution_tree(dgm, output_path: Optional[str] = None, 
                            show_plot: bool = True, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualisasikan pohon evolusi DGM.
    
    Args:
        dgm: Objek DGM
        output_path: Path untuk menyimpan visualisasi (opsional)
        show_plot: Apakah akan menampilkan plot
        figsize: Ukuran gambar
    """
    if not VISUALIZATION_AVAILABLE:
        print("Matplotlib dan NetworkX diperlukan untuk visualisasi. Instal dengan 'pip install matplotlib networkx'.")
        return
    
    # Dapatkan pohon evolusi
    tree = dgm.get_evolution_tree()
    
    # Buat graf
    G = nx.DiGraph()
    
    # Tambahkan node dan edge
    def add_nodes_and_edges(node, parent_id=None):
        if "id" not in node:
            return
        
        node_id = node["id"]
        score = node.get("score", 0.0)
        generation = node.get("generation", 0)
        
        # Tambahkan node
        G.add_node(node_id, score=score, generation=generation)
        
        # Tambahkan edge jika ada parent
        if parent_id:
            G.add_edge(parent_id, node_id)
        
        # Proses anak-anak
        for child in node.get("children", []):
            add_nodes_and_edges(child, node_id)
    
    # Mulai dari root
    if "children" in tree:
        for root_child in tree["children"]:
            add_nodes_and_edges(root_child)
    
    # Buat layout berdasarkan generasi
    pos = {}
    generation_nodes = {}
    
    # Kelompokkan node berdasarkan generasi
    for node_id in G.nodes():
        generation = G.nodes[node_id]["generation"]
        if generation not in generation_nodes:
            generation_nodes[generation] = []
        generation_nodes[generation].append(node_id)
    
    # Atur posisi node
    max_generation = max(generation_nodes.keys()) if generation_nodes else 0
    for generation, nodes in generation_nodes.items():
        y = 1.0 - generation / (max_generation + 1)
        for i, node_id in enumerate(nodes):
            x = (i + 1) / (len(nodes) + 1)
            pos[node_id] = (x, y)
    
    # Buat plot
    plt.figure(figsize=figsize)
    
    # Dapatkan skor untuk pewarnaan
    scores = [G.nodes[node]["score"] for node in G.nodes()]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1
    
    # Buat colormap
    cmap = LinearSegmentedColormap.from_list("score_cmap", ["#FF9999", "#FFFF99", "#99FF99"])
    
    # Normalisasi skor
    norm_scores = [(score - min_score) / (max_score - min_score) if max_score > min_score else 0.5 for score in scores]
    
    # Gambar node
    nx.draw_networkx_nodes(G, pos, 
                          node_color=norm_scores, 
                          cmap=cmap, 
                          node_size=500, 
                          alpha=0.8)
    
    # Gambar edge
    nx.draw_networkx_edges(G, pos, 
                          edge_color="gray", 
                          width=1.0, 
                          alpha=0.5, 
                          arrows=True, 
                          arrowsize=15)
    
    # Tambahkan label
    labels = {node: f"{node[:4]}...\n{G.nodes[node]['score']:.2f}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Tambahkan judul dan label
    plt.title("DGM Evolution Tree")
    plt.xlabel("Agent Variants")
    plt.ylabel("Generations")
    
    # Tambahkan colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    try:
        cbar = plt.colorbar(sm)
        cbar.set_label("Agent Score")
    except ValueError:
        # Lewati colorbar jika ada masalah
        pass
    
    # Hapus sumbu
    plt.axis("off")
    
    # Simpan plot jika path diberikan
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    # Tampilkan plot jika diminta
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_performance_history(dgm, output_path: Optional[str] = None, 
                                 show_plot: bool = True, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualisasikan riwayat performa DGM.
    
    Args:
        dgm: Objek DGM
        output_path: Path untuk menyimpan visualisasi (opsional)
        show_plot: Apakah akan menampilkan plot
        figsize: Ukuran gambar
    """
    if not VISUALIZATION_AVAILABLE:
        print("Matplotlib diperlukan untuk visualisasi. Instal dengan 'pip install matplotlib'.")
        return
    
    # Dapatkan riwayat evolusi
    history = dgm.evolution_history
    
    # Ekstrak generasi dan skor
    generations = [entry[3] for entry in history]
    scores = [entry[2] for entry in history]
    
    # Buat plot
    plt.figure(figsize=figsize)
    
    # Plot skor vs generasi
    plt.plot(generations, scores, "o-", color="#3498db", alpha=0.7)
    
    # Tambahkan garis tren
    if len(generations) > 1:
        try:
            z = np.polyfit(generations, scores, 1)
            p = np.poly1d(z)
            plt.plot(generations, p(generations), "--", color="#e74c3c", alpha=0.7, 
                    label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
            plt.legend()
        except:
            pass
    
    # Tambahkan judul dan label
    plt.title("DGM Performance History")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    
    # Tambahkan grid
    plt.grid(True, alpha=0.3)
    
    # Simpan plot jika path diberikan
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    # Tampilkan plot jika diminta
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_agent_comparison(agents: List[Dict[str, Any]], metrics: List[str], 
                              output_path: Optional[str] = None, 
                              show_plot: bool = True, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualisasikan perbandingan agen.
    
    Args:
        agents: Daftar agen (dict dengan keys: name, metrics)
        metrics: Daftar metrik untuk dibandingkan
        output_path: Path untuk menyimpan visualisasi (opsional)
        show_plot: Apakah akan menampilkan plot
        figsize: Ukuran gambar
    """
    if not VISUALIZATION_AVAILABLE:
        print("Matplotlib diperlukan untuk visualisasi. Instal dengan 'pip install matplotlib'.")
        return
    
    # Periksa apakah ada agen dan metrik
    if not agents or not metrics:
        print("Tidak ada agen atau metrik untuk divisualisasikan.")
        return
    
    # Buat plot
    plt.figure(figsize=figsize)
    
    # Siapkan data
    agent_names = [agent["name"] for agent in agents]
    x = np.arange(len(agent_names))
    width = 0.8 / len(metrics)
    
    # Plot batang untuk setiap metrik
    for i, metric in enumerate(metrics):
        values = [agent["metrics"].get(metric, 0) for agent in agents]
        plt.bar(x + i * width, values, width, label=metric, alpha=0.7)
    
    # Tambahkan judul dan label
    plt.title("Agent Comparison")
    plt.xlabel("Agent")
    plt.ylabel("Score")
    
    # Tambahkan label sumbu x
    plt.xticks(x + width * (len(metrics) - 1) / 2, agent_names)
    
    # Tambahkan legenda
    plt.legend()
    
    # Tambahkan grid
    plt.grid(True, alpha=0.3, axis="y")
    
    # Simpan plot jika path diberikan
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    # Tampilkan plot jika diminta
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_agent_network(agents: List[Dict[str, Any]], connections: List[Tuple[int, int]], 
                           output_path: Optional[str] = None, 
                           show_plot: bool = True, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualisasikan jaringan agen.
    
    Args:
        agents: Daftar agen (dict dengan keys: id, name, type, score)
        connections: Daftar koneksi antar agen (tuple dari indeks agen)
        output_path: Path untuk menyimpan visualisasi (opsional)
        show_plot: Apakah akan menampilkan plot
        figsize: Ukuran gambar
    """
    if not VISUALIZATION_AVAILABLE:
        print("Matplotlib dan NetworkX diperlukan untuk visualisasi. Instal dengan 'pip install matplotlib networkx'.")
        return
    
    # Periksa apakah ada agen dan koneksi
    if not agents:
        print("Tidak ada agen untuk divisualisasikan.")
        return
    
    # Buat graf
    G = nx.Graph()
    
    # Tambahkan node
    for i, agent in enumerate(agents):
        G.add_node(i, **agent)
    
    # Tambahkan edge
    for source, target in connections:
        G.add_edge(source, target)
    
    # Buat layout
    pos = nx.spring_layout(G, seed=42)
    
    # Buat plot
    plt.figure(figsize=figsize)
    
    # Dapatkan tipe agen untuk pewarnaan
    agent_types = list(set(agent["type"] for agent in agents))
    type_colors = plt.cm.tab10(np.linspace(0, 1, len(agent_types)))
    type_color_map = {agent_type: color for agent_type, color in zip(agent_types, type_colors)}
    
    # Dapatkan skor untuk ukuran node
    scores = [agent["score"] for agent in agents]
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1
    
    # Normalisasi skor untuk ukuran node
    node_sizes = [300 + 700 * ((agent["score"] - min_score) / (max_score - min_score) if max_score > min_score else 0.5) for agent in agents]
    
    # Gambar node
    for agent_type in agent_types:
        node_indices = [i for i, agent in enumerate(agents) if agent["type"] == agent_type]
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=node_indices,
                              node_color=[type_color_map[agent_type]] * len(node_indices), 
                              node_size=[node_sizes[i] for i in node_indices], 
                              alpha=0.8,
                              label=agent_type)
    
    # Gambar edge
    nx.draw_networkx_edges(G, pos, 
                          edge_color="gray", 
                          width=1.0, 
                          alpha=0.5)
    
    # Tambahkan label
    labels = {i: agent["name"] for i, agent in enumerate(agents)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Tambahkan judul dan legenda
    plt.title("Agent Network")
    plt.legend()
    
    # Hapus sumbu
    plt.axis("off")
    
    # Simpan plot jika path diberikan
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    # Tampilkan plot jika diminta
    if show_plot:
        plt.show()
    else:
        plt.close()