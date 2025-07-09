"""
Mesin kolaborasi untuk Darwin-Gödel Machine.

Modul ini berisi implementasi mesin kolaborasi yang digunakan oleh DGM
untuk memungkinkan kolaborasi antar agen.
"""

import random
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar, Generic, Union, Set

from simple_dgm.agents.base_agent import BaseAgent, Tool

# Tipe generik untuk individu dalam populasi
T = TypeVar('T', bound=BaseAgent)


class CollaborationEngine:
    """
    Mesin kolaborasi untuk DGM.
    
    Mesin ini memungkinkan kolaborasi antar agen dalam DGM.
    """
    
    def __init__(self):
        """
        Inisialisasi mesin kolaborasi.
        """
        self.agent_network = AgentNetwork()
        self.knowledge_sharing = KnowledgeSharing()
        self.task_decomposition = TaskDecomposition()
        self.consensus_builder = ConsensusBuilder()
        self.collaboration_history = []
    
    def add_agent(self, agent: BaseAgent, agent_id: str, agent_type: str):
        """
        Tambahkan agen ke jaringan kolaborasi.
        
        Args:
            agent: Agen yang akan ditambahkan
            agent_id: ID agen
            agent_type: Tipe agen
        """
        self.agent_network.add_agent(agent, agent_id, agent_type)
    
    def remove_agent(self, agent_id: str):
        """
        Hapus agen dari jaringan kolaborasi.
        
        Args:
            agent_id: ID agen
        """
        self.agent_network.remove_agent(agent_id)
    
    def connect_agents(self, agent_id1: str, agent_id2: str, weight: float = 1.0):
        """
        Hubungkan dua agen dalam jaringan kolaborasi.
        
        Args:
            agent_id1: ID agen pertama
            agent_id2: ID agen kedua
            weight: Bobot koneksi
        """
        self.agent_network.connect_agents(agent_id1, agent_id2, weight)
    
    def disconnect_agents(self, agent_id1: str, agent_id2: str):
        """
        Putuskan hubungan dua agen dalam jaringan kolaborasi.
        
        Args:
            agent_id1: ID agen pertama
            agent_id2: ID agen kedua
        """
        self.agent_network.disconnect_agents(agent_id1, agent_id2)
    
    def share_knowledge(self, source_id: str, target_id: str, 
                       knowledge_type: str, knowledge: Any):
        """
        Bagikan pengetahuan antar agen.
        
        Args:
            source_id: ID agen sumber
            target_id: ID agen target
            knowledge_type: Tipe pengetahuan
            knowledge: Pengetahuan yang akan dibagikan
        """
        self.knowledge_sharing.share_knowledge(
            self.agent_network.get_agent(source_id),
            self.agent_network.get_agent(target_id),
            knowledge_type,
            knowledge
        )
    
    def decompose_task(self, task: Any, agent_ids: List[str]) -> Dict[str, Any]:
        """
        Dekomposisi tugas untuk agen-agen.
        
        Args:
            task: Tugas yang akan didekomposisi
            agent_ids: Daftar ID agen
            
        Returns:
            Tugas yang didekomposisi untuk setiap agen
        """
        agents = [self.agent_network.get_agent(agent_id) for agent_id in agent_ids]
        return self.task_decomposition.decompose_task(task, agents)
    
    def build_consensus(self, agent_ids: List[str], 
                       solutions: List[Any]) -> Any:
        """
        Bangun konsensus dari solusi-solusi agen.
        
        Args:
            agent_ids: Daftar ID agen
            solutions: Daftar solusi
            
        Returns:
            Solusi konsensus
        """
        agents = [self.agent_network.get_agent(agent_id) for agent_id in agent_ids]
        return self.consensus_builder.build_consensus(agents, solutions)
    
    def collaborate(self, task: Any, agent_ids: Optional[List[str]] = None) -> Any:
        """
        Kolaborasi antar agen untuk menyelesaikan tugas.
        
        Args:
            task: Tugas yang akan diselesaikan
            agent_ids: Daftar ID agen (opsional)
            
        Returns:
            Solusi kolaboratif
        """
        # Jika agent_ids tidak diberikan, gunakan semua agen
        if agent_ids is None:
            agent_ids = list(self.agent_network.agents.keys())
        
        # Dekomposisi tugas
        subtasks = self.decompose_task(task, agent_ids)
        
        # Selesaikan subtasks
        solutions = {}
        for agent_id, subtask in subtasks.items():
            agent = self.agent_network.get_agent(agent_id)
            try:
                solution = agent.solve(subtask)
                solutions[agent_id] = solution
            except Exception as e:
                solutions[agent_id] = {"error": str(e)}
        
        # Bangun konsensus
        consensus = self.consensus_builder.build_consensus(
            [self.agent_network.get_agent(agent_id) for agent_id in agent_ids],
            list(solutions.values())
        )
        
        # Catat kolaborasi
        self.collaboration_history.append({
            "task": task,
            "agent_ids": agent_ids,
            "subtasks": subtasks,
            "solutions": solutions,
            "consensus": consensus
        })
        
        return consensus


class AgentNetwork:
    """
    Jaringan agen untuk mesin kolaborasi.
    
    Jaringan ini merepresentasikan hubungan antar agen.
    """
    
    def __init__(self):
        """
        Inisialisasi jaringan agen.
        """
        self.agents = {}  # Tipe: Dict[str, BaseAgent]
        self.agent_types = {}  # Tipe: Dict[str, str]
        self.graph = nx.Graph()
    
    def add_agent(self, agent: BaseAgent, agent_id: str, agent_type: str):
        """
        Tambahkan agen ke jaringan.
        
        Args:
            agent: Agen yang akan ditambahkan
            agent_id: ID agen
            agent_type: Tipe agen
        """
        self.agents[agent_id] = agent
        self.agent_types[agent_id] = agent_type
        self.graph.add_node(agent_id, type=agent_type)
    
    def remove_agent(self, agent_id: str):
        """
        Hapus agen dari jaringan.
        
        Args:
            agent_id: ID agen
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_types[agent_id]
            self.graph.remove_node(agent_id)
    
    def get_agent(self, agent_id: str) -> BaseAgent:
        """
        Dapatkan agen dari jaringan.
        
        Args:
            agent_id: ID agen
            
        Returns:
            Agen
        """
        return self.agents.get(agent_id)
    
    def get_agent_type(self, agent_id: str) -> str:
        """
        Dapatkan tipe agen.
        
        Args:
            agent_id: ID agen
            
        Returns:
            Tipe agen
        """
        return self.agent_types.get(agent_id, "unknown")
    
    def connect_agents(self, agent_id1: str, agent_id2: str, weight: float = 1.0):
        """
        Hubungkan dua agen dalam jaringan.
        
        Args:
            agent_id1: ID agen pertama
            agent_id2: ID agen kedua
            weight: Bobot koneksi
        """
        if agent_id1 in self.agents and agent_id2 in self.agents:
            self.graph.add_edge(agent_id1, agent_id2, weight=weight)
    
    def disconnect_agents(self, agent_id1: str, agent_id2: str):
        """
        Putuskan hubungan dua agen dalam jaringan.
        
        Args:
            agent_id1: ID agen pertama
            agent_id2: ID agen kedua
        """
        if self.graph.has_edge(agent_id1, agent_id2):
            self.graph.remove_edge(agent_id1, agent_id2)
    
    def get_neighbors(self, agent_id: str) -> List[str]:
        """
        Dapatkan tetangga agen dalam jaringan.
        
        Args:
            agent_id: ID agen
            
        Returns:
            Daftar ID agen tetangga
        """
        if agent_id in self.graph:
            return list(self.graph.neighbors(agent_id))
        return []
    
    def get_connection_weight(self, agent_id1: str, agent_id2: str) -> float:
        """
        Dapatkan bobot koneksi antara dua agen.
        
        Args:
            agent_id1: ID agen pertama
            agent_id2: ID agen kedua
            
        Returns:
            Bobot koneksi
        """
        if self.graph.has_edge(agent_id1, agent_id2):
            return self.graph.get_edge_data(agent_id1, agent_id2)["weight"]
        return 0.0
    
    def get_centrality(self, agent_id: str) -> float:
        """
        Dapatkan sentralitas agen dalam jaringan.
        
        Args:
            agent_id: ID agen
            
        Returns:
            Sentralitas agen
        """
        if agent_id in self.graph:
            return nx.degree_centrality(self.graph)[agent_id]
        return 0.0
    
    def get_communities(self) -> List[List[str]]:
        """
        Dapatkan komunitas dalam jaringan.
        
        Returns:
            Daftar komunitas (daftar ID agen)
        """
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            communities = {}
            
            for agent_id, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(agent_id)
            
            return list(communities.values())
        except ImportError:
            # Fallback jika python-louvain tidak tersedia
            return [list(self.graph.nodes())]
    
    def get_subgraph(self, agent_ids: List[str]) -> 'AgentNetwork':
        """
        Dapatkan subgraf dari jaringan.
        
        Args:
            agent_ids: Daftar ID agen
            
        Returns:
            Subgraf jaringan
        """
        subgraph = AgentNetwork()
        
        for agent_id in agent_ids:
            if agent_id in self.agents:
                subgraph.add_agent(
                    self.agents[agent_id],
                    agent_id,
                    self.agent_types[agent_id]
                )
        
        for agent_id1 in agent_ids:
            for agent_id2 in agent_ids:
                if agent_id1 != agent_id2 and self.graph.has_edge(agent_id1, agent_id2):
                    weight = self.get_connection_weight(agent_id1, agent_id2)
                    subgraph.connect_agents(agent_id1, agent_id2, weight)
        
        return subgraph


class KnowledgeSharing:
    """
    Berbagi pengetahuan untuk mesin kolaborasi.
    
    Kelas ini mengelola berbagi pengetahuan antar agen.
    """
    
    def __init__(self):
        """
        Inisialisasi berbagi pengetahuan.
        """
        self.knowledge_base = {}  # Tipe: Dict[str, Dict[str, Any]]
        self.sharing_history = []
    
    def share_knowledge(self, source_agent: BaseAgent, target_agent: BaseAgent,
                       knowledge_type: str, knowledge: Any) -> bool:
        """
        Bagikan pengetahuan dari agen sumber ke agen target.
        
        Args:
            source_agent: Agen sumber
            target_agent: Agen target
            knowledge_type: Tipe pengetahuan
            knowledge: Pengetahuan yang akan dibagikan
            
        Returns:
            True jika berhasil, False jika tidak
        """
        if source_agent is None or target_agent is None:
            return False
        
        # Catat berbagi pengetahuan
        self.sharing_history.append({
            "source_agent": id(source_agent),
            "target_agent": id(target_agent),
            "knowledge_type": knowledge_type,
            "timestamp": time.time()
        })
        
        # Berbagi pengetahuan berdasarkan tipe
        if knowledge_type == "parameter":
            return self._share_parameter(source_agent, target_agent, knowledge)
        elif knowledge_type == "tool":
            return self._share_tool(source_agent, target_agent, knowledge)
        elif knowledge_type == "memory":
            return self._share_memory(source_agent, target_agent, knowledge)
        elif knowledge_type == "strategy":
            return self._share_strategy(source_agent, target_agent, knowledge)
        else:
            # Simpan ke basis pengetahuan
            agent_id = id(target_agent)
            if agent_id not in self.knowledge_base:
                self.knowledge_base[agent_id] = {}
            self.knowledge_base[agent_id][knowledge_type] = knowledge
            return True
    
    def _share_parameter(self, source_agent: BaseAgent, target_agent: BaseAgent,
                        parameter: Dict[str, Any]) -> bool:
        """
        Bagikan parameter dari agen sumber ke agen target.
        
        Args:
            source_agent: Agen sumber
            target_agent: Agen target
            parameter: Parameter yang akan dibagikan
            
        Returns:
            True jika berhasil, False jika tidak
        """
        success = False
        
        for param_name, param_value in parameter.items():
            if hasattr(source_agent, param_name) and hasattr(target_agent, param_name):
                # Dapatkan nilai parameter dari agen sumber
                source_value = getattr(source_agent, param_name)
                
                # Tetapkan nilai parameter ke agen target
                setattr(target_agent, param_name, source_value)
                success = True
        
        return success
    
    def _share_tool(self, source_agent: BaseAgent, target_agent: BaseAgent,
                   tool_name: str) -> bool:
        """
        Bagikan alat dari agen sumber ke agen target.
        
        Args:
            source_agent: Agen sumber
            target_agent: Agen target
            tool_name: Nama alat yang akan dibagikan
            
        Returns:
            True jika berhasil, False jika tidak
        """
        # Periksa apakah agen sumber memiliki alat
        source_tool = None
        for tool in source_agent.tools:
            if tool.name == tool_name:
                source_tool = tool
                break
        
        if source_tool is None:
            return False
        
        # Periksa apakah agen target sudah memiliki alat
        for tool in target_agent.tools:
            if tool.name == tool_name:
                return False
        
        # Salin alat ke agen target
        target_agent.add_tool(Tool(
            name=source_tool.name,
            function=source_tool.function,
            description=source_tool.description
        ))
        
        return True
    
    def _share_memory(self, source_agent: BaseAgent, target_agent: BaseAgent,
                     memory_entries: List[Dict[str, Any]]) -> bool:
        """
        Bagikan memori dari agen sumber ke agen target.
        
        Args:
            source_agent: Agen sumber
            target_agent: Agen target
            memory_entries: Entri memori yang akan dibagikan
            
        Returns:
            True jika berhasil, False jika tidak
        """
        if not hasattr(target_agent, "_add_to_memory"):
            return False
        
        # Tambahkan entri memori ke agen target
        for entry in memory_entries:
            target_agent._add_to_memory(entry)
        
        return True
    
    def _share_strategy(self, source_agent: BaseAgent, target_agent: BaseAgent,
                       strategy_name: str) -> bool:
        """
        Bagikan strategi dari agen sumber ke agen target.
        
        Args:
            source_agent: Agen sumber
            target_agent: Agen target
            strategy_name: Nama strategi yang akan dibagikan
            
        Returns:
            True jika berhasil, False jika tidak
        """
        strategy_attr = f"_{strategy_name}_strategy"
        
        if hasattr(source_agent, strategy_attr) and hasattr(target_agent, strategy_attr):
            # Dapatkan strategi dari agen sumber
            strategy = getattr(source_agent, strategy_attr)
            
            # Tetapkan strategi ke agen target
            setattr(target_agent, strategy_attr, strategy)
            return True
        
        return False
    
    def get_knowledge(self, agent_id: int, knowledge_type: str) -> Any:
        """
        Dapatkan pengetahuan dari basis pengetahuan.
        
        Args:
            agent_id: ID agen
            knowledge_type: Tipe pengetahuan
            
        Returns:
            Pengetahuan
        """
        if agent_id in self.knowledge_base and knowledge_type in self.knowledge_base[agent_id]:
            return self.knowledge_base[agent_id][knowledge_type]
        return None
    
    def get_sharing_statistics(self) -> Dict[str, Any]:
        """
        Dapatkan statistik berbagi pengetahuan.
        
        Returns:
            Statistik berbagi pengetahuan
        """
        stats = {
            "total_shares": len(self.sharing_history),
            "by_type": {},
            "by_agent": {}
        }
        
        # Hitung berbagi berdasarkan tipe
        for entry in self.sharing_history:
            knowledge_type = entry["knowledge_type"]
            stats["by_type"][knowledge_type] = stats["by_type"].get(knowledge_type, 0) + 1
        
        # Hitung berbagi berdasarkan agen
        for entry in self.sharing_history:
            source_agent = entry["source_agent"]
            target_agent = entry["target_agent"]
            
            if source_agent not in stats["by_agent"]:
                stats["by_agent"][source_agent] = {"shared": 0, "received": 0}
            if target_agent not in stats["by_agent"]:
                stats["by_agent"][target_agent] = {"shared": 0, "received": 0}
            
            stats["by_agent"][source_agent]["shared"] += 1
            stats["by_agent"][target_agent]["received"] += 1
        
        return stats


class TaskDecomposition:
    """
    Dekomposisi tugas untuk mesin kolaborasi.
    
    Kelas ini mendekomposisi tugas untuk agen-agen.
    """
    
    def __init__(self):
        """
        Inisialisasi dekomposisi tugas.
        """
        pass
    
    def decompose_task(self, task: Any, agents: List[BaseAgent]) -> Dict[str, Any]:
        """
        Dekomposisi tugas untuk agen-agen.
        
        Args:
            task: Tugas yang akan didekomposisi
            agents: Daftar agen
            
        Returns:
            Tugas yang didekomposisi untuk setiap agen
        """
        if not agents:
            return {}
        
        # Tentukan metode dekomposisi berdasarkan tipe tugas
        if isinstance(task, dict):
            return self._decompose_dict_task(task, agents)
        elif isinstance(task, list):
            return self._decompose_list_task(task, agents)
        else:
            # Tugas tidak dapat didekomposisi, berikan ke semua agen
            return {id(agent): task for agent in agents}
    
    def _decompose_dict_task(self, task: Dict[str, Any], 
                            agents: List[BaseAgent]) -> Dict[str, Any]:
        """
        Dekomposisi tugas dictionary.
        
        Args:
            task: Tugas dictionary
            agents: Daftar agen
            
        Returns:
            Tugas yang didekomposisi untuk setiap agen
        """
        subtasks = {}
        
        # Periksa apakah tugas memiliki kunci khusus
        if "subtasks" in task:
            # Tugas sudah memiliki subtasks
            subtask_list = task["subtasks"]
            
            # Alokasikan subtasks ke agen
            for i, agent in enumerate(agents):
                if i < len(subtask_list):
                    subtasks[id(agent)] = subtask_list[i]
                else:
                    # Jika ada lebih banyak agen daripada subtasks, berikan tugas kosong
                    subtasks[id(agent)] = {}
        
        elif "parts" in task:
            # Tugas memiliki bagian-bagian
            parts = task["parts"]
            
            # Alokasikan bagian ke agen
            for i, agent in enumerate(agents):
                if i < len(parts):
                    subtasks[id(agent)] = parts[i]
                else:
                    # Jika ada lebih banyak agen daripada bagian, berikan tugas kosong
                    subtasks[id(agent)] = {}
        
        else:
            # Dekomposisi berdasarkan kunci
            keys = list(task.keys())
            
            # Bagi kunci menjadi kelompok
            num_agents = len(agents)
            key_groups = [[] for _ in range(num_agents)]
            
            for i, key in enumerate(keys):
                key_groups[i % num_agents].append(key)
            
            # Buat subtasks
            for i, agent in enumerate(agents):
                agent_keys = key_groups[i]
                subtasks[id(agent)] = {key: task[key] for key in agent_keys}
        
        return subtasks
    
    def _decompose_list_task(self, task: List[Any], 
                            agents: List[BaseAgent]) -> Dict[str, Any]:
        """
        Dekomposisi tugas daftar.
        
        Args:
            task: Tugas daftar
            agents: Daftar agen
            
        Returns:
            Tugas yang didekomposisi untuk setiap agen
        """
        subtasks = {}
        
        # Bagi daftar menjadi bagian-bagian
        num_agents = len(agents)
        chunk_size = max(1, len(task) // num_agents)
        
        for i, agent in enumerate(agents):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_agents - 1 else len(task)
            
            subtasks[id(agent)] = task[start:end]
        
        return subtasks


class ConsensusBuilder:
    """
    Pembangun konsensus untuk mesin kolaborasi.
    
    Kelas ini membangun konsensus dari solusi-solusi agen.
    """
    
    def __init__(self):
        """
        Inisialisasi pembangun konsensus.
        """
        pass
    
    def build_consensus(self, agents: List[BaseAgent], 
                       solutions: List[Any]) -> Any:
        """
        Bangun konsensus dari solusi-solusi agen.
        
        Args:
            agents: Daftar agen
            solutions: Daftar solusi
            
        Returns:
            Solusi konsensus
        """
        if not solutions:
            return None
        
        # Tentukan metode konsensus berdasarkan tipe solusi
        if all(isinstance(sol, dict) for sol in solutions):
            return self._build_dict_consensus(solutions)
        elif all(isinstance(sol, list) for sol in solutions):
            return self._build_list_consensus(solutions)
        elif all(isinstance(sol, (int, float)) for sol in solutions):
            return self._build_numeric_consensus(solutions)
        elif all(isinstance(sol, str) for sol in solutions):
            return self._build_string_consensus(solutions)
        else:
            # Tipe solusi campuran, gunakan voting
            return self._build_voting_consensus(solutions)
    
    def _build_dict_consensus(self, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bangun konsensus dari solusi dictionary.
        
        Args:
            solutions: Daftar solusi dictionary
            
        Returns:
            Solusi konsensus
        """
        consensus = {}
        
        # Gabungkan semua kunci
        all_keys = set()
        for sol in solutions:
            all_keys.update(sol.keys())
        
        # Untuk setiap kunci, bangun konsensus
        for key in all_keys:
            # Kumpulkan nilai untuk kunci ini
            values = [sol.get(key) for sol in solutions if key in sol]
            
            if not values:
                continue
            
            # Bangun konsensus berdasarkan tipe nilai
            if all(isinstance(val, dict) for val in values):
                consensus[key] = self._build_dict_consensus(values)
            elif all(isinstance(val, list) for val in values):
                consensus[key] = self._build_list_consensus(values)
            elif all(isinstance(val, (int, float)) for val in values):
                consensus[key] = self._build_numeric_consensus(values)
            elif all(isinstance(val, str) for val in values):
                consensus[key] = self._build_string_consensus(values)
            else:
                # Tipe nilai campuran, gunakan voting
                consensus[key] = self._build_voting_consensus(values)
        
        return consensus
    
    def _build_list_consensus(self, solutions: List[List[Any]]) -> List[Any]:
        """
        Bangun konsensus dari solusi daftar.
        
        Args:
            solutions: Daftar solusi daftar
            
        Returns:
            Solusi konsensus
        """
        # Gabungkan semua daftar
        all_items = []
        for sol in solutions:
            all_items.extend(sol)
        
        # Jika daftar kosong, kembalikan daftar kosong
        if not all_items:
            return []
        
        # Bangun konsensus berdasarkan tipe item
        if all(isinstance(item, dict) for item in all_items):
            # Kelompokkan item berdasarkan kunci yang sama
            groups = {}
            for item in all_items:
                key = frozenset(item.keys())
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)
            
            # Bangun konsensus untuk setiap kelompok
            consensus = []
            for items in groups.values():
                consensus.append(self._build_dict_consensus(items))
            
            return consensus
        
        elif all(isinstance(item, list) for item in all_items):
            # Kelompokkan item berdasarkan panjang
            groups = {}
            for item in all_items:
                key = len(item)
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)
            
            # Bangun konsensus untuk setiap kelompok
            consensus = []
            for items in groups.values():
                consensus.append(self._build_list_consensus(items))
            
            return consensus
        
        elif all(isinstance(item, (int, float)) for item in all_items):
            # Kembalikan daftar nilai numerik yang diurutkan
            return sorted(all_items)
        
        elif all(isinstance(item, str) for item in all_items):
            # Kembalikan daftar string yang diurutkan
            return sorted(all_items)
        
        else:
            # Tipe item campuran, kembalikan daftar asli
            return all_items
    
    def _build_numeric_consensus(self, solutions: List[Union[int, float]]) -> Union[int, float]:
        """
        Bangun konsensus dari solusi numerik.
        
        Args:
            solutions: Daftar solusi numerik
            
        Returns:
            Solusi konsensus
        """
        # Gunakan median untuk menghindari outlier
        return float(np.median(solutions))
    
    def _build_string_consensus(self, solutions: List[str]) -> str:
        """
        Bangun konsensus dari solusi string.
        
        Args:
            solutions: Daftar solusi string
            
        Returns:
            Solusi konsensus
        """
        # Gunakan voting
        counts = {}
        for sol in solutions:
            counts[sol] = counts.get(sol, 0) + 1
        
        # Kembalikan string dengan jumlah terbanyak
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def _build_voting_consensus(self, solutions: List[Any]) -> Any:
        """
        Bangun konsensus dari solusi menggunakan voting.
        
        Args:
            solutions: Daftar solusi
            
        Returns:
            Solusi konsensus
        """
        # Hitung frekuensi setiap solusi
        counts = {}
        for sol in solutions:
            # Konversi ke string untuk membuat hashable
            sol_str = str(sol)
            counts[sol_str] = counts.get(sol_str, 0) + 1
        
        # Kembalikan solusi dengan jumlah terbanyak
        most_common = max(counts.items(), key=lambda x: x[1])[0]
        
        # Temukan solusi asli
        for sol in solutions:
            if str(sol) == most_common:
                return sol
        
        # Fallback ke solusi pertama
        return solutions[0]