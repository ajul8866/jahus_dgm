"""
Implementasi agen dasar untuk Darwin-Gödel Machine.
"""

import random
import json
import inspect
import copy
from typing import Dict, Any, List, Optional, Callable, Union, Tuple


class Tool:
    """
    Representasi alat yang dapat digunakan oleh agen.
    """
    
    def __init__(self, name: str, function: Callable, description: str):
        """
        Inisialisasi alat.
        
        Args:
            name: Nama alat
            function: Fungsi yang diimplementasikan oleh alat
            description: Deskripsi alat
        """
        self.name = name
        self.function = function
        self.description = description
        self.signature = inspect.signature(function)
    
    def __call__(self, *args, **kwargs):
        """
        Panggil fungsi alat.
        """
        return self.function(*args, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konversi alat ke dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": str(self.signature)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], function_map: Dict[str, Callable]) -> 'Tool':
        """
        Buat alat dari dictionary.
        
        Args:
            data: Data alat dalam format dictionary
            function_map: Pemetaan nama fungsi ke implementasi fungsi
            
        Returns:
            Objek Tool
        """
        return cls(
            name=data["name"],
            function=function_map[data["name"]],
            description=data["description"]
        )


class BaseAgent:
    """
    Agen dasar untuk Darwin-Gödel Machine.
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None, 
                 memory_capacity: int = 10,
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.1):
        """
        Inisialisasi agen dasar.
        
        Args:
            tools: Daftar alat yang tersedia untuk agen
            memory_capacity: Kapasitas memori agen
            learning_rate: Tingkat pembelajaran agen
            exploration_rate: Tingkat eksplorasi agen
        """
        self.tools = tools or []
        self.memory = []
        self.memory_capacity = memory_capacity
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.performance_history = []
        self.metadata = {
            "type": self.__class__.__name__,
            "version": "1.0.0",
            "created_at": None,
            "modified_at": None
        }
    
    def add_tool(self, tool: Tool) -> None:
        """
        Tambahkan alat baru ke agen.
        
        Args:
            tool: Alat yang akan ditambahkan
        """
        self.tools.append(tool)
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        Hapus alat dari agen.
        
        Args:
            tool_name: Nama alat yang akan dihapus
            
        Returns:
            True jika alat berhasil dihapus, False jika tidak ditemukan
        """
        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                self.tools.pop(i)
                return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Dapatkan alat berdasarkan nama.
        
        Args:
            tool_name: Nama alat
            
        Returns:
            Objek Tool jika ditemukan, None jika tidak
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def solve(self, problem: Any) -> Any:
        """
        Selesaikan masalah menggunakan alat yang tersedia.
        
        Args:
            problem: Masalah yang akan diselesaikan
            
        Returns:
            Solusi untuk masalah
        """
        # Implementasi dasar: pilih alat secara acak dan gunakan
        if not self.tools:
            return None
        
        # Dengan probabilitas exploration_rate, pilih alat secara acak
        if random.random() < self.exploration_rate:
            selected_tool = random.choice(self.tools)
        else:
            # Pilih alat berdasarkan performa sebelumnya (implementasi sederhana)
            selected_tool = self.tools[0]  # Default ke alat pertama
        
        # Coba gunakan alat untuk menyelesaikan masalah
        try:
            result = selected_tool(problem)
            # Simpan hasil ke memori
            self._add_to_memory({"problem": problem, "tool": selected_tool.name, "result": result})
            return result
        except Exception as e:
            # Tangani kegagalan
            self._add_to_memory({"problem": problem, "tool": selected_tool.name, "error": str(e)})
            return None
    
    def _add_to_memory(self, entry: Dict[str, Any]) -> None:
        """
        Tambahkan entri ke memori agen.
        
        Args:
            entry: Entri yang akan ditambahkan
        """
        self.memory.append(entry)
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)  # Hapus entri tertua
    
    def mutate(self) -> None:
        """
        Mutasi agen untuk menghasilkan variasi.
        Metode ini akan diimplementasikan oleh subclass.
        """
        # Implementasi dasar: mutasi parameter agen
        self.learning_rate *= random.uniform(0.8, 1.2)
        self.exploration_rate *= random.uniform(0.8, 1.2)
        
        # Batasi nilai parameter
        self.learning_rate = max(0.001, min(0.1, self.learning_rate))
        self.exploration_rate = max(0.01, min(0.5, self.exploration_rate))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konversi agen ke dictionary.
        
        Returns:
            Representasi agen dalam format dictionary
        """
        return {
            "type": self.__class__.__name__,
            "tools": [tool.to_dict() for tool in self.tools],
            "memory_capacity": self.memory_capacity,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], function_map: Dict[str, Callable]) -> 'BaseAgent':
        """
        Buat agen dari dictionary.
        
        Args:
            data: Data agen dalam format dictionary
            function_map: Pemetaan nama fungsi ke implementasi fungsi
            
        Returns:
            Objek BaseAgent
        """
        agent = cls(
            memory_capacity=data["memory_capacity"],
            learning_rate=data["learning_rate"],
            exploration_rate=data["exploration_rate"]
        )
        
        # Tambahkan alat
        for tool_data in data["tools"]:
            tool = Tool.from_dict(tool_data, function_map)
            agent.add_tool(tool)
        
        # Set metadata
        agent.metadata = data["metadata"]
        
        return agent
    
    def save(self, filepath: str) -> None:
        """
        Simpan agen ke file.
        
        Args:
            filepath: Path file untuk menyimpan agen
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, function_map: Dict[str, Callable]) -> 'BaseAgent':
        """
        Muat agen dari file.
        
        Args:
            filepath: Path file untuk memuat agen
            function_map: Pemetaan nama fungsi ke implementasi fungsi
            
        Returns:
            Objek BaseAgent
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, function_map)