"""
Utilitas serialisasi untuk Darwin-Gödel Machine.
"""

import os
import json
import pickle
import importlib
from typing import Dict, Any, List, Optional, Callable, Union, Tuple, Type

from simple_dgm.agents.base_agent import BaseAgent


def save_agent(agent: BaseAgent, filepath: str, format: str = "json") -> None:
    """
    Simpan agen ke file.
    
    Args:
        agent: Agen yang akan disimpan
        filepath: Path file untuk menyimpan agen
        format: Format penyimpanan ("json" atau "pickle")
    """
    if format == "json":
        # Simpan sebagai JSON
        with open(filepath, 'w') as f:
            json.dump(agent.to_dict(), f, indent=2)
    
    elif format == "pickle":
        # Simpan sebagai pickle
        with open(filepath, 'wb') as f:
            pickle.dump(agent, f)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_agent(filepath: str, format: str = "json", function_map: Optional[Dict[str, Callable]] = None) -> BaseAgent:
    """
    Muat agen dari file.
    
    Args:
        filepath: Path file untuk memuat agen
        format: Format penyimpanan ("json" atau "pickle")
        function_map: Pemetaan nama fungsi ke implementasi fungsi (diperlukan untuk format JSON)
        
    Returns:
        Objek BaseAgent
    """
    if format == "json":
        # Muat dari JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Periksa apakah function_map diberikan
        if function_map is None:
            function_map = {}
        
        # Tentukan kelas agen
        agent_type = data.get("type", "BaseAgent")
        
        # Impor kelas agen
        try:
            module_name = f"simple_dgm.agents.{agent_type.lower()}"
            module = importlib.import_module(module_name)
            agent_class = getattr(module, agent_type)
        except (ImportError, AttributeError):
            # Default ke BaseAgent
            from simple_dgm.agents.base_agent import BaseAgent
            agent_class = BaseAgent
        
        # Buat agen
        return agent_class.from_dict(data, function_map)
    
    elif format == "pickle":
        # Muat dari pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def export_agent_code(agent: BaseAgent, filepath: str) -> None:
    """
    Ekspor kode agen.
    
    Args:
        agent: Agen yang akan diekspor
        filepath: Path file untuk menyimpan kode agen
    """
    from simple_dgm.utils.code_generation import generate_agent_code
    
    # Dapatkan informasi agen
    agent_type = agent.__class__.__name__
    
    # Buat spesifikasi agen
    spec = {
        "name": agent_type,
        "base_class": "BaseAgent",
        "description": f"Exported {agent_type}",
        "params": []
    }
    
    # Tambahkan parameter khusus
    if hasattr(agent, "code_style") and hasattr(agent, "preferred_language"):
        # CodingAgent
        spec["base_class"] = "CodingAgent"
        spec["params"].extend([
            {
                "name": "code_style",
                "type": "str",
                "default": agent.code_style,
                "description": "Gaya pengkodean"
            },
            {
                "name": "preferred_language",
                "type": "str",
                "default": agent.preferred_language,
                "description": "Bahasa pemrograman yang disukai"
            }
        ])
    
    elif hasattr(agent, "problem_types") and hasattr(agent, "max_iterations"):
        # ProblemSolvingAgent
        spec["base_class"] = "ProblemSolvingAgent"
        spec["params"].extend([
            {
                "name": "problem_types",
                "type": "List[str]",
                "default": agent.problem_types,
                "description": "Jenis masalah yang dapat ditangani"
            },
            {
                "name": "max_iterations",
                "type": "int",
                "default": agent.max_iterations,
                "description": "Jumlah maksimum iterasi untuk pemecahan masalah"
            },
            {
                "name": "timeout",
                "type": "float",
                "default": agent.timeout,
                "description": "Batas waktu untuk pemecahan masalah (detik)"
            }
        ])
    
    elif hasattr(agent, "agent_types") and hasattr(agent, "code_generation_capacity"):
        # MetaAgent
        spec["base_class"] = "MetaAgent"
        spec["params"].extend([
            {
                "name": "agent_types",
                "type": "List[str]",
                "default": agent.agent_types,
                "description": "Jenis agen yang dapat dimodifikasi"
            },
            {
                "name": "code_generation_capacity",
                "type": "float",
                "default": agent.code_generation_capacity,
                "description": "Kapasitas untuk menghasilkan kode (0.0 - 1.0)"
            }
        ])
    
    # Hasilkan kode agen
    code = generate_agent_code(spec)
    
    # Simpan kode ke file
    with open(filepath, 'w') as f:
        f.write(code)


def import_agent_from_code(filepath: str) -> Type[BaseAgent]:
    """
    Impor kelas agen dari file kode.
    
    Args:
        filepath: Path file kode agen
        
    Returns:
        Kelas agen
    """
    # Dapatkan nama modul dari path file
    module_name = os.path.basename(filepath).replace('.py', '')
    
    # Impor modul
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Cari kelas agen dalam modul
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, BaseAgent) and obj != BaseAgent:
            return obj
    
    # Jika tidak ditemukan, kembalikan None
    return None