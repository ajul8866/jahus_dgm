"""
Implementasi agen meta untuk Darwin-Gödel Machine.
"""

import random
import copy
import json
import importlib
import inspect
from typing import Dict, Any, List, Optional, Callable, Union, Tuple, Type

from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.agents.coding_agent import CodingAgent
from simple_dgm.agents.problem_solving_agent import ProblemSolvingAgent


class MetaAgent(BaseAgent):
    """
    Agen meta yang dapat memodifikasi dan mengembangkan agen lain.
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None, 
                 memory_capacity: int = 20,
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.1,
                 agent_types: Optional[List[str]] = None,
                 code_generation_capacity: float = 0.5):
        """
        Inisialisasi agen meta.
        
        Args:
            tools: Daftar alat yang tersedia untuk agen
            memory_capacity: Kapasitas memori agen
            learning_rate: Tingkat pembelajaran agen
            exploration_rate: Tingkat eksplorasi agen
            agent_types: Jenis agen yang dapat dimodifikasi
            code_generation_capacity: Kapasitas untuk menghasilkan kode (0.0 - 1.0)
        """
        super().__init__(tools, memory_capacity, learning_rate, exploration_rate)
        
        self.agent_types = agent_types or ["BaseAgent", "CodingAgent", "ProblemSolvingAgent"]
        self.code_generation_capacity = code_generation_capacity
        self.agent_templates = self._initialize_agent_templates()
        self.modification_history = []
        
        # Tambahkan alat khusus untuk agen meta
        self._add_meta_tools()
        
        # Tambahkan metadata khusus
        self.metadata.update({
            "type": "MetaAgent",
            "agent_types": self.agent_types,
            "code_generation_capacity": code_generation_capacity
        })
    
    def _initialize_agent_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Inisialisasi template untuk berbagai jenis agen.
        
        Returns:
            Dictionary template agen
        """
        templates = {}
        
        # Template untuk BaseAgent
        templates["BaseAgent"] = {
            "class": BaseAgent,
            "params": {
                "memory_capacity": 10,
                "learning_rate": 0.01,
                "exploration_rate": 0.1
            },
            "tools": []
        }
        
        # Template untuk CodingAgent
        templates["CodingAgent"] = {
            "class": CodingAgent,
            "params": {
                "memory_capacity": 10,
                "learning_rate": 0.01,
                "exploration_rate": 0.1,
                "code_style": "clean",
                "preferred_language": "python"
            },
            "tools": []
        }
        
        # Template untuk ProblemSolvingAgent
        templates["ProblemSolvingAgent"] = {
            "class": ProblemSolvingAgent,
            "params": {
                "memory_capacity": 10,
                "learning_rate": 0.01,
                "exploration_rate": 0.1,
                "problem_types": ["optimization", "search", "planning", "classification"],
                "max_iterations": 100,
                "timeout": 30.0
            },
            "tools": []
        }
        
        return templates
    
    def _add_meta_tools(self) -> None:
        """
        Tambahkan alat khusus untuk agen meta.
        """
        # Alat untuk membuat agen baru
        self.add_tool(Tool(
            name="create_agent",
            function=self._create_agent,
            description="Buat agen baru berdasarkan template"
        ))
        
        # Alat untuk memodifikasi agen
        self.add_tool(Tool(
            name="modify_agent",
            function=self._modify_agent,
            description="Modifikasi parameter dan alat agen"
        ))
        
        # Alat untuk menggabungkan agen
        self.add_tool(Tool(
            name="merge_agents",
            function=self._merge_agents,
            description="Gabungkan dua agen menjadi agen baru"
        ))
        
        # Alat untuk menganalisis agen
        self.add_tool(Tool(
            name="analyze_agent",
            function=self._analyze_agent,
            description="Analisis struktur dan kemampuan agen"
        ))
        
        # Alat untuk menghasilkan kode agen baru
        self.add_tool(Tool(
            name="generate_agent_code",
            function=self._generate_agent_code,
            description="Hasilkan kode untuk jenis agen baru"
        ))
    
    def _create_agent(self, agent_type: str, params: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """
        Buat agen baru berdasarkan template.
        
        Args:
            agent_type: Jenis agen yang akan dibuat
            params: Parameter untuk agen (opsional)
            
        Returns:
            Agen baru
        """
        if agent_type not in self.agent_templates:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        template = self.agent_templates[agent_type]
        agent_class = template["class"]
        
        # Gabungkan parameter template dengan parameter yang diberikan
        merged_params = template["params"].copy()
        if params:
            merged_params.update(params)
        
        # Buat agen baru
        agent = agent_class(**merged_params)
        
        # Tambahkan alat dari template
        for tool_data in template["tools"]:
            tool = Tool(
                name=tool_data["name"],
                function=tool_data["function"],
                description=tool_data["description"]
            )
            agent.add_tool(tool)
        
        return agent
    
    def _modify_agent(self, agent: BaseAgent, modifications: Dict[str, Any]) -> BaseAgent:
        """
        Modifikasi parameter dan alat agen.
        
        Args:
            agent: Agen yang akan dimodifikasi
            modifications: Modifikasi yang akan diterapkan
            
        Returns:
            Agen yang dimodifikasi
        """
        # Buat salinan agen
        modified_agent = copy.deepcopy(agent)
        
        # Terapkan modifikasi parameter
        for param, value in modifications.get("params", {}).items():
            if hasattr(modified_agent, param):
                setattr(modified_agent, param, value)
        
        # Terapkan modifikasi alat
        for tool_mod in modifications.get("tools", []):
            action = tool_mod.get("action")
            
            if action == "add":
                # Tambahkan alat baru
                tool_data = tool_mod.get("tool", {})
                tool = Tool(
                    name=tool_data.get("name", "unnamed_tool"),
                    function=tool_data.get("function", lambda x: None),
                    description=tool_data.get("description", "")
                )
                modified_agent.add_tool(tool)
            
            elif action == "remove":
                # Hapus alat
                tool_name = tool_mod.get("name")
                if tool_name:
                    modified_agent.remove_tool(tool_name)
            
            elif action == "modify":
                # Modifikasi alat yang ada
                tool_name = tool_mod.get("name")
                tool_data = tool_mod.get("tool", {})
                
                if tool_name:
                    # Hapus alat lama
                    modified_agent.remove_tool(tool_name)
                    
                    # Tambahkan alat yang dimodifikasi
                    tool = Tool(
                        name=tool_data.get("name", tool_name),
                        function=tool_data.get("function", lambda x: None),
                        description=tool_data.get("description", "")
                    )
                    modified_agent.add_tool(tool)
        
        # Catat modifikasi
        self.modification_history.append({
            "agent_type": agent.__class__.__name__,
            "modifications": modifications
        })
        
        return modified_agent
    
    def _merge_agents(self, agent1: BaseAgent, agent2: BaseAgent) -> BaseAgent:
        """
        Gabungkan dua agen menjadi agen baru.
        
        Args:
            agent1: Agen pertama
            agent2: Agen kedua
            
        Returns:
            Agen gabungan
        """
        # Tentukan kelas agen gabungan (gunakan kelas yang lebih spesifik)
        if isinstance(agent1, type(agent2)):
            merged_class = type(agent1)
        elif isinstance(agent2, type(agent1)):
            merged_class = type(agent2)
        else:
            # Default ke BaseAgent jika kelas berbeda
            merged_class = BaseAgent
        
        # Gabungkan parameter
        merged_params = {}
        
        # Parameter dasar untuk semua agen
        for param in ["memory_capacity", "learning_rate", "exploration_rate"]:
            if hasattr(agent1, param):
                merged_params[param] = getattr(agent1, param)
        
        # Parameter khusus berdasarkan kelas yang dipilih
        if merged_class == CodingAgent:
            # Parameter khusus untuk CodingAgent
            if isinstance(agent1, CodingAgent):
                merged_params["code_style"] = agent1.code_style
                merged_params["preferred_language"] = agent1.preferred_language
            elif isinstance(agent2, CodingAgent):
                merged_params["code_style"] = agent2.code_style
                merged_params["preferred_language"] = agent2.preferred_language
            else:
                merged_params["code_style"] = "clean"
                merged_params["preferred_language"] = "python"
                
        elif merged_class == ProblemSolvingAgent:
            # Parameter khusus untuk ProblemSolvingAgent
            if isinstance(agent1, ProblemSolvingAgent):
                merged_params["problem_types"] = agent1.problem_types
                merged_params["max_iterations"] = agent1.max_iterations
                merged_params["timeout"] = agent1.timeout
            elif isinstance(agent2, ProblemSolvingAgent):
                merged_params["problem_types"] = agent2.problem_types
                merged_params["max_iterations"] = agent2.max_iterations
                merged_params["timeout"] = agent2.timeout
            else:
                merged_params["problem_types"] = ["optimization", "search"]
                merged_params["max_iterations"] = 100
                merged_params["timeout"] = 30.0
        
        # Buat agen gabungan
        merged_agent = merged_class(**merged_params)
        
        # Gabungkan alat
        tool_names = set()
        
        # Tambahkan alat dari agent1
        for tool in agent1.tools:
            merged_agent.add_tool(tool)
            tool_names.add(tool.name)
        
        # Tambahkan alat dari agent2 (jika belum ada)
        for tool in agent2.tools:
            if tool.name not in tool_names:
                merged_agent.add_tool(tool)
        
        # Gabungkan metadata
        merged_agent.metadata.update({
            "merged_from": [agent1.__class__.__name__, agent2.__class__.__name__],
            "merged_at": None  # Akan diisi oleh pengguna
        })
        
        return merged_agent
    
    def _analyze_agent(self, agent: BaseAgent) -> Dict[str, Any]:
        """
        Analisis struktur dan kemampuan agen.
        
        Args:
            agent: Agen yang akan dianalisis
            
        Returns:
            Hasil analisis
        """
        analysis = {
            "type": agent.__class__.__name__,
            "tools": [tool.to_dict() for tool in agent.tools],
            "num_tools": len(agent.tools),
            "memory_capacity": agent.memory_capacity,
            "learning_rate": agent.learning_rate,
            "exploration_rate": agent.exploration_rate,
            "metadata": agent.metadata
        }
        
        # Analisis khusus berdasarkan jenis agen
        if isinstance(agent, CodingAgent):
            analysis.update({
                "code_style": agent.code_style,
                "preferred_language": agent.preferred_language,
                "capabilities": ["code_analysis", "bug_fixing", "code_optimization", "code_generation"]
            })
        
        elif isinstance(agent, ProblemSolvingAgent):
            analysis.update({
                "problem_types": agent.problem_types,
                "max_iterations": agent.max_iterations,
                "timeout": agent.timeout,
                "capabilities": agent.problem_types
            })
        
        elif isinstance(agent, MetaAgent):
            analysis.update({
                "agent_types": agent.agent_types,
                "code_generation_capacity": agent.code_generation_capacity,
                "capabilities": ["agent_creation", "agent_modification", "agent_merging", "agent_analysis"]
            })
        
        else:
            analysis.update({
                "capabilities": ["basic_problem_solving"]
            })
        
        return analysis
    
    def _generate_agent_code(self, spec: Dict[str, Any]) -> str:
        """
        Hasilkan kode untuk jenis agen baru.
        
        Args:
            spec: Spesifikasi agen baru
            
        Returns:
            Kode agen baru
        """
        agent_name = spec.get("name", "CustomAgent")
        base_class = spec.get("base_class", "BaseAgent")
        description = spec.get("description", "Custom agent for Darwin-Gödel Machine.")
        params = spec.get("params", [])
        methods = spec.get("methods", [])
        
        # Hasilkan kode
        code = f'"""\n{description}\n"""\n\n'
        code += "import random\n"
        code += "from typing import Dict, Any, List, Optional, Callable\n\n"
        code += f"from simple_dgm.agents.{base_class.lower()} import {base_class}\n"
        code += "from simple_dgm.agents.base_agent import Tool\n\n\n"
        
        # Definisi kelas
        code += f"class {agent_name}({base_class}):\n"
        code += f'    """\n    {description}\n    """\n\n'
        
        # Metode __init__
        code += "    def __init__(self"
        
        # Parameter __init__
        for param in params:
            param_name = param.get("name")
            param_type = param.get("type", "Any")
            param_default = param.get("default")
            
            if param_default is not None:
                if isinstance(param_default, str):
                    code += f', {param_name}: {param_type} = "{param_default}"'
                else:
                    code += f", {param_name}: {param_type} = {param_default}"
            else:
                code += f", {param_name}: {param_type}"
        
        # Parameter dasar
        code += ", tools: Optional[List[Tool]] = None"
        code += ", memory_capacity: int = 10"
        code += ", learning_rate: float = 0.01"
        code += ", exploration_rate: float = 0.1):\n"
        
        # Docstring __init__
        code += '        """\n'
        code += f"        Inisialisasi {agent_name}.\n\n"
        code += "        Args:\n"
        
        for param in params:
            param_name = param.get("name")
            param_desc = param.get("description", "")
            code += f"            {param_name}: {param_desc}\n"
        
        code += "            tools: Daftar alat yang tersedia untuk agen\n"
        code += "            memory_capacity: Kapasitas memori agen\n"
        code += "            learning_rate: Tingkat pembelajaran agen\n"
        code += "            exploration_rate: Tingkat eksplorasi agen\n"
        code += '        """\n'
        
        # Panggil __init__ kelas induk
        code += f"        super().__init__(tools, memory_capacity, learning_rate, exploration_rate)\n\n"
        
        # Inisialisasi parameter khusus
        for param in params:
            param_name = param.get("name")
            code += f"        self.{param_name} = {param_name}\n"
        
        # Tambahkan alat khusus
        code += "\n        # Tambahkan alat khusus\n"
        code += "        self._add_custom_tools()\n\n"
        
        # Tambahkan metadata khusus
        code += "        # Tambahkan metadata khusus\n"
        code += "        self.metadata.update({\n"
        code += f'            "type": "{agent_name}"'
        
        for param in params:
            param_name = param.get("name")
            code += f',\n            "{param_name}": self.{param_name}'
        
        code += "\n        })\n\n"
        
        # Metode _add_custom_tools
        code += "    def _add_custom_tools(self) -> None:\n"
        code += '        """\n'
        code += "        Tambahkan alat khusus untuk agen.\n"
        code += '        """\n'
        code += "        # Tambahkan alat khusus di sini\n"
        code += "        pass\n\n"
        
        # Tambahkan metode khusus
        for method in methods:
            method_name = method.get("name")
            method_params = method.get("params", [])
            method_return = method.get("return", "Any")
            method_body = method.get("body", ["pass"])
            method_desc = method.get("description", "")
            
            # Definisi metode
            code += f"    def {method_name}(self"
            
            # Parameter metode
            for param in method_params:
                param_name = param.get("name")
                param_type = param.get("type", "Any")
                param_default = param.get("default")
                
                if param_default is not None:
                    if isinstance(param_default, str):
                        code += f', {param_name}: {param_type} = "{param_default}"'
                    else:
                        code += f", {param_name}: {param_type} = {param_default}"
                else:
                    code += f", {param_name}: {param_type}"
            
            code += f") -> {method_return}:\n"
            
            # Docstring metode
            code += '        """\n'
            code += f"        {method_desc}\n\n"
            
            if method_params:
                code += "        Args:\n"
                for param in method_params:
                    param_name = param.get("name")
                    param_desc = param.get("description", "")
                    code += f"            {param_name}: {param_desc}\n"
            
            if method_return != "None":
                code += "\n        Returns:\n"
                code += f"            {method.get('return_desc', '')}\n"
            
            code += '        """\n'
            
            # Badan metode
            for line in method_body:
                code += f"        {line}\n"
            
            code += "\n"
        
        # Override metode mutate
        code += "    def mutate(self) -> None:\n"
        code += '        """\n'
        code += f"        Mutasi {agent_name}.\n"
        code += '        """\n'
        code += "        super().mutate()  # Panggil mutasi dasar\n\n"
        
        # Mutasi parameter khusus
        code += "        # Mutasi parameter khusus\n"
        for param in params:
            param_name = param.get("name")
            param_type = param.get("type", "Any")
            
            if param_type == "float":
                code += f"        self.{param_name} *= random.uniform(0.8, 1.2)\n"
            elif param_type == "int":
                code += f"        self.{param_name} = int(self.{param_name} * random.uniform(0.8, 1.2))\n"
            elif param_type == "str" and param.get("options"):
                options = param.get("options")
                code += f"        if random.random() < 0.3:\n"
                code += f"            self.{param_name} = random.choice({options})\n"
        
        code += "\n        # Update metadata\n"
        code += "        self.metadata.update({\n"
        
        for param in params:
            param_name = param.get("name")
            code += f'            "{param_name}": self.{param_name}'
            if param != params[-1]:
                code += ","
            code += "\n"
        
        code += "        })\n\n"
        
        # Override metode to_dict
        code += "    def to_dict(self) -> Dict[str, Any]:\n"
        code += '        """\n'
        code += f"        Konversi {agent_name} ke dictionary.\n"
        code += '        """\n'
        code += "        data = super().to_dict()\n"
        code += "        data.update({\n"
        
        for param in params:
            param_name = param.get("name")
            code += f'            "{param_name}": self.{param_name}'
            if param != params[-1]:
                code += ","
            code += "\n"
        
        code += "        })\n"
        code += "        return data\n\n"
        
        # Override metode from_dict
        code += "    @classmethod\n"
        code += "    def from_dict(cls, data: Dict[str, Any], function_map: Dict[str, Callable]) -> 'BaseAgent':\n"
        code += '        """\n'
        code += f"        Buat {agent_name} dari dictionary.\n"
        code += '        """\n'
        code += "        agent = super().from_dict(data, function_map)\n"
        
        for param in params:
            param_name = param.get("name")
            code += f'        agent.{param_name} = data["{param_name}"]\n'
        
        code += "        return agent\n"
        
        return code
    
    def solve(self, problem: Dict[str, Any]) -> Any:
        """
        Selesaikan masalah meta-agen.
        
        Args:
            problem: Masalah yang akan diselesaikan
            
        Returns:
            Solusi untuk masalah
        """
        problem_type = problem.get("type", "")
        
        if problem_type == "create_agent":
            # Buat agen baru
            agent_type = problem.get("agent_type", "BaseAgent")
            params = problem.get("params", {})
            return self._create_agent(agent_type, params)
        
        elif problem_type == "modify_agent":
            # Modifikasi agen
            agent = problem.get("agent")
            modifications = problem.get("modifications", {})
            return self._modify_agent(agent, modifications)
        
        elif problem_type == "merge_agents":
            # Gabungkan agen
            agent1 = problem.get("agent1")
            agent2 = problem.get("agent2")
            return self._merge_agents(agent1, agent2)
        
        elif problem_type == "analyze_agent":
            # Analisis agen
            agent = problem.get("agent")
            return self._analyze_agent(agent)
        
        elif problem_type == "generate_agent_code":
            # Hasilkan kode agen baru
            spec = problem.get("spec", {})
            return self._generate_agent_code(spec)
        
        else:
            # Gunakan implementasi dasar
            return super().solve(problem)
    
    def mutate(self) -> None:
        """
        Mutasi agen meta.
        """
        super().mutate()  # Panggil mutasi dasar
        
        # Mutasi parameter khusus
        self.code_generation_capacity *= random.uniform(0.8, 1.2)
        
        # Batasi nilai parameter
        self.code_generation_capacity = max(0.1, min(1.0, self.code_generation_capacity))
        
        # Update metadata
        self.metadata.update({
            "code_generation_capacity": self.code_generation_capacity
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konversi agen meta ke dictionary.
        """
        data = super().to_dict()
        data.update({
            "agent_types": self.agent_types,
            "code_generation_capacity": self.code_generation_capacity,
            "modification_history": self.modification_history
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], function_map: Dict[str, Callable]) -> 'MetaAgent':
        """
        Buat agen meta dari dictionary.
        """
        agent = super().from_dict(data, function_map)
        agent.agent_types = data["agent_types"]
        agent.code_generation_capacity = data["code_generation_capacity"]
        agent.modification_history = data.get("modification_history", [])
        agent.agent_templates = agent._initialize_agent_templates()
        return agent