"""
Integrasi LLM untuk Darwin-Gödel Machine.

Modul ini berisi implementasi integrasi dengan model bahasa besar (LLM)
yang digunakan oleh DGM untuk pembuatan kode, pemecahan masalah,
ekstraksi pengetahuan, dan modifikasi diri.
"""

import os
import json
import re
import time
import requests
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

from simple_dgm.agents.base_agent import BaseAgent, Tool


class LLMInterface:
    """
    Antarmuka LLM untuk DGM.
    
    Antarmuka ini menyediakan akses ke model bahasa besar (LLM).
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4",
                base_url: str = "https://api.openai.com/v1"):
        """
        Inisialisasi antarmuka LLM.
        
        Args:
            api_key: Kunci API (opsional, default ke OPENAI_API_KEY)
            model: Model yang akan digunakan
            base_url: URL dasar API
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self.history = []
    
    def generate(self, prompt: str, max_tokens: int = 1000, 
                temperature: float = 0.7, stop: Optional[List[str]] = None) -> str:
        """
        Hasilkan teks dari LLM.
        
        Args:
            prompt: Prompt untuk LLM
            max_tokens: Jumlah maksimum token yang akan dihasilkan
            temperature: Temperatur sampling
            stop: Daftar string yang menghentikan generasi
            
        Returns:
            Teks yang dihasilkan
        """
        if not self.api_key:
            raise ValueError("API key is required")
        
        # Buat permintaan
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if stop:
            data["stop"] = stop
        
        # Kirim permintaan
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            # Parse respons
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            # Catat riwayat
            self.history.append({
                "prompt": prompt,
                "response": generated_text,
                "timestamp": time.time()
            })
            
            return generated_text
        
        except requests.exceptions.RequestException as e:
            # Tangani kesalahan
            error_msg = f"Error generating text: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                error_msg += f" - {e.response.text}"
            
            raise RuntimeError(error_msg)
    
    def generate_with_history(self, messages: List[Dict[str, str]], 
                             max_tokens: int = 1000, temperature: float = 0.7,
                             stop: Optional[List[str]] = None) -> str:
        """
        Hasilkan teks dari LLM dengan riwayat percakapan.
        
        Args:
            messages: Daftar pesan (dicts dengan kunci "role" dan "content")
            max_tokens: Jumlah maksimum token yang akan dihasilkan
            temperature: Temperatur sampling
            stop: Daftar string yang menghentikan generasi
            
        Returns:
            Teks yang dihasilkan
        """
        if not self.api_key:
            raise ValueError("API key is required")
        
        # Buat permintaan
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if stop:
            data["stop"] = stop
        
        # Kirim permintaan
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            # Parse respons
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            # Catat riwayat
            self.history.append({
                "messages": messages,
                "response": generated_text,
                "timestamp": time.time()
            })
            
            return generated_text
        
        except requests.exceptions.RequestException as e:
            # Tangani kesalahan
            error_msg = f"Error generating text: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                error_msg += f" - {e.response.text}"
            
            raise RuntimeError(error_msg)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Dapatkan embedding untuk teks.
        
        Args:
            text: Teks yang akan di-embed
            
        Returns:
            Vektor embedding
        """
        if not self.api_key:
            raise ValueError("API key is required")
        
        # Buat permintaan
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "text-embedding-ada-002",
            "input": text
        }
        
        # Kirim permintaan
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            # Parse respons
            result = response.json()
            embedding = result["data"][0]["embedding"]
            
            return embedding
        
        except requests.exceptions.RequestException as e:
            # Tangani kesalahan
            error_msg = f"Error getting embedding: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                error_msg += f" - {e.response.text}"
            
            raise RuntimeError(error_msg)


class CodeGeneration:
    """
    Pembuatan kode untuk DGM.
    
    Kelas ini menggunakan LLM untuk menghasilkan kode.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Inisialisasi pembuatan kode.
        
        Args:
            llm_interface: Antarmuka LLM
        """
        self.llm = llm_interface
    
    def generate_function(self, function_name: str, description: str, 
                         parameters: List[Dict[str, str]], 
                         return_type: str) -> str:
        """
        Hasilkan kode fungsi.
        
        Args:
            function_name: Nama fungsi
            description: Deskripsi fungsi
            parameters: Daftar parameter (dicts dengan kunci "name", "type", dan "description")
            return_type: Tipe return
            
        Returns:
            Kode fungsi
        """
        # Buat prompt
        prompt = f"Generate a Python function named '{function_name}' with the following description:\n"
        prompt += f"{description}\n\n"
        
        prompt += "Parameters:\n"
        for param in parameters:
            prompt += f"- {param['name']} ({param['type']}): {param['description']}\n"
        
        prompt += f"\nReturn type: {return_type}\n\n"
        
        prompt += "Please provide only the function code without any explanation or additional text."
        
        # Hasilkan kode
        code = self.llm.generate(prompt, max_tokens=1000, temperature=0.2)
        
        # Bersihkan kode
        code = self._clean_code(code)
        
        return code
    
    def generate_class(self, class_name: str, description: str, 
                      base_classes: List[str], methods: List[Dict[str, Any]]) -> str:
        """
        Hasilkan kode kelas.
        
        Args:
            class_name: Nama kelas
            description: Deskripsi kelas
            base_classes: Daftar kelas dasar
            methods: Daftar metode (dicts dengan kunci "name", "description", "parameters", dan "return_type")
            
        Returns:
            Kode kelas
        """
        # Buat prompt
        prompt = f"Generate a Python class named '{class_name}' with the following description:\n"
        prompt += f"{description}\n\n"
        
        if base_classes:
            prompt += f"Base classes: {', '.join(base_classes)}\n\n"
        
        prompt += "Methods:\n"
        for method in methods:
            prompt += f"- {method['name']}: {method['description']}\n"
            
            if "parameters" in method:
                prompt += "  Parameters:\n"
                for param in method["parameters"]:
                    prompt += f"  - {param['name']} ({param['type']}): {param['description']}\n"
            
            if "return_type" in method:
                prompt += f"  Return type: {method['return_type']}\n"
            
            prompt += "\n"
        
        prompt += "Please provide only the class code without any explanation or additional text."
        
        # Hasilkan kode
        code = self.llm.generate(prompt, max_tokens=2000, temperature=0.2)
        
        # Bersihkan kode
        code = self._clean_code(code)
        
        return code
    
    def generate_module(self, module_name: str, description: str, 
                       imports: List[str], components: List[Dict[str, Any]]) -> str:
        """
        Hasilkan kode modul.
        
        Args:
            module_name: Nama modul
            description: Deskripsi modul
            imports: Daftar impor
            components: Daftar komponen (dicts dengan kunci "type", "name", dan "description")
            
        Returns:
            Kode modul
        """
        # Buat prompt
        prompt = f"Generate a Python module named '{module_name}' with the following description:\n"
        prompt += f"{description}\n\n"
        
        if imports:
            prompt += "Imports:\n"
            for imp in imports:
                prompt += f"- {imp}\n"
            prompt += "\n"
        
        prompt += "Components:\n"
        for component in components:
            comp_type = component["type"]
            comp_name = component["name"]
            comp_desc = component["description"]
            
            prompt += f"- {comp_type} '{comp_name}': {comp_desc}\n"
        
        prompt += "\nPlease provide only the module code without any explanation or additional text."
        
        # Hasilkan kode
        code = self.llm.generate(prompt, max_tokens=3000, temperature=0.2)
        
        # Bersihkan kode
        code = self._clean_code(code)
        
        return code
    
    def generate_agent(self, agent_name: str, base_class: str, 
                      description: str, capabilities: List[str]) -> str:
        """
        Hasilkan kode agen.
        
        Args:
            agent_name: Nama agen
            base_class: Kelas dasar
            description: Deskripsi agen
            capabilities: Daftar kemampuan
            
        Returns:
            Kode agen
        """
        # Buat prompt
        prompt = f"Generate a Python class for a Darwin-Gödel Machine agent named '{agent_name}' with the following description:\n"
        prompt += f"{description}\n\n"
        
        prompt += f"Base class: {base_class}\n\n"
        
        prompt += "Capabilities:\n"
        for capability in capabilities:
            prompt += f"- {capability}\n"
        
        prompt += "\nThe agent should have the following methods:\n"
        prompt += "- __init__: Initialize the agent\n"
        prompt += "- solve: Solve a problem\n"
        prompt += "- mutate: Mutate the agent\n"
        prompt += "- to_dict: Convert the agent to a dictionary\n"
        prompt += "- from_dict (class method): Create an agent from a dictionary\n"
        
        prompt += "\nPlease provide only the agent code without any explanation or additional text."
        
        # Hasilkan kode
        code = self.llm.generate(prompt, max_tokens=3000, temperature=0.2)
        
        # Bersihkan kode
        code = self._clean_code(code)
        
        return code
    
    def _clean_code(self, code: str) -> str:
        """
        Bersihkan kode yang dihasilkan.
        
        Args:
            code: Kode yang akan dibersihkan
            
        Returns:
            Kode yang dibersihkan
        """
        # Hapus blok kode markdown
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Hapus komentar penjelasan di awal
        code = re.sub(r'^#.*?\n\n', '', code, flags=re.DOTALL)
        
        # Hapus whitespace di awal dan akhir
        code = code.strip()
        
        return code


class ProblemSolving:
    """
    Pemecahan masalah untuk DGM.
    
    Kelas ini menggunakan LLM untuk memecahkan masalah.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Inisialisasi pemecahan masalah.
        
        Args:
            llm_interface: Antarmuka LLM
        """
        self.llm = llm_interface
    
    def solve_problem(self, problem: str, context: Optional[str] = None) -> str:
        """
        Pecahkan masalah.
        
        Args:
            problem: Masalah yang akan dipecahkan
            context: Konteks masalah (opsional)
            
        Returns:
            Solusi masalah
        """
        # Buat prompt
        prompt = "Solve the following problem:\n\n"
        prompt += f"{problem}\n\n"
        
        if context:
            prompt += f"Context:\n{context}\n\n"
        
        prompt += "Please provide a detailed solution."
        
        # Hasilkan solusi
        solution = self.llm.generate(prompt, max_tokens=2000, temperature=0.3)
        
        return solution
    
    def analyze_problem(self, problem: str) -> Dict[str, Any]:
        """
        Analisis masalah.
        
        Args:
            problem: Masalah yang akan dianalisis
            
        Returns:
            Hasil analisis
        """
        # Buat prompt
        prompt = "Analyze the following problem and provide a structured analysis in JSON format:\n\n"
        prompt += f"{problem}\n\n"
        
        prompt += "Your analysis should include:\n"
        prompt += "- problem_type: The type of problem\n"
        prompt += "- complexity: The complexity of the problem (easy, medium, hard)\n"
        prompt += "- key_concepts: List of key concepts involved\n"
        prompt += "- approach: Suggested approach to solve the problem\n"
        prompt += "- potential_challenges: List of potential challenges\n"
        
        prompt += "\nProvide your analysis in valid JSON format."
        
        # Hasilkan analisis
        analysis_text = self.llm.generate(prompt, max_tokens=1500, temperature=0.3)
        
        # Parse JSON
        try:
            # Ekstrak JSON dari teks
            json_match = re.search(r'```json\s*(.*?)\s*```', analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json_match.group(1)
            else:
                analysis_json = analysis_text
            
            analysis = json.loads(analysis_json)
            return analysis
        
        except json.JSONDecodeError:
            # Jika parsing gagal, kembalikan teks asli
            return {"raw_analysis": analysis_text}
    
    def decompose_problem(self, problem: str) -> List[Dict[str, Any]]:
        """
        Dekomposisi masalah menjadi sub-masalah.
        
        Args:
            problem: Masalah yang akan didekomposisi
            
        Returns:
            Daftar sub-masalah
        """
        # Buat prompt
        prompt = "Decompose the following problem into smaller sub-problems in JSON format:\n\n"
        prompt += f"{problem}\n\n"
        
        prompt += "For each sub-problem, provide:\n"
        prompt += "- id: A unique identifier for the sub-problem\n"
        prompt += "- description: A description of the sub-problem\n"
        prompt += "- dependencies: List of sub-problem IDs that this sub-problem depends on\n"
        
        prompt += "\nProvide your decomposition in valid JSON format as an array of sub-problems."
        
        # Hasilkan dekomposisi
        decomposition_text = self.llm.generate(prompt, max_tokens=2000, temperature=0.3)
        
        # Parse JSON
        try:
            # Ekstrak JSON dari teks
            json_match = re.search(r'```json\s*(.*?)\s*```', decomposition_text, re.DOTALL)
            if json_match:
                decomposition_json = json_match.group(1)
            else:
                decomposition_json = decomposition_text
            
            decomposition = json.loads(decomposition_json)
            return decomposition
        
        except json.JSONDecodeError:
            # Jika parsing gagal, kembalikan daftar dengan teks asli
            return [{"raw_decomposition": decomposition_text}]
    
    def evaluate_solution(self, problem: str, solution: str) -> Dict[str, Any]:
        """
        Evaluasi solusi.
        
        Args:
            problem: Masalah
            solution: Solusi yang akan dievaluasi
            
        Returns:
            Hasil evaluasi
        """
        # Buat prompt
        prompt = "Evaluate the following solution to the given problem and provide a structured evaluation in JSON format:\n\n"
        prompt += f"Problem:\n{problem}\n\n"
        prompt += f"Solution:\n{solution}\n\n"
        
        prompt += "Your evaluation should include:\n"
        prompt += "- correctness: Whether the solution is correct (0.0 to 1.0)\n"
        prompt += "- efficiency: The efficiency of the solution (0.0 to 1.0)\n"
        prompt += "- clarity: The clarity of the solution (0.0 to 1.0)\n"
        prompt += "- overall_score: The overall score of the solution (0.0 to 1.0)\n"
        prompt += "- strengths: List of strengths of the solution\n"
        prompt += "- weaknesses: List of weaknesses of the solution\n"
        prompt += "- suggestions: List of suggestions for improvement\n"
        
        prompt += "\nProvide your evaluation in valid JSON format."
        
        # Hasilkan evaluasi
        evaluation_text = self.llm.generate(prompt, max_tokens=1500, temperature=0.3)
        
        # Parse JSON
        try:
            # Ekstrak JSON dari teks
            json_match = re.search(r'```json\s*(.*?)\s*```', evaluation_text, re.DOTALL)
            if json_match:
                evaluation_json = json_match.group(1)
            else:
                evaluation_json = evaluation_text
            
            evaluation = json.loads(evaluation_json)
            return evaluation
        
        except json.JSONDecodeError:
            # Jika parsing gagal, kembalikan teks asli
            return {"raw_evaluation": evaluation_text}


class KnowledgeExtraction:
    """
    Ekstraksi pengetahuan untuk DGM.
    
    Kelas ini menggunakan LLM untuk mengekstrak pengetahuan.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Inisialisasi ekstraksi pengetahuan.
        
        Args:
            llm_interface: Antarmuka LLM
        """
        self.llm = llm_interface
    
    def extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """
        Ekstrak konsep dari teks.
        
        Args:
            text: Teks yang akan diekstrak
            
        Returns:
            Daftar konsep
        """
        # Buat prompt
        prompt = "Extract key concepts from the following text in JSON format:\n\n"
        prompt += f"{text}\n\n"
        
        prompt += "For each concept, provide:\n"
        prompt += "- name: The name of the concept\n"
        prompt += "- description: A brief description of the concept\n"
        prompt += "- category: The category of the concept\n"
        
        prompt += "\nProvide your extraction in valid JSON format as an array of concepts."
        
        # Hasilkan ekstraksi
        extraction_text = self.llm.generate(prompt, max_tokens=2000, temperature=0.3)
        
        # Parse JSON
        try:
            # Ekstrak JSON dari teks
            json_match = re.search(r'```json\s*(.*?)\s*```', extraction_text, re.DOTALL)
            if json_match:
                extraction_json = json_match.group(1)
            else:
                extraction_json = extraction_text
            
            extraction = json.loads(extraction_json)
            return extraction
        
        except json.JSONDecodeError:
            # Jika parsing gagal, kembalikan daftar dengan teks asli
            return [{"raw_extraction": extraction_text}]
    
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Ekstrak hubungan dari teks.
        
        Args:
            text: Teks yang akan diekstrak
            
        Returns:
            Daftar hubungan
        """
        # Buat prompt
        prompt = "Extract relationships between concepts from the following text in JSON format:\n\n"
        prompt += f"{text}\n\n"
        
        prompt += "For each relationship, provide:\n"
        prompt += "- source: The source concept\n"
        prompt += "- target: The target concept\n"
        prompt += "- type: The type of relationship\n"
        prompt += "- description: A brief description of the relationship\n"
        
        prompt += "\nProvide your extraction in valid JSON format as an array of relationships."
        
        # Hasilkan ekstraksi
        extraction_text = self.llm.generate(prompt, max_tokens=2000, temperature=0.3)
        
        # Parse JSON
        try:
            # Ekstrak JSON dari teks
            json_match = re.search(r'```json\s*(.*?)\s*```', extraction_text, re.DOTALL)
            if json_match:
                extraction_json = json_match.group(1)
            else:
                extraction_json = extraction_text
            
            extraction = json.loads(extraction_json)
            return extraction
        
        except json.JSONDecodeError:
            # Jika parsing gagal, kembalikan daftar dengan teks asli
            return [{"raw_extraction": extraction_text}]
    
    def extract_knowledge_graph(self, text: str) -> Dict[str, Any]:
        """
        Ekstrak grafik pengetahuan dari teks.
        
        Args:
            text: Teks yang akan diekstrak
            
        Returns:
            Grafik pengetahuan
        """
        # Buat prompt
        prompt = "Extract a knowledge graph from the following text in JSON format:\n\n"
        prompt += f"{text}\n\n"
        
        prompt += "The knowledge graph should include:\n"
        prompt += "- nodes: Array of nodes, each with 'id', 'label', and 'type'\n"
        prompt += "- edges: Array of edges, each with 'source', 'target', 'label', and 'type'\n"
        
        prompt += "\nProvide your extraction in valid JSON format."
        
        # Hasilkan ekstraksi
        extraction_text = self.llm.generate(prompt, max_tokens=3000, temperature=0.3)
        
        # Parse JSON
        try:
            # Ekstrak JSON dari teks
            json_match = re.search(r'```json\s*(.*?)\s*```', extraction_text, re.DOTALL)
            if json_match:
                extraction_json = json_match.group(1)
            else:
                extraction_json = extraction_text
            
            extraction = json.loads(extraction_json)
            return extraction
        
        except json.JSONDecodeError:
            # Jika parsing gagal, kembalikan teks asli
            return {"raw_extraction": extraction_text}
    
    def extract_code_patterns(self, code: str) -> List[Dict[str, Any]]:
        """
        Ekstrak pola kode dari kode.
        
        Args:
            code: Kode yang akan diekstrak
            
        Returns:
            Daftar pola kode
        """
        # Buat prompt
        prompt = "Extract code patterns from the following code in JSON format:\n\n"
        prompt += f"```python\n{code}\n```\n\n"
        
        prompt += "For each pattern, provide:\n"
        prompt += "- name: The name of the pattern\n"
        prompt += "- description: A brief description of the pattern\n"
        prompt += "- category: The category of the pattern\n"
        prompt += "- code_snippet: A snippet of code that demonstrates the pattern\n"
        
        prompt += "\nProvide your extraction in valid JSON format as an array of patterns."
        
        # Hasilkan ekstraksi
        extraction_text = self.llm.generate(prompt, max_tokens=2000, temperature=0.3)
        
        # Parse JSON
        try:
            # Ekstrak JSON dari teks
            json_match = re.search(r'```json\s*(.*?)\s*```', extraction_text, re.DOTALL)
            if json_match:
                extraction_json = json_match.group(1)
            else:
                extraction_json = extraction_text
            
            extraction = json.loads(extraction_json)
            return extraction
        
        except json.JSONDecodeError:
            # Jika parsing gagal, kembalikan daftar dengan teks asli
            return [{"raw_extraction": extraction_text}]


class SelfModification:
    """
    Modifikasi diri untuk DGM.
    
    Kelas ini menggunakan LLM untuk memodifikasi kode DGM.
    """
    
    def __init__(self, llm_interface: LLMInterface):
        """
        Inisialisasi modifikasi diri.
        
        Args:
            llm_interface: Antarmuka LLM
        """
        self.llm = llm_interface
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Analisis kode.
        
        Args:
            code: Kode yang akan dianalisis
            
        Returns:
            Hasil analisis
        """
        # Buat prompt
        prompt = "Analyze the following code and provide a structured analysis in JSON format:\n\n"
        prompt += f"```python\n{code}\n```\n\n"
        
        prompt += "Your analysis should include:\n"
        prompt += "- complexity: The complexity of the code (low, medium, high)\n"
        prompt += "- quality: The quality of the code (low, medium, high)\n"
        prompt += "- issues: List of issues in the code\n"
        prompt += "- strengths: List of strengths of the code\n"
        prompt += "- improvement_suggestions: List of suggestions for improvement\n"
        
        prompt += "\nProvide your analysis in valid JSON format."
        
        # Hasilkan analisis
        analysis_text = self.llm.generate(prompt, max_tokens=2000, temperature=0.3)
        
        # Parse JSON
        try:
            # Ekstrak JSON dari teks
            json_match = re.search(r'```json\s*(.*?)\s*```', analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json_match.group(1)
            else:
                analysis_json = analysis_text
            
            analysis = json.loads(analysis_json)
            return analysis
        
        except json.JSONDecodeError:
            # Jika parsing gagal, kembalikan teks asli
            return {"raw_analysis": analysis_text}
    
    def improve_code(self, code: str, issues: Optional[List[str]] = None) -> str:
        """
        Tingkatkan kode.
        
        Args:
            code: Kode yang akan ditingkatkan
            issues: Daftar masalah yang akan diperbaiki (opsional)
            
        Returns:
            Kode yang ditingkatkan
        """
        # Buat prompt
        prompt = "Improve the following code:\n\n"
        prompt += f"```python\n{code}\n```\n\n"
        
        if issues:
            prompt += "Focus on fixing these issues:\n"
            for issue in issues:
                prompt += f"- {issue}\n"
            prompt += "\n"
        
        prompt += "Please provide only the improved code without any explanation or additional text."
        
        # Hasilkan kode yang ditingkatkan
        improved_code = self.llm.generate(prompt, max_tokens=3000, temperature=0.2)
        
        # Bersihkan kode
        improved_code = self._clean_code(improved_code)
        
        return improved_code
    
    def add_feature(self, code: str, feature_description: str) -> str:
        """
        Tambahkan fitur ke kode.
        
        Args:
            code: Kode yang akan ditambahkan fitur
            feature_description: Deskripsi fitur
            
        Returns:
            Kode dengan fitur baru
        """
        # Buat prompt
        prompt = "Add the following feature to the code:\n\n"
        prompt += f"Feature: {feature_description}\n\n"
        prompt += f"Code:\n```python\n{code}\n```\n\n"
        
        prompt += "Please provide only the modified code with the new feature without any explanation or additional text."
        
        # Hasilkan kode dengan fitur baru
        modified_code = self.llm.generate(prompt, max_tokens=3000, temperature=0.2)
        
        # Bersihkan kode
        modified_code = self._clean_code(modified_code)
        
        return modified_code
    
    def refactor_code(self, code: str, refactoring_type: str) -> str:
        """
        Refaktor kode.
        
        Args:
            code: Kode yang akan direfaktor
            refactoring_type: Tipe refaktor
            
        Returns:
            Kode yang direfaktor
        """
        # Buat prompt
        prompt = f"Refactor the following code using {refactoring_type}:\n\n"
        prompt += f"```python\n{code}\n```\n\n"
        
        prompt += "Please provide only the refactored code without any explanation or additional text."
        
        # Hasilkan kode yang direfaktor
        refactored_code = self.llm.generate(prompt, max_tokens=3000, temperature=0.2)
        
        # Bersihkan kode
        refactored_code = self._clean_code(refactored_code)
        
        return refactored_code
    
    def _clean_code(self, code: str) -> str:
        """
        Bersihkan kode yang dihasilkan.
        
        Args:
            code: Kode yang akan dibersihkan
            
        Returns:
            Kode yang dibersihkan
        """
        # Hapus blok kode markdown
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Hapus komentar penjelasan di awal
        code = re.sub(r'^#.*?\n\n', '', code, flags=re.DOTALL)
        
        # Hapus whitespace di awal dan akhir
        code = code.strip()
        
        return code