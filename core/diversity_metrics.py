"""
Metrik keragaman untuk Darwin-Gödel Machine.

Modul ini berisi implementasi berbagai metrik keragaman yang digunakan oleh DGM
untuk mengukur keragaman populasi agen selama proses evolusi.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar, Generic, Union, Set

from simple_dgm.agents.base_agent import BaseAgent

# Tipe generik untuk individu dalam populasi
T = TypeVar('T', bound=BaseAgent)


class DiversityMetric(Generic[T]):
    """
    Kelas dasar untuk metrik keragaman.
    
    Metrik keragaman mengukur keragaman populasi individu.
    """
    
    def __init__(self):
        """
        Inisialisasi metrik keragaman.
        """
        pass
    
    def measure(self, population: List[T]) -> float:
        """
        Ukur keragaman populasi.
        
        Args:
            population: Populasi individu
            
        Returns:
            Nilai keragaman
        """
        raise NotImplementedError("Subclass must implement abstract method")


class BehavioralDiversity(DiversityMetric[T]):
    """
    Metrik keragaman perilaku.
    
    Metrik ini mengukur keragaman perilaku individu dalam populasi.
    """
    
    def __init__(self, behavior_descriptor: Callable[[T], List[float]],
                distance_metric: str = "euclidean"):
        """
        Inisialisasi metrik keragaman perilaku.
        
        Args:
            behavior_descriptor: Fungsi yang mendeskripsikan perilaku individu
            distance_metric: Metrik jarak ("euclidean", "manhattan", "cosine")
        """
        super().__init__()
        self.behavior_descriptor = behavior_descriptor
        self.distance_metric = distance_metric
    
    def measure(self, population: List[T]) -> float:
        """
        Ukur keragaman perilaku populasi.
        
        Args:
            population: Populasi individu
            
        Returns:
            Nilai keragaman
        """
        if len(population) <= 1:
            return 0.0
        
        # Hitung deskriptor perilaku untuk setiap individu
        descriptors = [self.behavior_descriptor(ind) for ind in population]
        
        # Hitung matriks jarak
        n = len(descriptors)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Hitung jarak berdasarkan metrik
                if self.distance_metric == "euclidean":
                    dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(descriptors[i], descriptors[j])))
                elif self.distance_metric == "manhattan":
                    dist = sum(abs(a - b) for a, b in zip(descriptors[i], descriptors[j]))
                elif self.distance_metric == "cosine":
                    dot = sum(a * b for a, b in zip(descriptors[i], descriptors[j]))
                    norm_i = np.sqrt(sum(a ** 2 for a in descriptors[i]))
                    norm_j = np.sqrt(sum(b ** 2 for b in descriptors[j]))
                    dist = 1.0 - (dot / (norm_i * norm_j)) if norm_i > 0 and norm_j > 0 else 1.0
                else:
                    # Default ke Euclidean
                    dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(descriptors[i], descriptors[j])))
                
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Hitung keragaman sebagai jarak rata-rata
        return np.mean(distances)


class StructuralDiversity(DiversityMetric[T]):
    """
    Metrik keragaman struktural.
    
    Metrik ini mengukur keragaman struktur individu dalam populasi.
    """
    
    def __init__(self, attribute_extractor: Optional[Callable[[T], Dict[str, Any]]] = None):
        """
        Inisialisasi metrik keragaman struktural.
        
        Args:
            attribute_extractor: Fungsi yang mengekstrak atribut individu
        """
        super().__init__()
        self.attribute_extractor = attribute_extractor
    
    def _default_attribute_extractor(self, individual: T) -> Dict[str, Any]:
        """
        Ekstrak atribut default dari individu.
        
        Args:
            individual: Individu
            
        Returns:
            Atribut individu
        """
        attributes = {}
        
        # Ekstrak atribut dasar
        for attr in ["learning_rate", "exploration_rate", "memory_capacity"]:
            if hasattr(individual, attr):
                attributes[attr] = getattr(individual, attr)
        
        # Ekstrak atribut khusus berdasarkan jenis agen
        if hasattr(individual, "__class__") and hasattr(individual.__class__, "__name__"):
            class_name = individual.__class__.__name__
            attributes["class"] = class_name
            
            if class_name == "CodingAgent":
                for attr in ["code_style", "preferred_language"]:
                    if hasattr(individual, attr):
                        attributes[attr] = getattr(individual, attr)
            
            elif class_name == "ProblemSolvingAgent":
                for attr in ["problem_types", "max_iterations", "timeout"]:
                    if hasattr(individual, attr):
                        value = getattr(individual, attr)
                        if isinstance(value, list):
                            attributes[attr] = tuple(value)  # Konversi list ke tuple agar hashable
                        else:
                            attributes[attr] = value
            
            elif class_name == "MetaAgent":
                for attr in ["agent_types", "code_generation_capacity"]:
                    if hasattr(individual, attr):
                        value = getattr(individual, attr)
                        if isinstance(value, list):
                            attributes[attr] = tuple(value)  # Konversi list ke tuple agar hashable
                        else:
                            attributes[attr] = value
        
        # Ekstrak informasi alat
        if hasattr(individual, "tools"):
            tool_names = tuple(tool.name for tool in individual.tools)
            attributes["tools"] = tool_names
        
        return attributes
    
    def measure(self, population: List[T]) -> float:
        """
        Ukur keragaman struktural populasi.
        
        Args:
            population: Populasi individu
            
        Returns:
            Nilai keragaman
        """
        if len(population) <= 1:
            return 0.0
        
        # Gunakan ekstractor default jika tidak diberikan
        extractor = self.attribute_extractor or self._default_attribute_extractor
        
        # Ekstrak atribut untuk setiap individu
        attributes = [extractor(ind) for ind in population]
        
        # Hitung keragaman berdasarkan atribut unik
        unique_attributes = set()
        
        for attr_dict in attributes:
            # Konversi dict ke frozenset untuk membuat hashable
            attr_items = frozenset(attr_dict.items())
            unique_attributes.add(attr_items)
        
        # Keragaman adalah rasio atribut unik terhadap ukuran populasi
        return len(unique_attributes) / len(population)


class GenotypeDiversity(DiversityMetric[T]):
    """
    Metrik keragaman genotipe.
    
    Metrik ini mengukur keragaman genotipe individu dalam populasi.
    """
    
    def __init__(self, genotype_extractor: Callable[[T], List[float]]):
        """
        Inisialisasi metrik keragaman genotipe.
        
        Args:
            genotype_extractor: Fungsi yang mengekstrak genotipe individu
        """
        super().__init__()
        self.genotype_extractor = genotype_extractor
    
    def measure(self, population: List[T]) -> float:
        """
        Ukur keragaman genotipe populasi.
        
        Args:
            population: Populasi individu
            
        Returns:
            Nilai keragaman
        """
        if len(population) <= 1:
            return 0.0
        
        # Ekstrak genotipe untuk setiap individu
        genotypes = [self.genotype_extractor(ind) for ind in population]
        
        # Konversi ke array numpy
        genotypes = np.array(genotypes)
        
        # Hitung varians untuk setiap gen
        variances = np.var(genotypes, axis=0)
        
        # Keragaman adalah rata-rata varians
        return float(np.mean(variances))


class PhenotypeDiversity(DiversityMetric[T]):
    """
    Metrik keragaman fenotipe.
    
    Metrik ini mengukur keragaman fenotipe individu dalam populasi.
    """
    
    def __init__(self, phenotype_extractor: Callable[[T], List[float]],
                task: Any = None):
        """
        Inisialisasi metrik keragaman fenotipe.
        
        Args:
            phenotype_extractor: Fungsi yang mengekstrak fenotipe individu
            task: Tugas untuk evaluasi (opsional)
        """
        super().__init__()
        self.phenotype_extractor = phenotype_extractor
        self.task = task
    
    def measure(self, population: List[T]) -> float:
        """
        Ukur keragaman fenotipe populasi.
        
        Args:
            population: Populasi individu
            
        Returns:
            Nilai keragaman
        """
        if len(population) <= 1:
            return 0.0
        
        # Ekstrak fenotipe untuk setiap individu
        phenotypes = [self.phenotype_extractor(ind) for ind in population]
        
        # Konversi ke array numpy
        phenotypes = np.array(phenotypes)
        
        # Hitung matriks jarak
        n = len(phenotypes)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Hitung jarak Euclidean
                dist = np.sqrt(np.sum((phenotypes[i] - phenotypes[j]) ** 2))
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Keragaman adalah jarak rata-rata
        return float(np.mean(distances))


class FunctionalDiversity(DiversityMetric[T]):
    """
    Metrik keragaman fungsional.
    
    Metrik ini mengukur keragaman fungsional individu dalam populasi
    berdasarkan hasil yang dihasilkan untuk serangkaian tugas.
    """
    
    def __init__(self, tasks: List[Any], evaluation_function: Callable[[T, Any], Any]):
        """
        Inisialisasi metrik keragaman fungsional.
        
        Args:
            tasks: Daftar tugas untuk evaluasi
            evaluation_function: Fungsi evaluasi
        """
        super().__init__()
        self.tasks = tasks
        self.evaluation_function = evaluation_function
    
    def measure(self, population: List[T]) -> float:
        """
        Ukur keragaman fungsional populasi.
        
        Args:
            population: Populasi individu
            
        Returns:
            Nilai keragaman
        """
        if len(population) <= 1:
            return 0.0
        
        # Evaluasi setiap individu pada setiap tugas
        results = []
        
        for ind in population:
            ind_results = []
            for task in self.tasks:
                try:
                    result = self.evaluation_function(ind, task)
                    ind_results.append(result)
                except Exception:
                    # Jika evaluasi gagal, gunakan None
                    ind_results.append(None)
            results.append(ind_results)
        
        # Hitung keragaman berdasarkan hasil unik
        unique_results = set()
        
        for ind_results in results:
            # Konversi hasil ke string untuk membuat hashable
            result_str = str(ind_results)
            unique_results.add(result_str)
        
        # Keragaman adalah rasio hasil unik terhadap ukuran populasi
        return len(unique_results) / len(population)