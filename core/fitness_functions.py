"""
Fungsi fitness untuk Darwin-Gödel Machine.

Modul ini berisi implementasi berbagai fungsi fitness yang digunakan oleh DGM
untuk mengevaluasi kualitas agen-agen selama proses evolusi.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar, Generic, Union

from simple_dgm.agents.base_agent import BaseAgent

# Tipe generik untuk individu dalam populasi
T = TypeVar('T', bound=BaseAgent)


class FitnessFunction(Generic[T]):
    """
    Kelas dasar untuk fungsi fitness.
    
    Fungsi fitness mengevaluasi kualitas individu dalam populasi.
    """
    
    def __init__(self, evaluation_function: Optional[Callable[[T, Any], float]] = None):
        """
        Inisialisasi fungsi fitness.
        
        Args:
            evaluation_function: Fungsi evaluasi eksternal
        """
        self.evaluation_function = evaluation_function
    
    def evaluate(self, individual: T, task: Any) -> float:
        """
        Evaluasi individu.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Nilai fitness
        """
        if self.evaluation_function:
            return self.evaluation_function(individual, task)
        else:
            return self._evaluate(individual, task)
    
    def _evaluate(self, individual: T, task: Any) -> float:
        """
        Implementasi spesifik evaluasi.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Nilai fitness
        """
        raise NotImplementedError("Subclass must implement abstract method")


class MultiObjectiveFitness(FitnessFunction[T]):
    """
    Fungsi fitness multi-objektif.
    
    Fungsi ini mengevaluasi individu berdasarkan beberapa objektif.
    """
    
    def __init__(self, objectives: List[Callable[[T, Any], float]],
                weights: Optional[List[float]] = None):
        """
        Inisialisasi fungsi fitness multi-objektif.
        
        Args:
            objectives: Daftar fungsi objektif
            weights: Bobot untuk setiap objektif (opsional)
        """
        super().__init__()
        self.objectives = objectives
        
        # Bobot default jika tidak diberikan
        if weights is None:
            weights = [1.0] * len(objectives)
        
        # Normalisasi bobot
        total = sum(weights)
        self.weights = [w / total for w in weights]
    
    def _evaluate(self, individual: T, task: Any) -> List[float]:
        """
        Evaluasi individu berdasarkan beberapa objektif.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Daftar nilai fitness untuk setiap objektif
        """
        return [obj(individual, task) for obj in self.objectives]
    
    def evaluate(self, individual: T, task: Any) -> List[float]:
        """
        Evaluasi individu.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Daftar nilai fitness untuk setiap objektif
        """
        return self._evaluate(individual, task)
    
    def aggregate(self, individual: T, task: Any) -> float:
        """
        Agregasi nilai fitness multi-objektif menjadi nilai tunggal.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Nilai fitness agregat
        """
        # Evaluasi individu
        fitness_values = self._evaluate(individual, task)
        
        # Agregasi dengan bobot
        return sum(f * w for f, w in zip(fitness_values, self.weights))


class LexicographicFitness(FitnessFunction[T]):
    """
    Fungsi fitness leksikografis.
    
    Fungsi ini mengevaluasi individu berdasarkan beberapa objektif
    dengan prioritas leksikografis.
    """
    
    def __init__(self, objectives: List[Callable[[T, Any], float]],
                thresholds: Optional[List[float]] = None):
        """
        Inisialisasi fungsi fitness leksikografis.
        
        Args:
            objectives: Daftar fungsi objektif (dalam urutan prioritas)
            thresholds: Ambang batas untuk setiap objektif (opsional)
        """
        super().__init__()
        self.objectives = objectives
        
        # Ambang batas default jika tidak diberikan
        if thresholds is None:
            thresholds = [0.001] * len(objectives)
        
        self.thresholds = thresholds
    
    def _evaluate(self, individual: T, task: Any) -> List[float]:
        """
        Evaluasi individu berdasarkan beberapa objektif.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Daftar nilai fitness untuk setiap objektif
        """
        return [obj(individual, task) for obj in self.objectives]
    
    def evaluate(self, individual: T, task: Any) -> List[float]:
        """
        Evaluasi individu.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Daftar nilai fitness untuk setiap objektif
        """
        return self._evaluate(individual, task)
    
    def compare(self, fitness1: List[float], fitness2: List[float]) -> int:
        """
        Bandingkan dua nilai fitness leksikografis.
        
        Args:
            fitness1: Nilai fitness pertama
            fitness2: Nilai fitness kedua
            
        Returns:
            1 jika fitness1 lebih baik, -1 jika fitness2 lebih baik, 0 jika sama
        """
        for i in range(len(fitness1)):
            # Periksa apakah perbedaan signifikan
            if abs(fitness1[i] - fitness2[i]) > self.thresholds[i]:
                return 1 if fitness1[i] > fitness2[i] else -1
        
        # Jika semua objektif sama, kembalikan 0
        return 0


class AggregatedFitness(FitnessFunction[T]):
    """
    Fungsi fitness agregasi.
    
    Fungsi ini mengevaluasi individu berdasarkan beberapa objektif
    yang diagregasi menjadi nilai tunggal.
    """
    
    def __init__(self, objectives: List[Callable[[T, Any], float]],
                weights: Optional[List[float]] = None,
                aggregation_method: str = "weighted_sum"):
        """
        Inisialisasi fungsi fitness agregasi.
        
        Args:
            objectives: Daftar fungsi objektif
            weights: Bobot untuk setiap objektif (opsional)
            aggregation_method: Metode agregasi ("weighted_sum", "weighted_product", "tchebycheff")
        """
        super().__init__()
        self.objectives = objectives
        
        # Bobot default jika tidak diberikan
        if weights is None:
            weights = [1.0] * len(objectives)
        
        # Normalisasi bobot
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        # Metode agregasi
        self.aggregation_method = aggregation_method
        
        # Titik referensi untuk metode Tchebycheff
        self.reference_point = None
    
    def set_reference_point(self, reference_point: List[float]):
        """
        Tetapkan titik referensi untuk metode Tchebycheff.
        
        Args:
            reference_point: Titik referensi
        """
        self.reference_point = reference_point
    
    def _evaluate(self, individual: T, task: Any) -> float:
        """
        Evaluasi individu berdasarkan beberapa objektif yang diagregasi.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Nilai fitness agregat
        """
        # Evaluasi setiap objektif
        fitness_values = [obj(individual, task) for obj in self.objectives]
        
        # Agregasi berdasarkan metode
        if self.aggregation_method == "weighted_sum":
            # Jumlah berbobot
            return sum(f * w for f, w in zip(fitness_values, self.weights))
        
        elif self.aggregation_method == "weighted_product":
            # Produk berbobot
            return math.prod(f ** w for f, w in zip(fitness_values, self.weights))
        
        elif self.aggregation_method == "tchebycheff":
            # Metode Tchebycheff
            if self.reference_point is None:
                # Jika titik referensi tidak diberikan, gunakan nilai maksimum
                self.reference_point = [1.0] * len(fitness_values)
            
            # Hitung jarak berbobot maksimum
            return -max(w * abs(z - f) for w, z, f in zip(self.weights, self.reference_point, fitness_values))
        
        else:
            # Default ke jumlah berbobot
            return sum(f * w for f, w in zip(fitness_values, self.weights))


class ParetoDominanceFitness(FitnessFunction[T]):
    """
    Fungsi fitness dominasi Pareto.
    
    Fungsi ini mengevaluasi individu berdasarkan dominasi Pareto.
    """
    
    def __init__(self, objectives: List[Callable[[T, Any], float]]):
        """
        Inisialisasi fungsi fitness dominasi Pareto.
        
        Args:
            objectives: Daftar fungsi objektif
        """
        super().__init__()
        self.objectives = objectives
    
    def _evaluate(self, individual: T, task: Any) -> List[float]:
        """
        Evaluasi individu berdasarkan beberapa objektif.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Daftar nilai fitness untuk setiap objektif
        """
        return [obj(individual, task) for obj in self.objectives]
    
    def evaluate(self, individual: T, task: Any) -> List[float]:
        """
        Evaluasi individu.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Daftar nilai fitness untuk setiap objektif
        """
        return self._evaluate(individual, task)
    
    def dominates(self, fitness1: List[float], fitness2: List[float]) -> bool:
        """
        Periksa apakah fitness1 mendominasi fitness2.
        
        Args:
            fitness1: Nilai fitness pertama
            fitness2: Nilai fitness kedua
            
        Returns:
            True jika fitness1 mendominasi fitness2, False jika tidak
        """
        # fitness1 mendominasi fitness2 jika:
        # 1. fitness1 tidak lebih buruk dari fitness2 dalam semua objektif
        # 2. fitness1 lebih baik dari fitness2 dalam setidaknya satu objektif
        
        not_worse = all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2))
        better = any(f1 > f2 for f1, f2 in zip(fitness1, fitness2))
        
        return not_worse and better
    
    def get_non_dominated_rank(self, population: List[Tuple[T, List[float]]]) -> List[int]:
        """
        Dapatkan peringkat non-dominasi untuk populasi.
        
        Args:
            population: Populasi individu dan nilai fitness mereka
            
        Returns:
            Peringkat non-dominasi untuk setiap individu
        """
        n = len(population)
        ranks = [0] * n
        
        # Untuk setiap individu
        for i in range(n):
            # Hitung jumlah individu yang mendominasi i
            dominated_by = 0
            
            for j in range(n):
                if i != j and self.dominates(population[j][1], population[i][1]):
                    dominated_by += 1
            
            # Tetapkan peringkat
            ranks[i] = dominated_by
        
        return ranks
    
    def get_crowding_distance(self, population: List[Tuple[T, List[float]]]) -> List[float]:
        """
        Hitung jarak kerumunan untuk populasi.
        
        Args:
            population: Populasi individu dan nilai fitness mereka
            
        Returns:
            Jarak kerumunan untuk setiap individu
        """
        n = len(population)
        if n <= 2:
            return [float('inf')] * n
        
        # Inisialisasi jarak
        distances = [0.0] * n
        
        # Untuk setiap objektif
        for m in range(len(population[0][1])):
            # Urutkan populasi berdasarkan objektif m
            sorted_pop = sorted(range(n), key=lambda i: population[i][1][m])
            
            # Tetapkan jarak tak terhingga untuk individu di ujung
            distances[sorted_pop[0]] = float('inf')
            distances[sorted_pop[-1]] = float('inf')
            
            # Hitung jarak untuk individu di tengah
            f_max = population[sorted_pop[-1]][1][m]
            f_min = population[sorted_pop[0]][1][m]
            
            if f_max == f_min:
                continue
            
            for i in range(1, n - 1):
                distances[sorted_pop[i]] += (population[sorted_pop[i+1]][1][m] - 
                                           population[sorted_pop[i-1]][1][m]) / (f_max - f_min)
        
        return distances