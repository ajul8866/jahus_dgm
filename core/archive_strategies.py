"""
Strategi arsip untuk Darwin-Gödel Machine.

Modul ini berisi implementasi berbagai strategi arsip yang digunakan oleh DGM
untuk mempertahankan dan mengelola arsip agen selama proses evolusi.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar, Generic, Union, Set

from simple_dgm.agents.base_agent import BaseAgent

# Tipe generik untuk individu dalam populasi
T = TypeVar('T', bound=BaseAgent)


class ArchiveStrategy(Generic[T]):
    """
    Kelas dasar untuk strategi arsip.
    
    Strategi arsip menentukan bagaimana individu disimpan dan dikelola dalam arsip.
    """
    
    def __init__(self, capacity: int = 100):
        """
        Inisialisasi strategi arsip.
        
        Args:
            capacity: Kapasitas arsip
        """
        self.capacity = capacity
        self.archive = []  # Tipe: List[Tuple[T, float]]
    
    def add(self, individual: T, fitness: float) -> bool:
        """
        Tambahkan individu ke arsip.
        
        Args:
            individual: Individu yang akan ditambahkan
            fitness: Nilai fitness individu
            
        Returns:
            True jika individu berhasil ditambahkan, False jika tidak
        """
        raise NotImplementedError("Subclass must implement abstract method")
    
    def get(self, index: int) -> Tuple[T, float]:
        """
        Dapatkan individu dari arsip.
        
        Args:
            index: Indeks individu
            
        Returns:
            Individu dan nilai fitness
        """
        if 0 <= index < len(self.archive):
            return self.archive[index]
        else:
            raise IndexError("Archive index out of range")
    
    def get_random(self) -> Tuple[T, float]:
        """
        Dapatkan individu acak dari arsip.
        
        Returns:
            Individu dan nilai fitness
        """
        if self.archive:
            return random.choice(self.archive)
        else:
            raise IndexError("Archive is empty")
    
    def get_best(self) -> Tuple[T, float]:
        """
        Dapatkan individu terbaik dari arsip.
        
        Returns:
            Individu dan nilai fitness
        """
        if self.archive:
            return max(self.archive, key=lambda x: x[1])
        else:
            raise IndexError("Archive is empty")
    
    def get_all(self) -> List[Tuple[T, float]]:
        """
        Dapatkan semua individu dari arsip.
        
        Returns:
            Daftar individu dan nilai fitness
        """
        return self.archive.copy()
    
    def clear(self):
        """
        Kosongkan arsip.
        """
        self.archive = []
    
    def __len__(self) -> int:
        """
        Dapatkan ukuran arsip.
        
        Returns:
            Ukuran arsip
        """
        return len(self.archive)


class NoveltyArchive(ArchiveStrategy[T]):
    """
    Strategi arsip kebaruan.
    
    Strategi ini mempertahankan individu berdasarkan kebaruan perilaku mereka.
    """
    
    def __init__(self, capacity: int = 100, behavior_descriptor: Callable[[T], List[float]] = None,
                distance_metric: str = "euclidean", novelty_threshold: float = 0.1):
        """
        Inisialisasi strategi arsip kebaruan.
        
        Args:
            capacity: Kapasitas arsip
            behavior_descriptor: Fungsi yang mendeskripsikan perilaku individu
            distance_metric: Metrik jarak ("euclidean", "manhattan", "cosine")
            novelty_threshold: Ambang batas kebaruan
        """
        super().__init__(capacity)
        self.behavior_descriptor = behavior_descriptor
        self.distance_metric = distance_metric
        self.novelty_threshold = novelty_threshold
        self.behaviors = []  # Tipe: List[List[float]]
    
    def _calculate_distance(self, behavior1: List[float], behavior2: List[float]) -> float:
        """
        Hitung jarak antara dua perilaku.
        
        Args:
            behavior1: Perilaku pertama
            behavior2: Perilaku kedua
            
        Returns:
            Jarak antara dua perilaku
        """
        if self.distance_metric == "euclidean":
            return np.sqrt(sum((a - b) ** 2 for a, b in zip(behavior1, behavior2)))
        elif self.distance_metric == "manhattan":
            return sum(abs(a - b) for a, b in zip(behavior1, behavior2))
        elif self.distance_metric == "cosine":
            dot = sum(a * b for a, b in zip(behavior1, behavior2))
            norm1 = np.sqrt(sum(a ** 2 for a in behavior1))
            norm2 = np.sqrt(sum(b ** 2 for b in behavior2))
            return 1.0 - (dot / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 1.0
        else:
            # Default ke Euclidean
            return np.sqrt(sum((a - b) ** 2 for a, b in zip(behavior1, behavior2)))
    
    def _calculate_novelty(self, behavior: List[float], k: int = 5) -> float:
        """
        Hitung kebaruan perilaku.
        
        Args:
            behavior: Perilaku yang akan dihitung kebaruannya
            k: Jumlah tetangga terdekat
            
        Returns:
            Nilai kebaruan
        """
        if not self.behaviors:
            return float('inf')
        
        # Hitung jarak ke semua perilaku dalam arsip
        distances = [self._calculate_distance(behavior, b) for b in self.behaviors]
        
        # Urutkan jarak
        sorted_distances = sorted(distances)
        
        # Hitung kebaruan sebagai jarak rata-rata ke k tetangga terdekat
        k_nearest = sorted_distances[:min(k, len(sorted_distances))]
        return sum(k_nearest) / len(k_nearest)
    
    def add(self, individual: T, fitness: float) -> bool:
        """
        Tambahkan individu ke arsip berdasarkan kebaruan.
        
        Args:
            individual: Individu yang akan ditambahkan
            fitness: Nilai fitness individu
            
        Returns:
            True jika individu berhasil ditambahkan, False jika tidak
        """
        # Hitung perilaku individu
        behavior = self.behavior_descriptor(individual)
        
        # Hitung kebaruan
        novelty = self._calculate_novelty(behavior)
        
        # Periksa apakah individu cukup baru
        if novelty >= self.novelty_threshold:
            # Tambahkan individu ke arsip
            self.archive.append((individual, fitness))
            self.behaviors.append(behavior)
            
            # Jika arsip melebihi kapasitas, hapus individu dengan kebaruan terendah
            if len(self.archive) > self.capacity:
                # Hitung kebaruan untuk semua individu
                novelties = [self._calculate_novelty(b) for b in self.behaviors]
                
                # Temukan indeks individu dengan kebaruan terendah
                min_index = novelties.index(min(novelties))
                
                # Hapus individu
                self.archive.pop(min_index)
                self.behaviors.pop(min_index)
            
            return True
        
        return False


class EliteArchive(ArchiveStrategy[T]):
    """
    Strategi arsip elit.
    
    Strategi ini mempertahankan individu terbaik berdasarkan nilai fitness.
    """
    
    def __init__(self, capacity: int = 100, similarity_threshold: float = 0.1,
                similarity_measure: Optional[Callable[[T, T], float]] = None):
        """
        Inisialisasi strategi arsip elit.
        
        Args:
            capacity: Kapasitas arsip
            similarity_threshold: Ambang batas kesamaan
            similarity_measure: Fungsi pengukur kesamaan
        """
        super().__init__(capacity)
        self.similarity_threshold = similarity_threshold
        self.similarity_measure = similarity_measure
    
    def _is_similar(self, individual1: T, individual2: T) -> bool:
        """
        Periksa apakah dua individu mirip.
        
        Args:
            individual1: Individu pertama
            individual2: Individu kedua
            
        Returns:
            True jika individu mirip, False jika tidak
        """
        if self.similarity_measure:
            return self.similarity_measure(individual1, individual2) >= self.similarity_threshold
        
        # Default: periksa kesamaan atribut
        similar = True
        
        # Periksa atribut dasar
        for attr in ["learning_rate", "exploration_rate", "memory_capacity"]:
            if hasattr(individual1, attr) and hasattr(individual2, attr):
                value1 = getattr(individual1, attr)
                value2 = getattr(individual2, attr)
                
                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    # Untuk atribut numerik, periksa perbedaan relatif
                    max_val = max(abs(value1), abs(value2))
                    if max_val > 0:
                        diff = abs(value1 - value2) / max_val
                        if diff > self.similarity_threshold:
                            similar = False
                            break
                elif value1 != value2:
                    similar = False
                    break
        
        # Periksa alat
        if similar and hasattr(individual1, "tools") and hasattr(individual2, "tools"):
            tools1 = {tool.name for tool in individual1.tools}
            tools2 = {tool.name for tool in individual2.tools}
            
            # Hitung Jaccard similarity
            intersection = len(tools1 & tools2)
            union = len(tools1 | tools2)
            
            if union > 0:
                similarity = intersection / union
                if similarity < self.similarity_threshold:
                    similar = False
        
        return similar
    
    def add(self, individual: T, fitness: float) -> bool:
        """
        Tambahkan individu ke arsip berdasarkan elitisme.
        
        Args:
            individual: Individu yang akan ditambahkan
            fitness: Nilai fitness individu
            
        Returns:
            True jika individu berhasil ditambahkan, False jika tidak
        """
        # Periksa apakah individu mirip dengan yang sudah ada di arsip
        for i, (existing, _) in enumerate(self.archive):
            if self._is_similar(individual, existing):
                # Jika individu mirip, ganti jika lebih baik
                if fitness > self.archive[i][1]:
                    self.archive[i] = (individual, fitness)
                    return True
                return False
        
        # Jika arsip belum penuh, tambahkan individu
        if len(self.archive) < self.capacity:
            self.archive.append((individual, fitness))
            return True
        
        # Jika arsip penuh, ganti individu terburuk jika individu baru lebih baik
        if self.archive:
            worst_index = min(range(len(self.archive)), key=lambda i: self.archive[i][1])
            if fitness > self.archive[worst_index][1]:
                self.archive[worst_index] = (individual, fitness)
                return True
        
        return False


class QualityDiversityArchive(ArchiveStrategy[T]):
    """
    Strategi arsip kualitas-keragaman.
    
    Strategi ini mempertahankan individu berdasarkan kualitas dan keragaman.
    """
    
    def __init__(self, capacity: int = 100, feature_descriptor: Callable[[T], List[float]] = None,
                feature_dimensions: List[Tuple[float, float, int]] = None):
        """
        Inisialisasi strategi arsip kualitas-keragaman.
        
        Args:
            capacity: Kapasitas arsip
            feature_descriptor: Fungsi yang mendeskripsikan fitur individu
            feature_dimensions: Daftar dimensi fitur (min, max, bins)
        """
        super().__init__(capacity)
        self.feature_descriptor = feature_descriptor
        self.feature_dimensions = feature_dimensions
        self.grid = {}  # Tipe: Dict[Tuple[int, ...], Tuple[T, float]]
    
    def _get_cell_index(self, features: List[float]) -> Tuple[int, ...]:
        """
        Dapatkan indeks sel dalam grid berdasarkan fitur.
        
        Args:
            features: Fitur individu
            
        Returns:
            Indeks sel dalam grid
        """
        indices = []
        
        for i, (min_val, max_val, bins) in enumerate(self.feature_dimensions):
            if i < len(features):
                # Batasi fitur ke rentang yang valid
                feature = max(min_val, min(max_val, features[i]))
                
                # Hitung indeks bin
                bin_size = (max_val - min_val) / bins
                bin_index = min(bins - 1, int((feature - min_val) / bin_size))
                
                indices.append(bin_index)
            else:
                indices.append(0)
        
        return tuple(indices)
    
    def add(self, individual: T, fitness: float) -> bool:
        """
        Tambahkan individu ke arsip berdasarkan kualitas-keragaman.
        
        Args:
            individual: Individu yang akan ditambahkan
            fitness: Nilai fitness individu
            
        Returns:
            True jika individu berhasil ditambahkan, False jika tidak
        """
        # Hitung fitur individu
        features = self.feature_descriptor(individual)
        
        # Dapatkan indeks sel
        cell_index = self._get_cell_index(features)
        
        # Periksa apakah sel sudah terisi
        if cell_index in self.grid:
            # Jika individu baru lebih baik, ganti
            if fitness > self.grid[cell_index][1]:
                self.grid[cell_index] = (individual, fitness)
                return True
            return False
        
        # Jika sel kosong, tambahkan individu
        self.grid[cell_index] = (individual, fitness)
        
        # Perbarui arsip
        self.archive = list(self.grid.values())
        
        # Jika arsip melebihi kapasitas, hapus sel dengan fitness terendah
        if len(self.grid) > self.capacity:
            # Temukan sel dengan fitness terendah
            worst_cell = min(self.grid.keys(), key=lambda cell: self.grid[cell][1])
            
            # Hapus sel
            del self.grid[worst_cell]
            
            # Perbarui arsip
            self.archive = list(self.grid.values())
        
        return True
    
    def get_all(self) -> List[Tuple[T, float]]:
        """
        Dapatkan semua individu dari arsip.
        
        Returns:
            Daftar individu dan nilai fitness
        """
        return list(self.grid.values())


class ParetoDominanceArchive(ArchiveStrategy[T]):
    """
    Strategi arsip dominasi Pareto.
    
    Strategi ini mempertahankan individu berdasarkan dominasi Pareto.
    """
    
    def __init__(self, capacity: int = 100, objectives: List[Callable[[T], float]] = None):
        """
        Inisialisasi strategi arsip dominasi Pareto.
        
        Args:
            capacity: Kapasitas arsip
            objectives: Daftar fungsi objektif
        """
        super().__init__(capacity)
        self.objectives = objectives
        self.objective_values = []  # Tipe: List[List[float]]
    
    def _dominates(self, values1: List[float], values2: List[float]) -> bool:
        """
        Periksa apakah values1 mendominasi values2.
        
        Args:
            values1: Nilai objektif pertama
            values2: Nilai objektif kedua
            
        Returns:
            True jika values1 mendominasi values2, False jika tidak
        """
        # values1 mendominasi values2 jika:
        # 1. values1 tidak lebih buruk dari values2 dalam semua objektif
        # 2. values1 lebih baik dari values2 dalam setidaknya satu objektif
        
        not_worse = all(v1 >= v2 for v1, v2 in zip(values1, values2))
        better = any(v1 > v2 for v1, v2 in zip(values1, values2))
        
        return not_worse and better
    
    def _is_dominated(self, values: List[float]) -> bool:
        """
        Periksa apakah nilai objektif didominasi oleh nilai dalam arsip.
        
        Args:
            values: Nilai objektif
            
        Returns:
            True jika nilai didominasi, False jika tidak
        """
        return any(self._dominates(existing, values) for existing in self.objective_values)
    
    def add(self, individual: T, fitness: float) -> bool:
        """
        Tambahkan individu ke arsip berdasarkan dominasi Pareto.
        
        Args:
            individual: Individu yang akan ditambahkan
            fitness: Nilai fitness individu (tidak digunakan)
            
        Returns:
            True jika individu berhasil ditambahkan, False jika tidak
        """
        # Hitung nilai objektif
        objective_values = [obj(individual) for obj in self.objectives]
        
        # Periksa apakah individu didominasi oleh individu dalam arsip
        if self._is_dominated(objective_values):
            return False
        
        # Temukan individu yang didominasi oleh individu baru
        dominated_indices = [i for i, values in enumerate(self.objective_values)
                           if self._dominates(objective_values, values)]
        
        # Hapus individu yang didominasi
        for i in sorted(dominated_indices, reverse=True):
            self.archive.pop(i)
            self.objective_values.pop(i)
        
        # Tambahkan individu baru
        self.archive.append((individual, 0.0))  # Fitness tidak digunakan
        self.objective_values.append(objective_values)
        
        # Jika arsip melebihi kapasitas, gunakan crowding distance
        if len(self.archive) > self.capacity:
            # Hitung crowding distance
            distances = self._calculate_crowding_distance()
            
            # Hapus individu dengan crowding distance terendah
            min_index = distances.index(min(distances))
            self.archive.pop(min_index)
            self.objective_values.pop(min_index)
        
        return True
    
    def _calculate_crowding_distance(self) -> List[float]:
        """
        Hitung crowding distance untuk individu dalam arsip.
        
        Returns:
            Daftar crowding distance
        """
        n = len(self.archive)
        if n <= 2:
            return [float('inf')] * n
        
        # Inisialisasi jarak
        distances = [0.0] * n
        
        # Untuk setiap objektif
        for m in range(len(self.objectives)):
            # Urutkan individu berdasarkan objektif m
            sorted_indices = sorted(range(n), key=lambda i: self.objective_values[i][m])
            
            # Tetapkan jarak tak terhingga untuk individu di ujung
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Hitung jarak untuk individu di tengah
            obj_min = self.objective_values[sorted_indices[0]][m]
            obj_max = self.objective_values[sorted_indices[-1]][m]
            
            if obj_max == obj_min:
                continue
            
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (self.objective_values[sorted_indices[i+1]][m] - 
                                              self.objective_values[sorted_indices[i-1]][m]) / (obj_max - obj_min)
        
        return distances


class AgeLayeredArchive(ArchiveStrategy[T]):
    """
    Strategi arsip berlapis usia.
    
    Strategi ini mempertahankan individu berdasarkan usia dan kualitas.
    """
    
    def __init__(self, capacity: int = 100, num_layers: int = 5, layer_size: int = 20):
        """
        Inisialisasi strategi arsip berlapis usia.
        
        Args:
            capacity: Kapasitas arsip
            num_layers: Jumlah lapisan usia
            layer_size: Ukuran setiap lapisan
        """
        super().__init__(capacity)
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.layers = [[] for _ in range(num_layers)]  # Tipe: List[List[Tuple[T, float]]]
        self.generation = 0
    
    def add(self, individual: T, fitness: float) -> bool:
        """
        Tambahkan individu ke arsip berdasarkan usia dan kualitas.
        
        Args:
            individual: Individu yang akan ditambahkan
            fitness: Nilai fitness individu
            
        Returns:
            True jika individu berhasil ditambahkan, False jika tidak
        """
        # Tentukan lapisan berdasarkan generasi
        layer_index = self.generation % self.num_layers
        
        # Tambahkan individu ke lapisan
        self.layers[layer_index].append((individual, fitness))
        
        # Jika lapisan melebihi ukuran, hapus individu terburuk
        if len(self.layers[layer_index]) > self.layer_size:
            # Urutkan lapisan berdasarkan fitness
            self.layers[layer_index].sort(key=lambda x: x[1], reverse=True)
            
            # Potong lapisan
            self.layers[layer_index] = self.layers[layer_index][:self.layer_size]
        
        # Perbarui arsip
        self.archive = [item for layer in self.layers for item in layer]
        
        # Tingkatkan generasi
        self.generation += 1
        
        return True
    
    def get_layer(self, layer_index: int) -> List[Tuple[T, float]]:
        """
        Dapatkan individu dari lapisan tertentu.
        
        Args:
            layer_index: Indeks lapisan
            
        Returns:
            Daftar individu dan nilai fitness
        """
        if 0 <= layer_index < self.num_layers:
            return self.layers[layer_index].copy()
        else:
            raise IndexError("Layer index out of range")