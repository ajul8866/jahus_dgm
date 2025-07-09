"""
Operator crossover untuk Darwin-Gödel Machine.

Modul ini berisi implementasi berbagai operator crossover yang digunakan oleh DGM
untuk menggabungkan agen-agen induk selama proses evolusi.
"""

import random
import copy
import inspect
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar, Generic, Union

from simple_dgm.agents.base_agent import BaseAgent, Tool

# Tipe generik untuk individu dalam populasi
T = TypeVar('T', bound=BaseAgent)


class CrossoverOperator(Generic[T]):
    """
    Kelas dasar untuk operator crossover.
    
    Operator crossover menentukan bagaimana dua individu induk digabungkan
    untuk menghasilkan individu anak.
    """
    
    def __init__(self, crossover_rate: float = 0.7):
        """
        Inisialisasi operator crossover.
        
        Args:
            crossover_rate: Probabilitas crossover
        """
        self.crossover_rate = crossover_rate
    
    def crossover(self, parent1: T, parent2: T) -> T:
        """
        Lakukan crossover antara dua individu induk.
        
        Args:
            parent1: Individu induk pertama
            parent2: Individu induk kedua
            
        Returns:
            Individu anak hasil crossover
        """
        # Periksa apakah crossover akan dilakukan
        if random.random() > self.crossover_rate:
            # Jika tidak, kembalikan salinan parent1
            return copy.deepcopy(parent1)
        
        # Lakukan crossover
        return self._do_crossover(parent1, parent2)
    
    def _do_crossover(self, parent1: T, parent2: T) -> T:
        """
        Implementasi spesifik crossover.
        
        Args:
            parent1: Individu induk pertama
            parent2: Individu induk kedua
            
        Returns:
            Individu anak hasil crossover
        """
        raise NotImplementedError("Subclass must implement abstract method")


class UniformCrossover(CrossoverOperator[T]):
    """
    Operator crossover seragam.
    
    Operator ini memilih setiap atribut dari salah satu induk secara acak.
    """
    
    def __init__(self, crossover_rate: float = 0.7, attribute_swap_prob: float = 0.5,
                attributes: Optional[List[str]] = None):
        """
        Inisialisasi operator crossover seragam.
        
        Args:
            crossover_rate: Probabilitas crossover
            attribute_swap_prob: Probabilitas penukaran setiap atribut
            attributes: Daftar atribut yang akan ditukar
        """
        super().__init__(crossover_rate)
        self.attribute_swap_prob = attribute_swap_prob
        
        # Atribut default jika tidak diberikan
        if attributes is None:
            attributes = [
                "learning_rate", "exploration_rate", "memory_capacity",
                "code_style", "preferred_language",
                "problem_types", "max_iterations", "timeout",
                "agent_types", "code_generation_capacity"
            ]
        
        self.attributes = attributes
    
    def _do_crossover(self, parent1: T, parent2: T) -> T:
        """
        Lakukan crossover seragam.
        
        Args:
            parent1: Individu induk pertama
            parent2: Individu induk kedua
            
        Returns:
            Individu anak hasil crossover
        """
        # Buat salinan parent1 sebagai dasar
        child = copy.deepcopy(parent1)
        
        # Tukar atribut dengan probabilitas attribute_swap_prob
        for attr in self.attributes:
            if hasattr(parent1, attr) and hasattr(parent2, attr) and random.random() < self.attribute_swap_prob:
                # Dapatkan nilai dari parent2
                value = getattr(parent2, attr)
                
                # Salin nilai ke child
                if isinstance(value, (int, float, str, bool)):
                    # Untuk tipe data sederhana, salin langsung
                    setattr(child, attr, value)
                elif isinstance(value, list):
                    # Untuk list, buat salinan
                    setattr(child, attr, copy.deepcopy(value))
                elif isinstance(value, dict):
                    # Untuk dict, buat salinan
                    setattr(child, attr, copy.deepcopy(value))
        
        # Tukar alat
        if hasattr(parent1, "tools") and hasattr(parent2, "tools") and random.random() < self.attribute_swap_prob:
            # Dapatkan alat dari kedua induk
            tools1 = {tool.name: tool for tool in parent1.tools}
            tools2 = {tool.name: tool for tool in parent2.tools}
            
            # Buat daftar alat baru
            new_tools = []
            
            # Tambahkan alat dari kedua induk dengan probabilitas attribute_swap_prob
            all_tool_names = set(tools1.keys()) | set(tools2.keys())
            
            for name in all_tool_names:
                if name in tools1 and name in tools2:
                    # Jika alat ada di kedua induk, pilih salah satu
                    tool = tools1[name] if random.random() < 0.5 else tools2[name]
                    new_tools.append(copy.deepcopy(tool))
                elif name in tools1 and random.random() < self.attribute_swap_prob:
                    # Jika alat hanya ada di parent1, tambahkan dengan probabilitas
                    new_tools.append(copy.deepcopy(tools1[name]))
                elif name in tools2 and random.random() < self.attribute_swap_prob:
                    # Jika alat hanya ada di parent2, tambahkan dengan probabilitas
                    new_tools.append(copy.deepcopy(tools2[name]))
            
            # Tetapkan alat baru ke child
            child.tools = new_tools
        
        return child


class SinglePointCrossover(CrossoverOperator[T]):
    """
    Operator crossover satu titik.
    
    Operator ini memilih satu titik dan menukar semua atribut setelah titik tersebut.
    """
    
    def __init__(self, crossover_rate: float = 0.7, attributes: Optional[List[str]] = None):
        """
        Inisialisasi operator crossover satu titik.
        
        Args:
            crossover_rate: Probabilitas crossover
            attributes: Daftar atribut yang akan ditukar
        """
        super().__init__(crossover_rate)
        
        # Atribut default jika tidak diberikan
        if attributes is None:
            attributes = [
                "learning_rate", "exploration_rate", "memory_capacity",
                "code_style", "preferred_language",
                "problem_types", "max_iterations", "timeout",
                "agent_types", "code_generation_capacity"
            ]
        
        self.attributes = attributes
    
    def _do_crossover(self, parent1: T, parent2: T) -> T:
        """
        Lakukan crossover satu titik.
        
        Args:
            parent1: Individu induk pertama
            parent2: Individu induk kedua
            
        Returns:
            Individu anak hasil crossover
        """
        # Buat salinan parent1 sebagai dasar
        child = copy.deepcopy(parent1)
        
        # Dapatkan atribut yang ada di kedua induk
        common_attrs = [attr for attr in self.attributes if hasattr(parent1, attr) and hasattr(parent2, attr)]
        
        if not common_attrs:
            return child
        
        # Pilih titik crossover
        crossover_point = random.randint(1, len(common_attrs))
        
        # Tukar atribut setelah titik crossover
        for i in range(crossover_point, len(common_attrs)):
            attr = common_attrs[i]
            
            # Dapatkan nilai dari parent2
            value = getattr(parent2, attr)
            
            # Salin nilai ke child
            if isinstance(value, (int, float, str, bool)):
                # Untuk tipe data sederhana, salin langsung
                setattr(child, attr, value)
            elif isinstance(value, list):
                # Untuk list, buat salinan
                setattr(child, attr, copy.deepcopy(value))
            elif isinstance(value, dict):
                # Untuk dict, buat salinan
                setattr(child, attr, copy.deepcopy(value))
        
        # Tukar alat jika titik crossover melewati setengah atribut
        if hasattr(parent1, "tools") and hasattr(parent2, "tools") and crossover_point > len(common_attrs) / 2:
            # Salin alat dari parent2
            child.tools = [copy.deepcopy(tool) for tool in parent2.tools]
        
        return child


class MultiPointCrossover(CrossoverOperator[T]):
    """
    Operator crossover multi-titik.
    
    Operator ini memilih beberapa titik dan menukar atribut di antara titik-titik tersebut.
    """
    
    def __init__(self, crossover_rate: float = 0.7, num_points: int = 2,
                attributes: Optional[List[str]] = None):
        """
        Inisialisasi operator crossover multi-titik.
        
        Args:
            crossover_rate: Probabilitas crossover
            num_points: Jumlah titik crossover
            attributes: Daftar atribut yang akan ditukar
        """
        super().__init__(crossover_rate)
        self.num_points = num_points
        
        # Atribut default jika tidak diberikan
        if attributes is None:
            attributes = [
                "learning_rate", "exploration_rate", "memory_capacity",
                "code_style", "preferred_language",
                "problem_types", "max_iterations", "timeout",
                "agent_types", "code_generation_capacity"
            ]
        
        self.attributes = attributes
    
    def _do_crossover(self, parent1: T, parent2: T) -> T:
        """
        Lakukan crossover multi-titik.
        
        Args:
            parent1: Individu induk pertama
            parent2: Individu induk kedua
            
        Returns:
            Individu anak hasil crossover
        """
        # Buat salinan parent1 sebagai dasar
        child = copy.deepcopy(parent1)
        
        # Dapatkan atribut yang ada di kedua induk
        common_attrs = [attr for attr in self.attributes if hasattr(parent1, attr) and hasattr(parent2, attr)]
        
        if not common_attrs or len(common_attrs) <= self.num_points:
            return child
        
        # Pilih titik crossover
        points = sorted(random.sample(range(1, len(common_attrs)), self.num_points))
        
        # Tambahkan titik awal dan akhir
        points = [0] + points + [len(common_attrs)]
        
        # Tukar atribut di antara titik-titik
        for i in range(len(points) - 1):
            # Tentukan apakah segmen ini dari parent1 atau parent2
            from_parent2 = i % 2 == 1
            
            if from_parent2:
                # Tukar atribut dalam segmen ini
                for j in range(points[i], points[i+1]):
                    attr = common_attrs[j]
                    
                    # Dapatkan nilai dari parent2
                    value = getattr(parent2, attr)
                    
                    # Salin nilai ke child
                    if isinstance(value, (int, float, str, bool)):
                        # Untuk tipe data sederhana, salin langsung
                        setattr(child, attr, value)
                    elif isinstance(value, list):
                        # Untuk list, buat salinan
                        setattr(child, attr, copy.deepcopy(value))
                    elif isinstance(value, dict):
                        # Untuk dict, buat salinan
                        setattr(child, attr, copy.deepcopy(value))
        
        # Tukar alat berdasarkan segmen terakhir
        if hasattr(parent1, "tools") and hasattr(parent2, "tools"):
            # Jika segmen terakhir dari parent2, salin alat dari parent2
            if (len(points) - 2) % 2 == 1:
                child.tools = [copy.deepcopy(tool) for tool in parent2.tools]
        
        return child


class BlendCrossover(CrossoverOperator[T]):
    """
    Operator crossover blend (BLX-alpha).
    
    Operator ini menggabungkan parameter numerik dengan interpolasi.
    """
    
    def __init__(self, crossover_rate: float = 0.7, alpha: float = 0.5,
                numeric_attributes: Optional[List[str]] = None):
        """
        Inisialisasi operator crossover blend.
        
        Args:
            crossover_rate: Probabilitas crossover
            alpha: Parameter alpha untuk BLX (0.0 - 1.0)
            numeric_attributes: Daftar atribut numerik yang akan digabungkan
        """
        super().__init__(crossover_rate)
        self.alpha = alpha
        
        # Atribut numerik default jika tidak diberikan
        if numeric_attributes is None:
            numeric_attributes = [
                "learning_rate", "exploration_rate", "memory_capacity",
                "max_iterations", "timeout", "code_generation_capacity"
            ]
        
        self.numeric_attributes = numeric_attributes
    
    def _do_crossover(self, parent1: T, parent2: T) -> T:
        """
        Lakukan crossover blend.
        
        Args:
            parent1: Individu induk pertama
            parent2: Individu induk kedua
            
        Returns:
            Individu anak hasil crossover
        """
        # Buat salinan parent1 sebagai dasar
        child = copy.deepcopy(parent1)
        
        # Gabungkan atribut numerik
        for attr in self.numeric_attributes:
            if hasattr(parent1, attr) and hasattr(parent2, attr):
                value1 = getattr(parent1, attr)
                value2 = getattr(parent2, attr)
                
                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    # Hitung batas interpolasi
                    min_val = min(value1, value2)
                    max_val = max(value1, value2)
                    range_val = max_val - min_val
                    
                    # Perluas batas dengan alpha
                    lower = min_val - self.alpha * range_val
                    upper = max_val + self.alpha * range_val
                    
                    # Interpolasi acak
                    new_value = random.uniform(lower, upper)
                    
                    # Jika atribut asli adalah integer, bulatkan
                    if isinstance(value1, int):
                        new_value = int(round(new_value))
                    
                    # Tetapkan nilai baru
                    setattr(child, attr, new_value)
        
        # Gabungkan atribut non-numerik dengan crossover seragam
        uniform_crossover = UniformCrossover(crossover_rate=1.0)
        child = uniform_crossover._do_crossover(child, parent2)
        
        return child


class SimulatedBinaryCrossover(CrossoverOperator[T]):
    """
    Operator crossover biner tersimulasi (SBX).
    
    Operator ini menggabungkan parameter numerik dengan distribusi
    yang mensimulasikan crossover biner.
    """
    
    def __init__(self, crossover_rate: float = 0.7, eta: float = 15.0,
                numeric_attributes: Optional[List[str]] = None):
        """
        Inisialisasi operator crossover biner tersimulasi.
        
        Args:
            crossover_rate: Probabilitas crossover
            eta: Parameter distribusi (semakin besar, semakin dekat dengan induk)
            numeric_attributes: Daftar atribut numerik yang akan digabungkan
        """
        super().__init__(crossover_rate)
        self.eta = eta
        
        # Atribut numerik default jika tidak diberikan
        if numeric_attributes is None:
            numeric_attributes = [
                "learning_rate", "exploration_rate", "memory_capacity",
                "max_iterations", "timeout", "code_generation_capacity"
            ]
        
        self.numeric_attributes = numeric_attributes
    
    def _do_crossover(self, parent1: T, parent2: T) -> T:
        """
        Lakukan crossover biner tersimulasi.
        
        Args:
            parent1: Individu induk pertama
            parent2: Individu induk kedua
            
        Returns:
            Individu anak hasil crossover
        """
        # Buat salinan parent1 sebagai dasar
        child = copy.deepcopy(parent1)
        
        # Gabungkan atribut numerik
        for attr in self.numeric_attributes:
            if hasattr(parent1, attr) and hasattr(parent2, attr):
                value1 = getattr(parent1, attr)
                value2 = getattr(parent2, attr)
                
                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    # Pastikan value1 <= value2
                    if value1 > value2:
                        value1, value2 = value2, value1
                    
                    # Jika nilai sama, lewati
                    if abs(value1 - value2) < 1e-10:
                        continue
                    
                    # Hasilkan nilai acak
                    u = random.random()
                    
                    # Hitung faktor distribusi
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (self.eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (self.eta + 1))
                    
                    # Hitung nilai anak
                    new_value = 0.5 * ((1 + beta) * value1 + (1 - beta) * value2)
                    
                    # Jika atribut asli adalah integer, bulatkan
                    if isinstance(value1, int):
                        new_value = int(round(new_value))
                    
                    # Tetapkan nilai baru
                    setattr(child, attr, new_value)
        
        # Gabungkan atribut non-numerik dengan crossover seragam
        uniform_crossover = UniformCrossover(crossover_rate=1.0)
        child = uniform_crossover._do_crossover(child, parent2)
        
        return child