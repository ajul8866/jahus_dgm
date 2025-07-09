"""
Strategi evolusi untuk Darwin-Gödel Machine.

Modul ini berisi implementasi berbagai strategi evolusi yang digunakan oleh DGM
untuk menghasilkan agen-agen baru dan meningkatkan populasi agen dari waktu ke waktu.
"""

import random
import math
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, TypeVar, Generic, Union

from simple_dgm.agents.base_agent import BaseAgent

# Tipe generik untuk individu dalam populasi
T = TypeVar('T', bound=BaseAgent)


class EvolutionStrategy(Generic[T]):
    """
    Kelas dasar untuk strategi evolusi.
    
    Strategi evolusi menentukan bagaimana individu dipilih, direproduksi, dan
    dimasukkan kembali ke dalam populasi.
    """
    
    def __init__(self, population_size: int = 100, offspring_size: int = 100):
        """
        Inisialisasi strategi evolusi.
        
        Args:
            population_size: Ukuran populasi
            offspring_size: Jumlah keturunan yang dihasilkan per generasi
        """
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generation = 0
        self.statistics = {
            "best_fitness": [],
            "avg_fitness": [],
            "diversity": []
        }
    
    def select_parents(self, population: List[Tuple[T, float]], k: int = 2) -> List[T]:
        """
        Pilih individu dari populasi untuk menjadi induk.
        
        Args:
            population: Populasi individu dan nilai fitness mereka
            k: Jumlah induk yang akan dipilih
            
        Returns:
            Daftar individu yang dipilih sebagai induk
        """
        raise NotImplementedError("Subclass must implement abstract method")
    
    def create_offspring(self, parents: List[T], 
                        mutation_op: Callable[[T], T], 
                        crossover_op: Optional[Callable[[T, T], T]] = None) -> T:
        """
        Buat keturunan baru dari induk yang dipilih.
        
        Args:
            parents: Daftar individu induk
            mutation_op: Operator mutasi
            crossover_op: Operator crossover (opsional)
            
        Returns:
            Individu keturunan baru
        """
        if crossover_op and len(parents) >= 2:
            # Lakukan crossover
            offspring = crossover_op(parents[0], parents[1])
        else:
            # Salin induk pertama
            offspring = copy.deepcopy(parents[0])
        
        # Lakukan mutasi
        offspring = mutation_op(offspring)
        
        return offspring
    
    def select_survivors(self, population: List[Tuple[T, float]], 
                        offspring: List[Tuple[T, float]]) -> List[Tuple[T, float]]:
        """
        Pilih individu yang akan bertahan ke generasi berikutnya.
        
        Args:
            population: Populasi saat ini
            offspring: Keturunan baru
            
        Returns:
            Populasi baru untuk generasi berikutnya
        """
        raise NotImplementedError("Subclass must implement abstract method")
    
    def update_statistics(self, population: List[Tuple[T, float]], 
                         diversity_metric: Optional[Callable[[List[T]], float]] = None):
        """
        Perbarui statistik evolusi.
        
        Args:
            population: Populasi saat ini
            diversity_metric: Metrik keragaman (opsional)
        """
        if not population:
            return
        
        # Hitung fitness terbaik dan rata-rata
        fitnesses = [f for _, f in population]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        self.statistics["best_fitness"].append(best_fitness)
        self.statistics["avg_fitness"].append(avg_fitness)
        
        # Hitung keragaman jika metrik diberikan
        if diversity_metric:
            individuals = [ind for ind, _ in population]
            diversity = diversity_metric(individuals)
            self.statistics["diversity"].append(diversity)
    
    def evolve(self, population: List[Tuple[T, float]], 
              mutation_op: Callable[[T], T],
              crossover_op: Optional[Callable[[T, T], T]] = None,
              diversity_metric: Optional[Callable[[List[T]], float]] = None,
              elitism: int = 1) -> List[Tuple[T, float]]:
        """
        Evolusi populasi untuk satu generasi.
        
        Args:
            population: Populasi saat ini
            mutation_op: Operator mutasi
            crossover_op: Operator crossover (opsional)
            diversity_metric: Metrik keragaman (opsional)
            elitism: Jumlah individu terbaik yang dipertahankan
            
        Returns:
            Populasi baru untuk generasi berikutnya
        """
        # Tingkatkan nomor generasi
        self.generation += 1
        
        # Buat keturunan
        offspring = []
        for _ in range(self.offspring_size):
            # Pilih induk
            parents = self.select_parents(population)
            
            # Buat keturunan
            child = self.create_offspring(parents, mutation_op, crossover_op)
            
            # Tambahkan ke daftar keturunan (tanpa nilai fitness)
            offspring.append(child)
        
        # Kembalikan populasi baru
        return offspring


class TournamentSelection(EvolutionStrategy[T]):
    """
    Strategi seleksi turnamen.
    
    Dalam seleksi turnamen, k individu dipilih secara acak dari populasi,
    dan individu dengan fitness tertinggi dipilih sebagai induk.
    """
    
    def __init__(self, population_size: int = 100, offspring_size: int = 100, 
                tournament_size: int = 3):
        """
        Inisialisasi strategi seleksi turnamen.
        
        Args:
            population_size: Ukuran populasi
            offspring_size: Jumlah keturunan yang dihasilkan per generasi
            tournament_size: Ukuran turnamen
        """
        super().__init__(population_size, offspring_size)
        self.tournament_size = tournament_size
    
    def select_parents(self, population: List[Tuple[T, float]], k: int = 2) -> List[T]:
        """
        Pilih induk menggunakan seleksi turnamen.
        
        Args:
            population: Populasi individu dan nilai fitness mereka
            k: Jumlah induk yang akan dipilih
            
        Returns:
            Daftar individu yang dipilih sebagai induk
        """
        parents = []
        
        for _ in range(k):
            # Pilih kandidat secara acak
            candidates = random.sample(population, min(self.tournament_size, len(population)))
            
            # Pilih kandidat dengan fitness tertinggi
            winner = max(candidates, key=lambda x: x[1])
            
            # Tambahkan pemenang ke daftar induk
            parents.append(winner[0])
        
        return parents
    
    def select_survivors(self, population: List[Tuple[T, float]], 
                        offspring: List[Tuple[T, float]]) -> List[Tuple[T, float]]:
        """
        Pilih individu yang akan bertahan menggunakan seleksi turnamen.
        
        Args:
            population: Populasi saat ini
            offspring: Keturunan baru
            
        Returns:
            Populasi baru untuk generasi berikutnya
        """
        # Gabungkan populasi dan keturunan
        combined = population + offspring
        
        # Pilih individu terbaik (elitism)
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        elite = sorted_combined[:self.population_size]
        
        return elite


class RouletteWheelSelection(EvolutionStrategy[T]):
    """
    Strategi seleksi roda roulette.
    
    Dalam seleksi roda roulette, probabilitas seleksi individu proporsional
    dengan nilai fitness mereka.
    """
    
    def select_parents(self, population: List[Tuple[T, float]], k: int = 2) -> List[T]:
        """
        Pilih induk menggunakan seleksi roda roulette.
        
        Args:
            population: Populasi individu dan nilai fitness mereka
            k: Jumlah induk yang akan dipilih
            
        Returns:
            Daftar individu yang dipilih sebagai induk
        """
        parents = []
        
        # Hitung total fitness
        total_fitness = sum(f for _, f in population)
        
        if total_fitness <= 0:
            # Jika total fitness tidak positif, pilih secara acak
            return [ind for ind, _ in random.sample(population, k)]
        
        for _ in range(k):
            # Pilih titik acak pada roda
            point = random.uniform(0, total_fitness)
            
            # Temukan individu yang sesuai
            current = 0
            for ind, fitness in population:
                current += fitness
                if current >= point:
                    parents.append(ind)
                    break
            else:
                # Jika tidak ada yang dipilih (karena pembulatan), pilih yang terakhir
                parents.append(population[-1][0])
        
        return parents
    
    def select_survivors(self, population: List[Tuple[T, float]], 
                        offspring: List[Tuple[T, float]]) -> List[Tuple[T, float]]:
        """
        Pilih individu yang akan bertahan menggunakan seleksi roda roulette.
        
        Args:
            population: Populasi saat ini
            offspring: Keturunan baru
            
        Returns:
            Populasi baru untuk generasi berikutnya
        """
        # Gabungkan populasi dan keturunan
        combined = population + offspring
        
        # Pilih individu terbaik (elitism)
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        elite_count = max(1, int(self.population_size * 0.1))
        elite = sorted_combined[:elite_count]
        
        # Pilih sisanya menggunakan roda roulette
        remaining = sorted_combined[elite_count:]
        selected = elite.copy()
        
        # Hitung total fitness
        total_fitness = sum(f for _, f in remaining)
        
        if total_fitness <= 0:
            # Jika total fitness tidak positif, pilih secara acak
            selected.extend(random.sample(remaining, min(self.population_size - elite_count, len(remaining))))
        else:
            while len(selected) < self.population_size and remaining:
                # Pilih titik acak pada roda
                point = random.uniform(0, total_fitness)
                
                # Temukan individu yang sesuai
                current = 0
                for i, (ind, fitness) in enumerate(remaining):
                    current += fitness
                    if current >= point:
                        selected.append((ind, fitness))
                        total_fitness -= fitness
                        remaining.pop(i)
                        break
                else:
                    # Jika tidak ada yang dipilih (karena pembulatan), pilih yang terakhir
                    selected.append(remaining.pop())
        
        return selected


class NSGA2Selection(EvolutionStrategy[T]):
    """
    Strategi seleksi NSGA-II (Non-dominated Sorting Genetic Algorithm II).
    
    NSGA-II adalah algoritma evolusi multi-objektif yang menggunakan
    pengurutan non-dominasi dan jarak kerumunan untuk memilih individu.
    """
    
    def __init__(self, population_size: int = 100, offspring_size: int = 100):
        """
        Inisialisasi strategi seleksi NSGA-II.
        
        Args:
            population_size: Ukuran populasi
            offspring_size: Jumlah keturunan yang dihasilkan per generasi
        """
        super().__init__(population_size, offspring_size)
    
    def _fast_non_dominated_sort(self, population: List[Tuple[T, List[float]]]) -> List[List[int]]:
        """
        Lakukan pengurutan non-dominasi cepat.
        
        Args:
            population: Populasi individu dan nilai fitness multi-objektif mereka
            
        Returns:
            Daftar front Pareto
        """
        n = len(population)
        domination_count = [0] * n  # Jumlah individu yang mendominasi individu i
        dominated_set = [[] for _ in range(n)]  # Daftar individu yang didominasi oleh individu i
        fronts = [[]]  # Front Pareto
        
        # Untuk setiap individu
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                # Periksa dominasi
                i_dominates_j = False
                j_dominates_i = False
                
                # Individu i mendominasi j jika i lebih baik atau sama dalam semua objektif
                # dan lebih baik dalam setidaknya satu objektif
                i_fitness = population[i][1]
                j_fitness = population[j][1]
                
                if all(i_val >= j_val for i_val, j_val in zip(i_fitness, j_fitness)) and \
                   any(i_val > j_val for i_val, j_val in zip(i_fitness, j_fitness)):
                    i_dominates_j = True
                    dominated_set[i].append(j)
                elif all(j_val >= i_val for i_val, j_val in zip(i_fitness, j_fitness)) and \
                     any(j_val > i_val for i_val, j_val in zip(i_fitness, j_fitness)):
                    j_dominates_i = True
                    domination_count[i] += 1
            
            # Jika tidak ada yang mendominasi i, tambahkan ke front pertama
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Buat front berikutnya
        i = 0
        while fronts[i]:
            next_front = []
            
            for j in fronts[i]:
                for k in dominated_set[j]:
                    domination_count[k] -= 1
                    if domination_count[k] == 0:
                        next_front.append(k)
            
            i += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def _calculate_crowding_distance(self, population: List[Tuple[T, List[float]]], front: List[int]) -> List[float]:
        """
        Hitung jarak kerumunan untuk individu dalam front.
        
        Args:
            population: Populasi individu dan nilai fitness multi-objektif mereka
            front: Indeks individu dalam front
            
        Returns:
            Jarak kerumunan untuk setiap individu dalam front
        """
        n = len(front)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        
        # Untuk setiap objektif
        for m in range(len(population[0][1])):
            # Urutkan front berdasarkan objektif m
            sorted_front = sorted(front, key=lambda i: population[i][1][m])
            
            # Tetapkan jarak tak terhingga untuk individu di ujung
            distances[0] = float('inf')
            distances[-1] = float('inf')
            
            # Hitung jarak untuk individu di tengah
            f_max = population[sorted_front[-1]][1][m]
            f_min = population[sorted_front[0]][1][m]
            
            if f_max == f_min:
                continue
            
            for i in range(1, n - 1):
                distances[i] += (population[sorted_front[i+1]][1][m] - population[sorted_front[i-1]][1][m]) / (f_max - f_min)
        
        return distances
    
    def select_parents(self, population: List[Tuple[T, Union[float, List[float]]]], k: int = 2) -> List[T]:
        """
        Pilih induk menggunakan seleksi turnamen berdasarkan peringkat dan jarak kerumunan.
        
        Args:
            population: Populasi individu dan nilai fitness mereka
            k: Jumlah induk yang akan dipilih
            
        Returns:
            Daftar individu yang dipilih sebagai induk
        """
        # Konversi fitness tunggal menjadi daftar jika perlu
        multi_obj_population = []
        for ind, fitness in population:
            if isinstance(fitness, (int, float)):
                multi_obj_population.append((ind, [fitness]))
            else:
                multi_obj_population.append((ind, fitness))
        
        # Lakukan pengurutan non-dominasi
        fronts = self._fast_non_dominated_sort(multi_obj_population)
        
        # Hitung jarak kerumunan untuk setiap front
        crowding_distances = {}
        for i, front in enumerate(fronts):
            distances = self._calculate_crowding_distance(multi_obj_population, front)
            for j, idx in enumerate(front):
                crowding_distances[idx] = (i, distances[j])
        
        # Pilih induk menggunakan seleksi turnamen
        parents = []
        for _ in range(k):
            # Pilih dua kandidat secara acak
            candidates = random.sample(range(len(multi_obj_population)), 2)
            
            # Pilih kandidat dengan peringkat lebih baik atau jarak kerumunan lebih besar
            if crowding_distances[candidates[0]][0] < crowding_distances[candidates[1]][0] or \
               (crowding_distances[candidates[0]][0] == crowding_distances[candidates[1]][0] and 
                crowding_distances[candidates[0]][1] > crowding_distances[candidates[1]][1]):
                winner = candidates[0]
            else:
                winner = candidates[1]
            
            parents.append(multi_obj_population[winner][0])
        
        return parents
    
    def select_survivors(self, population: List[Tuple[T, Union[float, List[float]]]], 
                        offspring: List[Tuple[T, Union[float, List[float]]]]) -> List[Tuple[T, Union[float, List[float]]]]:
        """
        Pilih individu yang akan bertahan menggunakan NSGA-II.
        
        Args:
            population: Populasi saat ini
            offspring: Keturunan baru
            
        Returns:
            Populasi baru untuk generasi berikutnya
        """
        # Konversi fitness tunggal menjadi daftar jika perlu
        multi_obj_combined = []
        for ind, fitness in population + offspring:
            if isinstance(fitness, (int, float)):
                multi_obj_combined.append((ind, [fitness]))
            else:
                multi_obj_combined.append((ind, fitness))
        
        # Lakukan pengurutan non-dominasi
        fronts = self._fast_non_dominated_sort(multi_obj_combined)
        
        # Pilih individu berdasarkan front dan jarak kerumunan
        next_population = []
        i = 0
        
        while len(next_population) + len(fronts[i]) <= self.population_size:
            # Tambahkan seluruh front
            for idx in fronts[i]:
                next_population.append((multi_obj_combined[idx][0], multi_obj_combined[idx][1]))
            i += 1
            
            if i >= len(fronts):
                break
        
        # Jika masih ada ruang, tambahkan individu dari front berikutnya berdasarkan jarak kerumunan
        if len(next_population) < self.population_size and i < len(fronts):
            # Hitung jarak kerumunan
            distances = self._calculate_crowding_distance(multi_obj_combined, fronts[i])
            
            # Urutkan front berdasarkan jarak kerumunan
            sorted_front = sorted([(idx, dist) for idx, dist in zip(fronts[i], distances)], 
                                 key=lambda x: x[1], reverse=True)
            
            # Tambahkan individu hingga populasi penuh
            for idx, _ in sorted_front[:self.population_size - len(next_population)]:
                next_population.append((multi_obj_combined[idx][0], multi_obj_combined[idx][1]))
        
        # Konversi kembali ke format fitness asli
        final_population = []
        for ind, fitness in next_population:
            if len(fitness) == 1 and isinstance(population[0][1], (int, float)):
                final_population.append((ind, fitness[0]))
            else:
                final_population.append((ind, fitness))
        
        return final_population


class MAP_ElitesStrategy(EvolutionStrategy[T]):
    """
    Strategi MAP-Elites (Multi-dimensional Archive of Phenotypic Elites).
    
    MAP-Elites adalah algoritma evolusi yang mempertahankan arsip individu elit
    dalam ruang fitur multi-dimensi, memungkinkan eksplorasi yang lebih luas.
    """
    
    def __init__(self, population_size: int = 100, offspring_size: int = 100, 
                feature_dimensions: List[Tuple[float, float, int]] = None):
        """
        Inisialisasi strategi MAP-Elites.
        
        Args:
            population_size: Ukuran populasi
            offspring_size: Jumlah keturunan yang dihasilkan per generasi
            feature_dimensions: Daftar dimensi fitur (min, max, bins)
        """
        super().__init__(population_size, offspring_size)
        
        # Dimensi fitur default jika tidak diberikan
        if feature_dimensions is None:
            feature_dimensions = [(0.0, 1.0, 10), (0.0, 1.0, 10)]
        
        self.feature_dimensions = feature_dimensions
        self.num_dimensions = len(feature_dimensions)
        
        # Buat grid untuk menyimpan individu elit
        self.grid = {}
        
        # Fungsi untuk mengekstrak fitur dari individu
        self.feature_extractor = None
    
    def set_feature_extractor(self, extractor: Callable[[T], List[float]]):
        """
        Tetapkan fungsi untuk mengekstrak fitur dari individu.
        
        Args:
            extractor: Fungsi yang mengekstrak fitur dari individu
        """
        self.feature_extractor = extractor
    
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
    
    def select_parents(self, population: List[Tuple[T, float]], k: int = 2) -> List[T]:
        """
        Pilih induk dari grid MAP-Elites.
        
        Args:
            population: Populasi individu dan nilai fitness mereka
            k: Jumlah induk yang akan dipilih
            
        Returns:
            Daftar individu yang dipilih sebagai induk
        """
        parents = []
        
        # Jika grid kosong, gunakan populasi
        if not self.grid:
            return [ind for ind, _ in random.sample(population, min(k, len(population)))]
        
        # Pilih sel secara acak dari grid
        cells = list(self.grid.keys())
        
        for _ in range(k):
            cell = random.choice(cells)
            parents.append(self.grid[cell][0])
        
        return parents
    
    def select_survivors(self, population: List[Tuple[T, float]], 
                        offspring: List[Tuple[T, float]]) -> List[Tuple[T, float]]:
        """
        Perbarui grid MAP-Elites dengan individu baru.
        
        Args:
            population: Populasi saat ini
            offspring: Keturunan baru
            
        Returns:
            Populasi baru untuk generasi berikutnya
        """
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not set")
        
        # Gabungkan populasi dan keturunan
        combined = population + offspring
        
        # Perbarui grid
        for ind, fitness in combined:
            features = self.feature_extractor(ind)
            cell_index = self._get_cell_index(features)
            
            # Jika sel kosong atau individu baru lebih baik, perbarui sel
            if cell_index not in self.grid or fitness > self.grid[cell_index][1]:
                self.grid[cell_index] = (ind, fitness)
        
        # Konversi grid ke populasi
        new_population = [(ind, fitness) for ind, fitness in self.grid.values()]
        
        # Jika populasi terlalu kecil, tambahkan individu acak dari combined
        if len(new_population) < self.population_size:
            remaining = [item for item in combined if item[0] not in [ind for ind, _ in new_population]]
            new_population.extend(random.sample(remaining, min(self.population_size - len(new_population), len(remaining))))
        
        # Jika populasi terlalu besar, potong
        if len(new_population) > self.population_size:
            new_population = random.sample(new_population, self.population_size)
        
        return new_population


class CMA_ESStrategy(EvolutionStrategy[T]):
    """
    Strategi CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    CMA-ES adalah algoritma evolusi yang mengadaptasi matriks kovarians
    untuk mengoptimalkan distribusi pencarian.
    """
    
    def __init__(self, population_size: int = 100, offspring_size: int = 100, 
                initial_sigma: float = 1.0, parameter_count: int = 10):
        """
        Inisialisasi strategi CMA-ES.
        
        Args:
            population_size: Ukuran populasi
            offspring_size: Jumlah keturunan yang dihasilkan per generasi
            initial_sigma: Deviasi standar awal
            parameter_count: Jumlah parameter yang dioptimalkan
        """
        super().__init__(population_size, offspring_size)
        
        # Parameter CMA-ES
        self.sigma = initial_sigma  # Ukuran langkah global
        self.n = parameter_count  # Dimensi
        
        # Inisialisasi parameter
        self.mean = np.zeros(self.n)  # Vektor mean
        self.C = np.eye(self.n)  # Matriks kovarians
        self.B = np.eye(self.n)  # Matriks eigenvector
        self.D = np.ones(self.n)  # Eigenvalue
        
        # Parameter adaptasi
        self.pc = np.zeros(self.n)  # Evolution path for C
        self.ps = np.zeros(self.n)  # Evolution path for sigma
        
        # Konstanta
        self.mu = int(self.offspring_size / 2)  # Jumlah induk
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)  # Normalisasi bobot
        self.mueff = 1 / np.sum(self.weights ** 2)  # Variance effective selection mass
        
        # Parameter laju pembelajaran
        self.cc = 4 / (self.n + 4)  # Time constant for cumulation for C
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)  # Time constant for cumulation for sigma
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)  # Learning rate for rank-one update
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.n + 2) ** 2 + self.mueff))  # Learning rate for rank-mu update
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs  # Damping for sigma
        
        # Fungsi untuk mengekstrak parameter dari individu
        self.parameter_extractor = None
        self.parameter_injector = None
    
    def set_parameter_handlers(self, extractor: Callable[[T], np.ndarray], 
                              injector: Callable[[T, np.ndarray], T]):
        """
        Tetapkan fungsi untuk mengekstrak dan menyuntikkan parameter.
        
        Args:
            extractor: Fungsi yang mengekstrak parameter dari individu
            injector: Fungsi yang menyuntikkan parameter ke individu
        """
        self.parameter_extractor = extractor
        self.parameter_injector = injector
    
    def select_parents(self, population: List[Tuple[T, float]], k: int = 2) -> List[T]:
        """
        Pilih induk menggunakan CMA-ES.
        
        Args:
            population: Populasi individu dan nilai fitness mereka
            k: Jumlah induk yang akan dipilih
            
        Returns:
            Daftar individu yang dipilih sebagai induk
        """
        # Dalam CMA-ES, kita tidak memilih induk secara tradisional
        # Sebagai gantinya, kita menghasilkan keturunan dari distribusi
        
        # Jika parameter handler belum diatur, gunakan seleksi acak
        if self.parameter_extractor is None or self.parameter_injector is None:
            return [ind for ind, _ in random.sample(population, min(k, len(population)))]
        
        # Buat keturunan baru menggunakan distribusi CMA-ES
        parents = []
        
        for _ in range(k):
            # Hasilkan vektor acak dari distribusi normal
            z = np.random.randn(self.n)
            
            # Transformasi ke distribusi CMA-ES
            x = self.mean + self.sigma * np.dot(self.B, self.D * z)
            
            # Buat individu baru dengan parameter ini
            # Gunakan individu pertama sebagai template
            template = population[0][0]
            new_ind = self.parameter_injector(copy.deepcopy(template), x)
            
            parents.append(new_ind)
        
        return parents
    
    def select_survivors(self, population: List[Tuple[T, float]], 
                        offspring: List[Tuple[T, float]]) -> List[Tuple[T, float]]:
        """
        Perbarui parameter CMA-ES dan pilih individu yang akan bertahan.
        
        Args:
            population: Populasi saat ini
            offspring: Keturunan baru
            
        Returns:
            Populasi baru untuk generasi berikutnya
        """
        if self.parameter_extractor is None:
            # Jika parameter handler belum diatur, gunakan seleksi elitis
            combined = population + offspring
            sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
            return sorted_combined[:self.population_size]
        
        # Urutkan keturunan berdasarkan fitness
        sorted_offspring = sorted(offspring, key=lambda x: x[1], reverse=True)
        
        # Pilih mu individu terbaik
        selected = sorted_offspring[:self.mu]
        
        # Ekstrak parameter dari individu terpilih
        x_selected = np.array([self.parameter_extractor(ind) for ind, _ in selected])
        
        # Hitung mean baru dengan pembobotan
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * x_selected, axis=0)
        
        # Perbarui evolution path
        y = self.mean - old_mean
        z = np.dot(np.linalg.inv(np.dot(self.B, np.diag(self.D))), y) / self.sigma
        
        # Perbarui evolution path untuk sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z
        
        # Perbarui evolution path untuk C
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.generation)) < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        
        # Perbarui matriks kovarians
        artmp = (1 / self.sigma) * (x_selected - old_mean)
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + \
                self.cmu * np.sum(self.weights[:, np.newaxis, np.newaxis] * 
                                np.array([np.outer(artmp[i], artmp[i]) for i in range(self.mu)]), axis=0)
        
        # Perbarui sigma
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.n) - 1))
        
        # Dekomposisi eigenvalue dari C
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(self.D)
        
        # Buat populasi baru
        new_population = []
        
        # Tambahkan individu terbaik dari keturunan
        new_population.extend(selected)
        
        # Hasilkan individu baru jika perlu
        while len(new_population) < self.population_size:
            # Hasilkan vektor acak dari distribusi normal
            z = np.random.randn(self.n)
            
            # Transformasi ke distribusi CMA-ES
            x = self.mean + self.sigma * np.dot(self.B, self.D * z)
            
            # Buat individu baru dengan parameter ini
            template = population[0][0]
            new_ind = self.parameter_injector(copy.deepcopy(template), x)
            
            # Evaluasi individu baru (gunakan fungsi evaluasi dari luar)
            # Untuk sementara, tetapkan fitness ke 0
            new_population.append((new_ind, 0.0))
        
        return new_population[:self.population_size]