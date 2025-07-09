"""
Implementasi kompleks Darwin-Gödel Machine untuk optimasi portofolio keuangan.

Contoh ini mendemonstrasikan penggunaan DGM dengan integrasi LLM untuk:
1. Analisis sentimen pasar dari berita keuangan
2. Optimasi portofolio multi-objektif
3. Prediksi pergerakan harga aset
4. Manajemen risiko adaptif
5. Evolusi strategi trading berdasarkan kondisi pasar
"""

import os
import random
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.core.llm_integration_real import (
    LLMInterface, CodeGeneration, ProblemSolving, 
    KnowledgeExtraction, SelfModification
)
from simple_dgm.core.evolution_strategies import TournamentSelection
from simple_dgm.core.mutation_operators import ParameterMutation
from simple_dgm.core.crossover_operators import BlendCrossover
from simple_dgm.core.fitness_functions import MultiObjectiveFitness
from simple_dgm.core.diversity_metrics import BehavioralDiversity
from simple_dgm.core.archive_strategies import QualityDiversityArchive

# Konstanta untuk simulasi pasar
ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "PG"]
RISK_FREE_RATE = 0.02  # 2% risk-free rate
MARKET_VOLATILITY = 0.15  # 15% volatility pasar
MAX_LEVERAGE = 2.0  # Maksimum leverage 2x
TRANSACTION_COST = 0.001  # 0.1% biaya transaksi

# Kelas untuk simulasi data pasar
class MarketSimulator:
    """
    Simulator pasar keuangan untuk pengujian strategi trading.
    """
    
    def __init__(self, assets: List[str], start_date: str, end_date: str, seed: Optional[int] = None):
        """
        Inisialisasi simulator pasar.
        
        Args:
            assets: Daftar aset yang akan disimulasikan
            start_date: Tanggal mulai simulasi (format: 'YYYY-MM-DD')
            end_date: Tanggal akhir simulasi (format: 'YYYY-MM-DD')
            seed: Seed untuk generator angka acak (opsional)
        """
        self.assets = assets
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.current_date = self.start_date
        
        # Set seed untuk reprodusibilitas
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Inisialisasi data pasar
        self._initialize_market_data()
        
        # Inisialisasi berita pasar
        self._initialize_market_news()
    
    def _initialize_market_data(self):
        """
        Inisialisasi data pasar dengan simulasi pergerakan harga.
        """
        # Hitung jumlah hari dalam periode simulasi
        days = (self.end_date - self.start_date).days + 1
        
        # Inisialisasi DataFrame untuk harga
        dates = [self.start_date + timedelta(days=i) for i in range(days)]
        self.price_data = pd.DataFrame(index=dates, columns=self.assets)
        
        # Inisialisasi harga awal
        initial_prices = {}
        for asset in self.assets:
            # Harga awal acak antara $50 dan $500
            initial_prices[asset] = np.random.uniform(50, 500)
        
        # Simulasi pergerakan harga dengan Geometric Brownian Motion
        for asset in self.assets:
            # Parameter untuk GBM
            mu = np.random.uniform(0.05, 0.15)  # Return tahunan
            sigma = np.random.uniform(0.1, 0.4)  # Volatilitas tahunan
            
            # Konversi parameter tahunan ke harian
            daily_mu = mu / 252
            daily_sigma = sigma / np.sqrt(252)
            
            # Simulasi pergerakan harga
            price = initial_prices[asset]
            prices = [price]
            
            for i in range(1, days):
                # Geometric Brownian Motion
                daily_return = np.random.normal(daily_mu, daily_sigma)
                price = price * (1 + daily_return)
                prices.append(price)
            
            # Simpan harga ke DataFrame
            self.price_data[asset] = prices
        
        # Inisialisasi DataFrame untuk volume
        self.volume_data = pd.DataFrame(index=dates, columns=self.assets)
        
        # Simulasi volume perdagangan
        for asset in self.assets:
            # Volume dasar
            base_volume = np.random.uniform(1e6, 1e7)
            
            # Simulasi volume harian
            volumes = []
            for i in range(days):
                # Volume bervariasi dengan faktor acak
                volume = base_volume * np.random.lognormal(0, 0.5)
                volumes.append(volume)
            
            # Simpan volume ke DataFrame
            self.volume_data[asset] = volumes
    
    def _initialize_market_news(self):
        """
        Inisialisasi berita pasar dengan simulasi sentimen.
        """
        # Hitung jumlah hari dalam periode simulasi
        days = (self.end_date - self.start_date).days + 1
        
        # Inisialisasi DataFrame untuk berita
        dates = [self.start_date + timedelta(days=i) for i in range(days)]
        self.news_data = pd.DataFrame(index=dates, columns=['headline', 'sentiment'])
        
        # Template berita
        positive_templates = [
            "Markets rally as {asset} reports strong earnings",
            "Investors optimistic about {asset}'s new product launch",
            "Analysts upgrade {asset} citing growth potential",
            "{asset} exceeds expectations in quarterly report",
            "Economic indicators point to strong growth for {sector}",
            "{asset} announces strategic partnership with {partner}",
            "Fed signals continued support for markets",
            "Consumer confidence rises, boosting retail stocks",
            "Inflation data comes in lower than expected",
            "Trade tensions ease, markets respond positively"
        ]
        
        negative_templates = [
            "Markets tumble as {asset} misses earnings expectations",
            "Investors concerned about {asset}'s growth prospects",
            "Analysts downgrade {asset} citing competitive pressures",
            "{asset} faces regulatory scrutiny over business practices",
            "Economic indicators signal slowdown in {sector}",
            "{asset} announces layoffs amid restructuring",
            "Fed signals potential rate hikes, markets react",
            "Consumer spending falls, impacting retail stocks",
            "Inflation data higher than expected, raising concerns",
            "Trade tensions escalate, markets respond negatively"
        ]
        
        neutral_templates = [
            "{asset} reports earnings in line with expectations",
            "Markets mixed as investors assess economic data",
            "Analysts maintain neutral stance on {asset}",
            "{asset} announces expected quarterly results",
            "Economic indicators show stable growth in {sector}",
            "{asset} holds annual shareholder meeting",
            "Fed maintains current monetary policy",
            "Consumer confidence remains stable",
            "Inflation data in line with expectations",
            "Trade negotiations continue with no major developments"
        ]
        
        sectors = ["technology", "finance", "healthcare", "energy", "consumer goods"]
        partners = ["Microsoft", "Amazon", "Google", "Apple", "Meta", "IBM", "Oracle", "Salesforce"]
        
        # Simulasi berita harian
        for i, date in enumerate(dates):
            # Pilih template berdasarkan sentimen
            sentiment = np.random.choice(["positive", "negative", "neutral"], p=[0.4, 0.3, 0.3])
            
            if sentiment == "positive":
                template = random.choice(positive_templates)
                sentiment_score = np.random.uniform(0.6, 1.0)
            elif sentiment == "negative":
                template = random.choice(negative_templates)
                sentiment_score = np.random.uniform(0.0, 0.4)
            else:
                template = random.choice(neutral_templates)
                sentiment_score = np.random.uniform(0.4, 0.6)
            
            # Pilih aset atau sektor acak
            if "{asset}" in template:
                asset = random.choice(self.assets)
                headline = template.replace("{asset}", asset)
            elif "{sector}" in template:
                sector = random.choice(sectors)
                headline = template.replace("{sector}", sector)
            else:
                headline = template
            
            # Ganti {partner} jika ada
            if "{partner}" in headline:
                partner = random.choice(partners)
                headline = headline.replace("{partner}", partner)
            
            # Simpan berita ke DataFrame
            self.news_data.loc[date, 'headline'] = headline
            self.news_data.loc[date, 'sentiment'] = sentiment_score
    
    def get_current_data(self) -> Dict[str, Any]:
        """
        Dapatkan data pasar saat ini.
        
        Returns:
            Data pasar saat ini
        """
        # Dapatkan indeks untuk tanggal saat ini
        date_idx = self.current_date
        
        # Dapatkan harga dan volume untuk tanggal saat ini
        prices = {}
        volumes = {}
        for asset in self.assets:
            if date_idx in self.price_data.index:
                prices[asset] = self.price_data.loc[date_idx, asset]
                volumes[asset] = self.volume_data.loc[date_idx, asset]
            else:
                prices[asset] = None
                volumes[asset] = None
        
        # Dapatkan berita untuk tanggal saat ini
        if date_idx in self.news_data.index:
            headline = self.news_data.loc[date_idx, 'headline']
            sentiment = self.news_data.loc[date_idx, 'sentiment']
        else:
            headline = None
            sentiment = None
        
        return {
            "date": self.current_date.strftime('%Y-%m-%d'),
            "prices": prices,
            "volumes": volumes,
            "headline": headline,
            "sentiment": sentiment
        }
    
    def advance_day(self) -> bool:
        """
        Maju satu hari dalam simulasi.
        
        Returns:
            True jika simulasi masih berlanjut, False jika simulasi telah berakhir
        """
        # Maju satu hari
        self.current_date += timedelta(days=1)
        
        # Periksa apakah simulasi telah berakhir
        if self.current_date > self.end_date:
            return False
        
        return True
    
    def reset(self):
        """
        Reset simulasi ke tanggal awal.
        """
        self.current_date = self.start_date

# Kelas untuk portofolio
class Portfolio:
    """
    Portofolio keuangan untuk simulasi trading.
    """
    
    def __init__(self, initial_cash: float = 1000000.0):
        """
        Inisialisasi portofolio.
        
        Args:
            initial_cash: Jumlah uang tunai awal
        """
        self.cash = initial_cash
        self.positions = {}  # {asset: jumlah}
        self.transaction_history = []
        self.value_history = []
    
    def calculate_value(self, prices: Dict[str, float]) -> float:
        """
        Hitung nilai portofolio.
        
        Args:
            prices: Harga aset saat ini
            
        Returns:
            Nilai portofolio
        """
        value = self.cash
        
        for asset, quantity in self.positions.items():
            if asset in prices and prices[asset] is not None:
                value += quantity * prices[asset]
        
        return value
    
    def execute_trade(self, asset: str, quantity: float, price: float, date: str):
        """
        Eksekusi perdagangan.
        
        Args:
            asset: Aset yang akan diperdagangkan
            quantity: Jumlah aset (positif untuk beli, negatif untuk jual)
            price: Harga aset
            date: Tanggal perdagangan
        """
        # Hitung biaya transaksi
        transaction_cost = abs(quantity * price * TRANSACTION_COST)
        
        # Hitung total nilai transaksi
        transaction_value = quantity * price + transaction_cost
        
        # Periksa apakah ada cukup uang tunai untuk pembelian
        if quantity > 0 and transaction_value > self.cash:
            # Sesuaikan jumlah berdasarkan uang tunai yang tersedia
            max_quantity = (self.cash - transaction_cost) / price
            quantity = max(0, max_quantity)
            transaction_value = quantity * price + transaction_cost
        
        # Periksa apakah ada cukup aset untuk penjualan
        if quantity < 0:
            current_quantity = self.positions.get(asset, 0)
            if abs(quantity) > current_quantity:
                quantity = -current_quantity
            transaction_value = quantity * price + transaction_cost
        
        # Perbarui uang tunai
        self.cash -= transaction_value
        
        # Perbarui posisi
        if asset not in self.positions:
            self.positions[asset] = 0
        self.positions[asset] += quantity
        
        # Hapus posisi jika jumlahnya nol
        if self.positions[asset] == 0:
            del self.positions[asset]
        
        # Catat transaksi
        self.transaction_history.append({
            "date": date,
            "asset": asset,
            "quantity": quantity,
            "price": price,
            "transaction_cost": transaction_cost,
            "transaction_value": transaction_value
        })
    
    def rebalance(self, target_weights: Dict[str, float], prices: Dict[str, float], date: str):
        """
        Rebalance portofolio berdasarkan bobot target.
        
        Args:
            target_weights: Bobot target untuk setiap aset
            prices: Harga aset saat ini
            date: Tanggal rebalancing
        """
        # Hitung nilai portofolio saat ini
        portfolio_value = self.calculate_value(prices)
        
        # Hitung nilai target untuk setiap aset
        target_values = {}
        for asset, weight in target_weights.items():
            target_values[asset] = portfolio_value * weight
        
        # Hitung nilai saat ini untuk setiap aset
        current_values = {}
        for asset in target_weights.keys():
            quantity = self.positions.get(asset, 0)
            if asset in prices and prices[asset] is not None:
                current_values[asset] = quantity * prices[asset]
            else:
                current_values[asset] = 0
        
        # Hitung perbedaan nilai
        value_diffs = {}
        for asset in target_weights.keys():
            value_diffs[asset] = target_values[asset] - current_values[asset]
        
        # Eksekusi perdagangan untuk rebalancing
        for asset, value_diff in value_diffs.items():
            if asset in prices and prices[asset] is not None and abs(value_diff) > 100:  # Minimal $100 untuk perdagangan
                quantity = value_diff / prices[asset]
                self.execute_trade(asset, quantity, prices[asset], date)
    
    def get_summary(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Dapatkan ringkasan portofolio.
        
        Args:
            prices: Harga aset saat ini
            
        Returns:
            Ringkasan portofolio
        """
        # Hitung nilai portofolio
        portfolio_value = self.calculate_value(prices)
        
        # Hitung nilai posisi
        positions_value = {}
        for asset, quantity in self.positions.items():
            if asset in prices and prices[asset] is not None:
                positions_value[asset] = quantity * prices[asset]
            else:
                positions_value[asset] = 0
        
        # Hitung bobot posisi
        weights = {}
        for asset, value in positions_value.items():
            weights[asset] = value / portfolio_value if portfolio_value > 0 else 0
        
        return {
            "cash": self.cash,
            "positions": self.positions,
            "positions_value": positions_value,
            "weights": weights,
            "portfolio_value": portfolio_value
        }

# Fungsi alat untuk agen
def analyze_sentiment(headline: str, llm_interface: LLMInterface) -> float:
    """
    Analisis sentimen berita keuangan.
    
    Args:
        headline: Judul berita
        llm_interface: Antarmuka LLM
        
    Returns:
        Skor sentimen (0.0 - 1.0, di mana 0.0 sangat negatif dan 1.0 sangat positif)
    """
    if not headline:
        return 0.5  # Netral jika tidak ada berita
    
    prompt = f"""
    Analyze the sentiment of the following financial news headline:
    "{headline}"
    
    Please provide a sentiment score between 0.0 (extremely negative) and 1.0 (extremely positive).
    Your response should be in JSON format with the following structure:
    {{
        "sentiment_score": float,
        "reasoning": "Your reasoning for the sentiment score"
    }}
    """
    
    response = llm_interface.query_json(prompt)
    
    if "sentiment_score" in response:
        return response["sentiment_score"]
    else:
        # Fallback jika LLM gagal
        return 0.5

def calculate_expected_return(prices: Dict[str, float], historical_prices: pd.DataFrame, sentiment_score: float) -> Dict[str, float]:
    """
    Hitung expected return untuk setiap aset.
    
    Args:
        prices: Harga aset saat ini
        historical_prices: Harga historis aset
        sentiment_score: Skor sentimen pasar
        
    Returns:
        Expected return untuk setiap aset
    """
    expected_returns = {}
    
    # Hitung historical returns
    for asset in prices.keys():
        if asset in historical_prices.columns:
            # Hitung return harian
            returns = historical_prices[asset].pct_change().dropna()
            
            # Hitung mean dan std dari returns
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Sesuaikan expected return berdasarkan sentimen
            sentiment_adjustment = (sentiment_score - 0.5) * 2  # -1.0 to 1.0
            adjusted_return = mean_return + (sentiment_adjustment * std_return * 0.5)
            
            expected_returns[asset] = adjusted_return
        else:
            expected_returns[asset] = 0.0
    
    return expected_returns

def calculate_risk(historical_prices: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Hitung risiko untuk setiap aset dan matriks kovarians.
    
    Args:
        historical_prices: Harga historis aset
        
    Returns:
        Tuple (volatilitas untuk setiap aset, matriks kovarians)
    """
    # Hitung returns
    returns = historical_prices.pct_change().dropna()
    
    # Hitung volatilitas (standar deviasi)
    volatility = {}
    for asset in returns.columns:
        volatility[asset] = returns[asset].std()
    
    # Hitung matriks kovarians
    cov_matrix = returns.cov()
    
    return volatility, cov_matrix

def optimize_portfolio(expected_returns: Dict[str, float], cov_matrix: pd.DataFrame, risk_aversion: float = 1.0) -> Dict[str, float]:
    """
    Optimasi portofolio menggunakan Mean-Variance Optimization.
    
    Args:
        expected_returns: Expected return untuk setiap aset
        cov_matrix: Matriks kovarians
        risk_aversion: Parameter risk aversion (lambda)
        
    Returns:
        Bobot optimal untuk setiap aset
    """
    # Konversi expected returns ke array
    assets = list(expected_returns.keys())
    returns_array = np.array([expected_returns[asset] for asset in assets])
    
    # Ekstrak matriks kovarians untuk aset yang relevan
    cov_subset = cov_matrix.loc[assets, assets].values
    
    # Tambahkan regularisasi kecil untuk memastikan matriks definit positif
    cov_subset = cov_subset + np.eye(len(assets)) * 1e-8
    
    # Hitung bobot optimal
    try:
        # Invers matriks kovarians
        inv_cov = np.linalg.inv(cov_subset)
        
        # Hitung bobot optimal
        weights = np.dot(inv_cov, returns_array) / risk_aversion
        
        # Normalisasi bobot
        weights = weights / np.sum(np.abs(weights)) * MAX_LEVERAGE
        
        # Konversi ke dictionary
        optimal_weights = {asset: weight for asset, weight in zip(assets, weights)}
        
        return optimal_weights
    except np.linalg.LinAlgError:
        # Fallback jika ada masalah dengan invers matriks
        equal_weight = 1.0 / len(assets)
        return {asset: equal_weight for asset in assets}

def calculate_portfolio_metrics(weights: Dict[str, float], expected_returns: Dict[str, float], cov_matrix: pd.DataFrame) -> Dict[str, float]:
    """
    Hitung metrik portofolio.
    
    Args:
        weights: Bobot aset
        expected_returns: Expected return untuk setiap aset
        cov_matrix: Matriks kovarians
        
    Returns:
        Metrik portofolio
    """
    # Konversi weights dan expected returns ke array
    assets = list(weights.keys())
    weights_array = np.array([weights[asset] for asset in assets])
    returns_array = np.array([expected_returns[asset] for asset in assets])
    
    # Ekstrak matriks kovarians untuk aset yang relevan
    cov_subset = cov_matrix.loc[assets, assets].values
    
    # Hitung expected return portofolio
    portfolio_return = np.dot(weights_array, returns_array)
    
    # Hitung volatilitas portofolio
    portfolio_volatility = np.sqrt(np.dot(weights_array, np.dot(cov_subset, weights_array)))
    
    # Hitung Sharpe ratio
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    # Hitung diversifikasi (Herfindahl-Hirschman Index)
    hhi = np.sum(weights_array ** 2)
    diversification = 1 - hhi
    
    return {
        "expected_return": portfolio_return,
        "volatility": portfolio_volatility,
        "sharpe_ratio": sharpe_ratio,
        "diversification": diversification
    }

def generate_trading_strategy(market_data: Dict[str, Any], portfolio_summary: Dict[str, Any], llm_interface: LLMInterface) -> Dict[str, Any]:
    """
    Hasilkan strategi trading berdasarkan data pasar dan portofolio.
    
    Args:
        market_data: Data pasar saat ini
        portfolio_summary: Ringkasan portofolio
        llm_interface: Antarmuka LLM
        
    Returns:
        Strategi trading
    """
    prompt = f"""
    Based on the following market data and portfolio summary, generate a trading strategy:
    
    Market Data:
    {json.dumps(market_data, indent=2)}
    
    Portfolio Summary:
    {json.dumps(portfolio_summary, indent=2)}
    
    Please provide a trading strategy with target weights for each asset. Your response should be in JSON format with the following structure:
    {{
        "target_weights": {{
            "AAPL": float,
            "MSFT": float,
            ...
        }},
        "risk_aversion": float,
        "reasoning": "Your reasoning for the trading strategy"
    }}
    """
    
    response = llm_interface.query_json(prompt)
    
    if "target_weights" in response:
        return response
    else:
        # Fallback jika LLM gagal
        equal_weight = 1.0 / len(market_data["prices"])
        return {
            "target_weights": {asset: equal_weight for asset in market_data["prices"].keys()},
            "risk_aversion": 1.0,
            "reasoning": "Equal weight strategy (fallback)"
        }

def evaluate_strategy(portfolio_history: List[Dict[str, Any]], market_simulator: MarketSimulator) -> Dict[str, float]:
    """
    Evaluasi strategi trading.
    
    Args:
        portfolio_history: Riwayat nilai portofolio
        market_simulator: Simulator pasar
        
    Returns:
        Metrik evaluasi
    """
    # Ekstrak nilai portofolio dan tanggal
    dates = []
    values = []
    for entry in portfolio_history:
        dates.append(entry["date"])
        values.append(entry["value"])
    
    # Konversi ke array
    values = np.array(values)
    
    # Hitung returns
    returns = np.diff(values) / values[:-1]
    
    # Hitung metrik
    total_return = (values[-1] / values[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = np.std(returns) * np.sqrt(252)
    sharpe_ratio = (annualized_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
    
    # Hitung drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = np.max(drawdown)
    
    # Hitung win rate
    win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }

# Fungsi deskriptor perilaku untuk metrik keragaman
def behavior_descriptor(agent: BaseAgent) -> List[float]:
    """
    Deskriptor perilaku agen.
    
    Args:
        agent: Agen yang akan dideskripsikan
        
    Returns:
        Deskriptor perilaku
    """
    # Ekstrak parameter agen
    params = [
        agent.learning_rate,
        agent.exploration_rate,
        agent.memory_capacity / 10.0  # Normalisasi
    ]
    
    # Ekstrak informasi alat
    tool_types = {
        "sentiment": ["analyze_sentiment"],
        "return": ["calculate_expected_return"],
        "risk": ["calculate_risk"],
        "optimization": ["optimize_portfolio"],
        "metrics": ["calculate_portfolio_metrics"],
        "strategy": ["generate_trading_strategy"]
    }
    
    # Hitung proporsi untuk setiap jenis alat
    for type_name, tool_names in tool_types.items():
        count = sum(1 for tool in agent.tools if tool.name in tool_names)
        proportion = count / max(1, len(agent.tools))
        params.append(proportion)
    
    return params

# Fungsi evaluasi untuk agen
def evaluate_agent(agent: BaseAgent, task: Dict[str, Any]) -> Dict[str, float]:
    """
    Evaluasi agen pada tugas optimasi portofolio.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Metrik evaluasi
    """
    if "market_simulator" not in task or "llm_interface" not in task:
        return {
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "max_drawdown": 1.0,
            "win_rate": 0.0
        }
    
    market_simulator = task["market_simulator"]
    llm_interface = task["llm_interface"]
    
    # Reset simulator
    market_simulator.reset()
    
    # Inisialisasi portofolio
    portfolio = Portfolio()
    
    # Riwayat nilai portofolio
    portfolio_history = []
    
    # Simulasi trading
    while True:
        # Dapatkan data pasar saat ini
        market_data = market_simulator.get_current_data()
        date = market_data["date"]
        prices = market_data["prices"]
        headline = market_data["headline"]
        
        # Hitung nilai portofolio
        portfolio_value = portfolio.calculate_value(prices)
        
        # Catat nilai portofolio
        portfolio_history.append({
            "date": date,
            "value": portfolio_value
        })
        
        # Dapatkan ringkasan portofolio
        portfolio_summary = portfolio.get_summary(prices)
        
        # Cari alat yang diperlukan
        sentiment_tool = None
        expected_return_tool = None
        risk_tool = None
        optimize_tool = None
        metrics_tool = None
        strategy_tool = None
        
        for tool in agent.tools:
            if tool.name == "analyze_sentiment":
                sentiment_tool = tool
            elif tool.name == "calculate_expected_return":
                expected_return_tool = tool
            elif tool.name == "calculate_risk":
                risk_tool = tool
            elif tool.name == "optimize_portfolio":
                optimize_tool = tool
            elif tool.name == "calculate_portfolio_metrics":
                metrics_tool = tool
            elif tool.name == "generate_trading_strategy":
                strategy_tool = tool
        
        # Jika agen memiliki semua alat yang diperlukan, gunakan strategi berbasis alat
        if (sentiment_tool and expected_return_tool and risk_tool and 
            optimize_tool and metrics_tool):
            
            # Analisis sentimen
            sentiment_score = sentiment_tool.function(headline, llm_interface)
            
            # Dapatkan historical prices (30 hari terakhir)
            current_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = current_date - timedelta(days=30)
            historical_prices = market_simulator.price_data.loc[
                (market_simulator.price_data.index >= start_date) & 
                (market_simulator.price_data.index < current_date)
            ]
            
            # Hitung expected return
            expected_returns = expected_return_tool.function(prices, historical_prices, sentiment_score)
            
            # Hitung risk
            volatility, cov_matrix = risk_tool.function(historical_prices)
            
            # Optimasi portofolio
            risk_aversion = 1.0
            optimal_weights = optimize_tool.function(expected_returns, cov_matrix, risk_aversion)
            
            # Hitung metrik portofolio
            portfolio_metrics = metrics_tool.function(optimal_weights, expected_returns, cov_matrix)
            
            # Rebalance portofolio
            portfolio.rebalance(optimal_weights, prices, date)
        
        # Jika agen memiliki alat strategi, gunakan strategi berbasis LLM
        elif strategy_tool:
            # Hasilkan strategi trading
            strategy = strategy_tool.function(market_data, portfolio_summary, llm_interface)
            
            # Rebalance portofolio
            if "target_weights" in strategy:
                portfolio.rebalance(strategy["target_weights"], prices, date)
        
        # Jika tidak ada alat yang cukup, gunakan strategi equal weight
        else:
            # Equal weight strategy
            equal_weight = 1.0 / len(prices)
            weights = {asset: equal_weight for asset in prices.keys()}
            
            # Rebalance portofolio
            portfolio.rebalance(weights, prices, date)
        
        # Maju satu hari
        if not market_simulator.advance_day():
            break
    
    # Evaluasi strategi
    evaluation = evaluate_strategy(portfolio_history, market_simulator)
    
    return evaluation

# Fungsi fitness multi-objektif
class PortfolioFitness(MultiObjectiveFitness):
    """
    Fungsi fitness multi-objektif untuk optimasi portofolio.
    """
    
    def __init__(self, objectives: List[Callable[[BaseAgent, Any], float]], weights: Optional[List[float]] = None):
        """
        Inisialisasi fungsi fitness multi-objektif.
        
        Args:
            objectives: Daftar fungsi objektif
            weights: Bobot untuk setiap objektif (opsional)
        """
        super().__init__(objectives, weights)
    
    def _evaluate(self, individual: BaseAgent, task: Any) -> List[float]:
        """
        Evaluasi individu berdasarkan beberapa objektif.
        
        Args:
            individual: Individu yang akan dievaluasi
            task: Tugas untuk evaluasi
            
        Returns:
            Daftar nilai fitness untuk setiap objektif
        """
        # Evaluasi agen
        evaluation = evaluate_agent(individual, task)
        
        # Ekstrak metrik
        sharpe_ratio = evaluation.get("sharpe_ratio", 0.0)
        total_return = evaluation.get("total_return", 0.0)
        max_drawdown = evaluation.get("max_drawdown", 1.0)
        win_rate = evaluation.get("win_rate", 0.0)
        
        # Normalisasi metrik
        normalized_sharpe = max(0.0, min(1.0, (sharpe_ratio + 1.0) / 3.0))
        normalized_return = max(0.0, min(1.0, (total_return + 0.5) / 1.5))
        normalized_drawdown = max(0.0, min(1.0, 1.0 - max_drawdown))
        
        return [normalized_sharpe, normalized_return, normalized_drawdown, win_rate]

def main():
    print("=== Darwin-Gödel Machine: Complex Financial Portfolio Optimization ===")
    
    # Dapatkan API key dari environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return
    
    # Inisialisasi antarmuka LLM
    llm_interface = LLMInterface(api_key=api_key, model="gpt-3.5-turbo")
    
    # Inisialisasi simulator pasar
    market_simulator = MarketSimulator(
        assets=ASSETS,
        start_date="2023-01-01",
        end_date="2023-01-30",
        seed=42
    )
    
    # Buat agen dasar
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    
    # Tambahkan alat dasar
    agent.add_tool(Tool(
        name="analyze_sentiment",
        function=analyze_sentiment,
        description="Analyze sentiment of financial news headlines"
    ))
    
    agent.add_tool(Tool(
        name="calculate_expected_return",
        function=calculate_expected_return,
        description="Calculate expected returns for assets based on historical prices and sentiment"
    ))
    
    # Buat tugas
    task = {
        "type": "portfolio_optimization",
        "market_simulator": market_simulator,
        "llm_interface": llm_interface
    }
    
    # Evaluasi agen awal
    initial_evaluation = evaluate_agent(agent, task)
    print(f"Initial agent evaluation:")
    for metric, value in initial_evaluation.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Cetak alat agen awal
    print("\nInitial agent tools:")
    for tool in agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen awal
    print("\nInitial agent parameters:")
    print(f"  - Memory capacity: {agent.memory_capacity}")
    print(f"  - Learning rate: {agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {agent.exploration_rate:.4f}")
    
    # Tingkatkan agen menggunakan LLM
    print("\n=== Improving Agent using LLM ===")
    task_description = """
    Optimize a financial portfolio of stocks (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM, V, PG)
    based on market data, news sentiment, and portfolio metrics. The agent should analyze market sentiment,
    calculate expected returns and risks, optimize portfolio weights, and generate trading strategies
    that maximize Sharpe ratio while minimizing drawdowns.
    """
    
    improved_agent = llm_interface.improve_agent(agent, task_description)
    
    # Tambahkan alat yang mungkin belum ada
    if not any(tool.name == "calculate_risk" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(
            name="calculate_risk",
            function=calculate_risk,
            description="Calculate risk metrics for assets based on historical prices"
        ))
    
    if not any(tool.name == "optimize_portfolio" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(
            name="optimize_portfolio",
            function=optimize_portfolio,
            description="Optimize portfolio weights based on expected returns and risk"
        ))
    
    if not any(tool.name == "calculate_portfolio_metrics" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(
            name="calculate_portfolio_metrics",
            function=calculate_portfolio_metrics,
            description="Calculate portfolio metrics based on weights, expected returns, and risk"
        ))
    
    if not any(tool.name == "generate_trading_strategy" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(
            name="generate_trading_strategy",
            function=generate_trading_strategy,
            description="Generate trading strategy based on market data and portfolio summary"
        ))
    
    # Evaluasi agen yang ditingkatkan
    improved_evaluation = evaluate_agent(improved_agent, task)
    print(f"\nImproved agent evaluation:")
    for metric, value in improved_evaluation.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Cetak alat agen yang ditingkatkan
    print("\nImproved agent tools:")
    for tool in improved_agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen yang ditingkatkan
    print("\nImproved agent parameters:")
    print(f"  - Memory capacity: {improved_agent.memory_capacity}")
    print(f"  - Learning rate: {improved_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {improved_agent.exploration_rate:.4f}")
    
    # Inisialisasi DGM
    print("\n=== Initializing Darwin-Gödel Machine ===")
    dgm = DGM(initial_agent=improved_agent, population_size=10)
    
    # Definisikan fungsi objektif
    def sharpe_objective(agent, task):
        evaluation = evaluate_agent(agent, task)
        return max(0.0, min(1.0, (evaluation.get("sharpe_ratio", 0.0) + 1.0) / 3.0))
    
    def return_objective(agent, task):
        evaluation = evaluate_agent(agent, task)
        return max(0.0, min(1.0, (evaluation.get("total_return", 0.0) + 0.5) / 1.5))
    
    def drawdown_objective(agent, task):
        evaluation = evaluate_agent(agent, task)
        return max(0.0, min(1.0, 1.0 - evaluation.get("max_drawdown", 1.0)))
    
    def win_rate_objective(agent, task):
        evaluation = evaluate_agent(agent, task)
        return evaluation.get("win_rate", 0.0)
    
    # Inisialisasi komponen DGM
    print("Initializing DGM components...")
    
    # Tetapkan strategi evolusi
    dgm.set_evolution_strategy(TournamentSelection(tournament_size=3, offspring_size=5))
    
    # Tetapkan operator mutasi
    dgm.set_mutation_operator(ParameterMutation(mutation_rate=0.2, mutation_strength=0.1))
    
    # Tetapkan operator crossover
    dgm.set_crossover_operator(BlendCrossover(alpha=0.5))
    
    # Tetapkan fungsi fitness
    dgm.set_fitness_function(PortfolioFitness(
        objectives=[sharpe_objective, return_objective, drawdown_objective, win_rate_objective],
        weights=[0.4, 0.3, 0.2, 0.1]
    ))
    
    # Tetapkan metrik keragaman
    dgm.set_diversity_metric(BehavioralDiversity(behavior_descriptor=behavior_descriptor))
    
    # Tetapkan strategi arsip
    dgm.set_archive_strategy(QualityDiversityArchive(
        capacity=20,
        feature_descriptor=behavior_descriptor,
        feature_dimensions=[(0.0, 1.0, 5), (0.0, 1.0, 5), (0.0, 1.0, 5)]
    ))
    
    # Tetapkan antarmuka LLM
    dgm.set_llm_interface(llm_interface)
    
    # Jalankan evolusi
    print("\n=== Evolving Portfolio Optimization Agents ===")
    print("Evolving for 3 generations...")
    
    evolution_start_time = time.time()
    dgm.evolve(generations=3, task=task)
    evolution_time = time.time() - evolution_start_time
    
    print(f"Evolution completed in {evolution_time:.2f} seconds.")
    
    # Dapatkan agen terbaik
    best_agent = dgm.get_best_agent()
    
    # Evaluasi agen terbaik
    best_evaluation = evaluate_agent(best_agent, task)
    print(f"\nBest agent evaluation:")
    for metric, value in best_evaluation.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Cetak alat agen terbaik
    print("\nBest agent tools:")
    for tool in best_agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen terbaik
    print("\nBest agent parameters:")
    print(f"  - Memory capacity: {best_agent.memory_capacity}")
    print(f"  - Learning rate: {best_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {best_agent.exploration_rate:.4f}")
    
    # Analisis hasil evolusi
    print("\n=== Evolution Results Analysis ===")
    evolution_results = {
        "initial_evaluation": initial_evaluation,
        "improved_evaluation": improved_evaluation,
        "best_evaluation": best_evaluation,
        "evolution_time": evolution_time,
        "generations": 3,
        "population_size": dgm.population_size,
        "best_agent_tools": [tool.name for tool in best_agent.tools],
        "best_agent_parameters": {
            "memory_capacity": best_agent.memory_capacity,
            "learning_rate": best_agent.learning_rate,
            "exploration_rate": best_agent.exploration_rate
        }
    }
    
    analysis = llm_interface.analyze_evolution_results(evolution_results)
    
    print("Evolution results analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Demonstrasi penggunaan agen terbaik pada periode baru
    print("\n=== Testing Best Agent on New Market Period ===")
    
    # Inisialisasi simulator pasar baru
    new_market_simulator = MarketSimulator(
        assets=ASSETS,
        start_date="2023-02-01",
        end_date="2023-02-28",
        seed=43
    )
    
    # Buat tugas baru
    new_task = {
        "type": "portfolio_optimization",
        "market_simulator": new_market_simulator,
        "llm_interface": llm_interface
    }
    
    # Evaluasi agen terbaik pada tugas baru
    new_evaluation = evaluate_agent(best_agent, new_task)
    print(f"Best agent evaluation on new market period:")
    for metric, value in new_evaluation.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Bandingkan dengan agen awal
    initial_new_evaluation = evaluate_agent(agent, new_task)
    print(f"\nInitial agent evaluation on new market period:")
    for metric, value in initial_new_evaluation.items():
        print(f"  - {metric}: {value:.4f}")
    
    # Analisis perbandingan
    print("\n=== Comparative Analysis ===")
    comparative_results = {
        "initial_agent": {
            "training_period": initial_evaluation,
            "test_period": initial_new_evaluation
        },
        "best_agent": {
            "training_period": best_evaluation,
            "test_period": new_evaluation
        }
    }
    
    prompt = f"""
    Analyze the performance of the initial agent and the best evolved agent on both the training period and test period:
    
    {json.dumps(comparative_results, indent=2)}
    
    Please provide insights on:
    1. How well did the agents generalize to the new market period?
    2. What are the key differences in performance between the initial and best agent?
    3. What recommendations would you make for further improvements?
    
    Your response should be in JSON format with the following structure:
    {{
        "generalization_analysis": "Your analysis of how well the agents generalized",
        "performance_differences": "Your analysis of performance differences",
        "recommendations": ["recommendation1", "recommendation2", ...]
    }}
    """
    
    comparative_analysis = llm_interface.query_json(prompt)
    
    print("Comparative analysis:")
    print(json.dumps(comparative_analysis, indent=2))

if __name__ == "__main__":
    main()