"""
Modul inti untuk Darwin-Gödel Machine.

Modul ini berisi komponen-komponen inti untuk implementasi DGM yang canggih,
termasuk strategi evolusi, operator mutasi dan crossover, fungsi fitness,
metrik keragaman, dan strategi arsip.
"""

from simple_dgm.core.evolution_strategies import (
    EvolutionStrategy, TournamentSelection, RouletteWheelSelection, 
    NSGA2Selection, MAP_ElitesStrategy, CMA_ESStrategy
)
from simple_dgm.core.mutation_operators import (
    MutationOperator, ParameterMutation, StructuralMutation, 
    FunctionalMutation, AdaptiveMutation, SelfAdaptiveMutation
)
from simple_dgm.core.crossover_operators import (
    CrossoverOperator, UniformCrossover, SinglePointCrossover,
    MultiPointCrossover, BlendCrossover, SimulatedBinaryCrossover
)
from simple_dgm.core.fitness_functions import (
    FitnessFunction, MultiObjectiveFitness, LexicographicFitness,
    AggregatedFitness, ParetoDominanceFitness
)
from simple_dgm.core.diversity_metrics import (
    DiversityMetric, BehavioralDiversity, StructuralDiversity,
    GenotypeDiversity, PhenotypeDiversity, FunctionalDiversity
)
from simple_dgm.core.archive_strategies import (
    ArchiveStrategy, NoveltyArchive, EliteArchive, QualityDiversityArchive,
    ParetoDominanceArchive, AgeLayeredArchive
)
from simple_dgm.core.introspection import (
    IntrospectionEngine, CodeAnalyzer, PerformanceProfiler,
    BehaviorTracker, SelfImprovementEngine
)
from simple_dgm.core.adaptation import (
    AdaptationEngine, EnvironmentModel, TaskModel,
    ResourceManager, LearningRateScheduler
)
from simple_dgm.core.collaboration import (
    CollaborationEngine, AgentNetwork, KnowledgeSharing,
    TaskDecomposition, ConsensusBuilder
)
from simple_dgm.core.llm_integration import (
    LLMInterface, CodeGeneration, ProblemSolving,
    KnowledgeExtraction, SelfModification
)

__all__ = [
    'EvolutionStrategy', 'TournamentSelection', 'RouletteWheelSelection', 
    'NSGA2Selection', 'MAP_ElitesStrategy', 'CMA_ESStrategy',
    'MutationOperator', 'ParameterMutation', 'StructuralMutation', 
    'FunctionalMutation', 'AdaptiveMutation', 'SelfAdaptiveMutation',
    'CrossoverOperator', 'UniformCrossover', 'SinglePointCrossover',
    'MultiPointCrossover', 'BlendCrossover', 'SimulatedBinaryCrossover',
    'FitnessFunction', 'MultiObjectiveFitness', 'LexicographicFitness',
    'AggregatedFitness', 'ParetoDominanceFitness',
    'DiversityMetric', 'BehavioralDiversity', 'StructuralDiversity',
    'GenotypeDiversity', 'PhenotypeDiversity', 'FunctionalDiversity',
    'ArchiveStrategy', 'NoveltyArchive', 'EliteArchive', 'QualityDiversityArchive',
    'ParetoDominanceArchive', 'AgeLayeredArchive',
    'IntrospectionEngine', 'CodeAnalyzer', 'PerformanceProfiler',
    'BehaviorTracker', 'SelfImprovementEngine',
    'AdaptationEngine', 'EnvironmentModel', 'TaskModel',
    'ResourceManager', 'LearningRateScheduler',
    'CollaborationEngine', 'AgentNetwork', 'KnowledgeSharing',
    'TaskDecomposition', 'ConsensusBuilder',
    'LLMInterface', 'CodeGeneration', 'ProblemSolving',
    'KnowledgeExtraction', 'SelfModification'
]