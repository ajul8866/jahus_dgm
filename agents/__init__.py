"""
Modul agen untuk Darwin-Gödel Machine.
"""

from simple_dgm.agents.base_agent import BaseAgent
from simple_dgm.agents.coding_agent import CodingAgent
from simple_dgm.agents.problem_solving_agent import ProblemSolvingAgent
from simple_dgm.agents.meta_agent import MetaAgent

__all__ = [
    'BaseAgent',
    'CodingAgent',
    'ProblemSolvingAgent',
    'MetaAgent',
]