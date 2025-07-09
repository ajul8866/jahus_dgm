"""
Contoh integrasi LLM untuk Darwin-Gödel Machine.

Contoh ini mendemonstrasikan penggunaan integrasi LLM untuk:
- Pembuatan kode
- Pemecahan masalah
- Ekstraksi pengetahuan
- Modifikasi diri
"""

import os
import random
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple

from simple_dgm.dgm import DGM
from simple_dgm.agents.coding_agent import CodingAgent
from simple_dgm.core.llm_integration import (
    LLMInterface, CodeGeneration, ProblemSolving, 
    KnowledgeExtraction, SelfModification
)


def main():
    print("=== Darwin-Gödel Machine: LLM Integration Example ===")
    
    # Dapatkan API key dari environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        print("Using mock LLM interface for demonstration purposes.")
        use_mock = True
    else:
        use_mock = False
    
    # Inisialisasi antarmuka LLM
    if use_mock:
        llm_interface = MockLLMInterface()
    else:
        llm_interface = LLMInterface(api_key=api_key, model="gpt-3.5-turbo")
    
    # Inisialisasi komponen LLM
    code_generation = CodeGeneration(llm_interface)
    problem_solving = ProblemSolving(llm_interface)
    knowledge_extraction = KnowledgeExtraction(llm_interface)
    self_modification = SelfModification(llm_interface)
    
    # Demonstrasi pembuatan kode
    print("\n=== Code Generation Demo ===")
    demo_code_generation(code_generation)
    
    # Demonstrasi pemecahan masalah
    print("\n=== Problem Solving Demo ===")
    demo_problem_solving(problem_solving)
    
    # Demonstrasi ekstraksi pengetahuan
    print("\n=== Knowledge Extraction Demo ===")
    demo_knowledge_extraction(knowledge_extraction)
    
    # Demonstrasi modifikasi diri
    print("\n=== Self Modification Demo ===")
    demo_self_modification(self_modification)
    
    # Demonstrasi integrasi dengan agen pengkodean
    print("\n=== Coding Agent Integration Demo ===")
    demo_coding_agent_integration(code_generation)


def demo_code_generation(code_generation):
    """
    Demonstrasi pembuatan kode.
    
    Args:
        code_generation: Komponen pembuatan kode
    """
    # Hasilkan fungsi
    print("Generating a function...")
    function_code = code_generation.generate_function(
        function_name="calculate_fibonacci",
        description="Calculate the nth Fibonacci number efficiently using dynamic programming",
        parameters=[
            {"name": "n", "type": "int", "description": "The position in the Fibonacci sequence (0-indexed)"}
        ],
        return_type="int"
    )
    
    print("Generated function:")
    print(function_code)
    print()
    
    # Hasilkan kelas
    print("Generating a class...")
    class_code = code_generation.generate_class(
        class_name="MathUtility",
        description="A utility class for mathematical operations",
        base_classes=["object"],
        methods=[
            {
                "name": "__init__",
                "description": "Initialize the MathUtility",
                "parameters": [
                    {"name": "precision", "type": "int", "description": "Decimal precision for floating point operations"}
                ]
            },
            {
                "name": "factorial",
                "description": "Calculate the factorial of a number",
                "parameters": [
                    {"name": "n", "type": "int", "description": "The number to calculate factorial for"}
                ],
                "return_type": "int"
            },
            {
                "name": "is_prime",
                "description": "Check if a number is prime",
                "parameters": [
                    {"name": "n", "type": "int", "description": "The number to check"}
                ],
                "return_type": "bool"
            }
        ]
    )
    
    print("Generated class:")
    print(class_code)


def demo_problem_solving(problem_solving):
    """
    Demonstrasi pemecahan masalah.
    
    Args:
        problem_solving: Komponen pemecahan masalah
    """
    # Analisis masalah
    problem = "Find the maximum subarray sum in an array of integers."
    
    print(f"Analyzing problem: {problem}")
    analysis = problem_solving.analyze_problem(problem)
    
    print("Problem analysis:")
    for key, value in analysis.items():
        print(f"  - {key}: {value}")
    print()
    
    # Dekomposisi masalah
    print("Decomposing problem...")
    decomposition = problem_solving.decompose_problem(problem)
    
    print("Problem decomposition:")
    for i, sub_problem in enumerate(decomposition):
        print(f"  Sub-problem {i+1}:")
        for key, value in sub_problem.items():
            print(f"    - {key}: {value}")
    print()
    
    # Solusi masalah
    print("Solving problem...")
    solution = problem_solving.solve_problem(problem)
    
    print("Problem solution:")
    print(solution[:300] + "..." if len(solution) > 300 else solution)
    print()
    
    # Evaluasi solusi
    print("Evaluating solution...")
    evaluation = problem_solving.evaluate_solution(problem, solution)
    
    print("Solution evaluation:")
    for key, value in evaluation.items():
        print(f"  - {key}: {value}")


def demo_knowledge_extraction(knowledge_extraction):
    """
    Demonstrasi ekstraksi pengetahuan.
    
    Args:
        knowledge_extraction: Komponen ekstraksi pengetahuan
    """
    # Teks untuk ekstraksi
    text = """
    Darwin-Gödel Machine (DGM) is a self-improving AI system that combines Darwinian evolution
    and Gödelian self-reference. It uses evolutionary algorithms to evolve a population of agents,
    and self-reference to modify its own code. The system has several components:
    
    1. Agents: The basic units that solve problems
    2. Evolution: The process of selecting and modifying agents
    3. Introspection: The ability to analyze and understand its own code
    4. Adaptation: The ability to adapt to changing environments and tasks
    5. Collaboration: The ability to collaborate with other agents
    
    DGM can be applied to various domains, including software development, scientific research,
    and creative tasks.
    """
    
    # Ekstrak konsep
    print("Extracting concepts...")
    concepts = knowledge_extraction.extract_concepts(text)
    
    print("Extracted concepts:")
    for i, concept in enumerate(concepts[:3]):  # Tampilkan 3 konsep pertama
        print(f"  Concept {i+1}:")
        for key, value in concept.items():
            print(f"    - {key}: {value}")
    print()
    
    # Ekstrak hubungan
    print("Extracting relationships...")
    relationships = knowledge_extraction.extract_relationships(text)
    
    print("Extracted relationships:")
    for i, relationship in enumerate(relationships[:3]):  # Tampilkan 3 hubungan pertama
        print(f"  Relationship {i+1}:")
        for key, value in relationship.items():
            print(f"    - {key}: {value}")
    print()
    
    # Ekstrak grafik pengetahuan
    print("Extracting knowledge graph...")
    knowledge_graph = knowledge_extraction.extract_knowledge_graph(text)
    
    print("Knowledge graph:")
    if "nodes" in knowledge_graph:
        print(f"  Nodes: {len(knowledge_graph['nodes'])}")
    if "edges" in knowledge_graph:
        print(f"  Edges: {len(knowledge_graph['edges'])}")


def demo_self_modification(self_modification):
    """
    Demonstrasi modifikasi diri.
    
    Args:
        self_modification: Komponen modifikasi diri
    """
    # Kode untuk dianalisis dan ditingkatkan
    code = """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
    """
    
    # Analisis kode
    print("Analyzing code...")
    analysis = self_modification.analyze_code(code)
    
    print("Code analysis:")
    for key, value in analysis.items():
        print(f"  - {key}: {value}")
    print()
    
    # Tingkatkan kode
    print("Improving code...")
    improved_code = self_modification.improve_code(code)
    
    print("Improved code:")
    print(improved_code)
    print()
    
    # Tambahkan fitur
    print("Adding feature...")
    feature_description = "Add a function to calculate the greatest common divisor (GCD) of two numbers"
    modified_code = self_modification.add_feature(improved_code, feature_description)
    
    print("Code with new feature:")
    print(modified_code)
    print()
    
    # Refaktor kode
    print("Refactoring code...")
    refactored_code = self_modification.refactor_code(modified_code, "extract method")
    
    print("Refactored code:")
    print(refactored_code)


def demo_coding_agent_integration(code_generation):
    """
    Demonstrasi integrasi dengan agen pengkodean.
    
    Args:
        code_generation: Komponen pembuatan kode
    """
    # Buat agen pengkodean dengan integrasi LLM
    agent = CodingAgent(
        code_style="clean",
        preferred_language="python",
        memory_capacity=10,
        learning_rate=0.01,
        exploration_rate=0.1
    )
    
    # Tetapkan mesin pembuatan kode
    agent.code_generation_engine = code_generation
    
    # Inisialisasi DGM
    dgm = DGM(initial_agent=agent)
    
    # Jalankan evolusi
    print("Evolving coding agent for 5 generations...")
    
    # Tugas pengkodean
    task = {
        "type": "code_generation",
        "description": "Generate a function to sort a list of integers",
        "requirements": [
            "The function should be named 'sort_list'",
            "It should take a list of integers as input",
            "It should return a sorted list in ascending order",
            "It should handle empty lists and lists with one element",
            "It should not use the built-in sort() method"
        ]
    }
    
    # Evolusi
    dgm.evolve(generations=5, task=task)
    
    # Dapatkan agen terbaik
    best_agent = dgm.get_best_agent()
    
    # Uji agen pada tugas baru
    print("\nTesting best agent on a new task...")
    
    new_task = {
        "type": "code_generation",
        "description": "Generate a function to find the median of a list of numbers",
        "requirements": [
            "The function should be named 'find_median'",
            "It should take a list of numbers as input",
            "It should return the median value",
            "It should handle lists with odd and even number of elements",
            "It should handle empty lists by returning None"
        ]
    }
    
    # Hasilkan kode
    code = best_agent.generate_code(new_task)
    
    print("Generated code:")
    print(code)


class MockLLMInterface:
    """
    Antarmuka LLM tiruan untuk demonstrasi.
    """
    
    def __init__(self):
        """
        Inisialisasi antarmuka LLM tiruan.
        """
        self.responses = {
            "function": """
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number efficiently using dynamic programming"""
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    
    # Handle base cases
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Use dynamic programming approach
    fib = [0] * (n + 1)
    fib[0] = 0
    fib[1] = 1
    
    for i in range(2, n + 1):
        fib[i] = fib[i-1] + fib[i-2]
    
    return fib[n]
""",
            "class": """
class MathUtility:
    """A utility class for mathematical operations"""
    
    def __init__(self, precision=2):
        """Initialize the MathUtility
        
        Args:
            precision: Decimal precision for floating point operations
        """
        self.precision = precision
    
    def factorial(self, n):
        """Calculate the factorial of a number
        
        Args:
            n: The number to calculate factorial for
            
        Returns:
            The factorial of n
        """
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        
        result = 1
        for i in range(2, n + 1):
            result *= i
        
        return result
    
    def is_prime(self, n):
        """Check if a number is prime
        
        Args:
            n: The number to check
            
        Returns:
            True if the number is prime, False otherwise
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        
        if n % 2 == 0 or n % 3 == 0:
            return False
        
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        
        return True
""",
            "problem_analysis": {
                "problem_type": "array processing",
                "complexity": "medium",
                "key_concepts": ["dynamic programming", "array traversal", "maximum subarray sum", "Kadane's algorithm"],
                "approach": "Use Kadane's algorithm to find the maximum subarray sum in O(n) time",
                "potential_challenges": ["Handling empty arrays", "Handling arrays with all negative numbers"]
            },
            "problem_decomposition": [
                {
                    "id": "subproblem1",
                    "description": "Understand the problem and identify edge cases",
                    "dependencies": []
                },
                {
                    "id": "subproblem2",
                    "description": "Implement Kadane's algorithm for finding maximum subarray sum",
                    "dependencies": ["subproblem1"]
                },
                {
                    "id": "subproblem3",
                    "description": "Test the solution with various test cases",
                    "dependencies": ["subproblem2"]
                }
            ],
            "problem_solution": """
To find the maximum subarray sum in an array of integers, we can use Kadane's algorithm, which is an efficient dynamic programming approach with O(n) time complexity.

Here's the algorithm:
1. Initialize two variables: max_so_far = -infinity and max_ending_here = 0
2. Iterate through each element in the array:
   a. max_ending_here = max(arr[i], max_ending_here + arr[i])
   b. max_so_far = max(max_so_far, max_ending_here)
3. Return max_so_far

Python implementation:

```python
def max_subarray_sum(arr):
    if not arr:
        return 0
    
    max_so_far = float('-inf')
    max_ending_here = 0
    
    for num in arr:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)
    
    return max_so_far
```

This algorithm handles all cases, including arrays with all negative numbers (it will return the largest negative number) and empty arrays (it will return 0).

Example:
- For array [−2, 1, −3, 4, −1, 2, 1, −5, 4], the maximum subarray sum is 6, corresponding to the subarray [4, −1, 2, 1].
- For array [−1, −2, −3, −4], the maximum subarray sum is -1, corresponding to the subarray [−1].
""",
            "solution_evaluation": {
                "correctness": 0.95,
                "efficiency": 0.9,
                "clarity": 0.85,
                "overall_score": 0.9,
                "strengths": ["Uses optimal Kadane's algorithm", "Handles edge cases", "Clear explanation"],
                "weaknesses": ["Could add more examples", "Could explain time and space complexity more explicitly"],
                "suggestions": ["Add time and space complexity analysis", "Add more test cases", "Consider handling empty array case separately"]
            },
            "concepts": [
                {
                    "name": "Darwin-Gödel Machine",
                    "description": "A self-improving AI system that combines Darwinian evolution and Gödelian self-reference",
                    "category": "AI System"
                },
                {
                    "name": "Agents",
                    "description": "The basic units in DGM that solve problems",
                    "category": "Component"
                },
                {
                    "name": "Evolution",
                    "description": "The process of selecting and modifying agents in DGM",
                    "category": "Process"
                }
            ],
            "relationships": [
                {
                    "source": "Darwin-Gödel Machine",
                    "target": "Agents",
                    "type": "contains",
                    "description": "DGM contains agents as its basic units"
                },
                {
                    "source": "Darwin-Gödel Machine",
                    "target": "Evolution",
                    "type": "uses",
                    "description": "DGM uses evolution to improve agents"
                },
                {
                    "source": "Evolution",
                    "target": "Agents",
                    "type": "modifies",
                    "description": "Evolution modifies agents to improve them"
                }
            ],
            "knowledge_graph": {
                "nodes": [
                    {"id": "dgm", "label": "Darwin-Gödel Machine", "type": "system"},
                    {"id": "agents", "label": "Agents", "type": "component"},
                    {"id": "evolution", "label": "Evolution", "type": "process"},
                    {"id": "introspection", "label": "Introspection", "type": "capability"},
                    {"id": "adaptation", "label": "Adaptation", "type": "capability"},
                    {"id": "collaboration", "label": "Collaboration", "type": "capability"}
                ],
                "edges": [
                    {"source": "dgm", "target": "agents", "label": "contains", "type": "composition"},
                    {"source": "dgm", "target": "evolution", "label": "uses", "type": "usage"},
                    {"source": "dgm", "target": "introspection", "label": "has", "type": "capability"},
                    {"source": "dgm", "target": "adaptation", "label": "has", "type": "capability"},
                    {"source": "dgm", "target": "collaboration", "label": "has", "type": "capability"},
                    {"source": "evolution", "target": "agents", "label": "modifies", "type": "action"}
                ]
            },
            "code_analysis": {
                "complexity": "medium",
                "quality": "low",
                "issues": [
                    "Recursive Fibonacci implementation has exponential time complexity",
                    "is_prime function has O(n) time complexity, can be improved",
                    "No input validation for negative numbers in is_prime",
                    "No docstrings or comments"
                ],
                "strengths": [
                    "Simple and readable code",
                    "Correct base cases for Fibonacci"
                ],
                "improvement_suggestions": [
                    "Use dynamic programming for Fibonacci to improve efficiency",
                    "Optimize is_prime to check up to sqrt(n)",
                    "Add input validation",
                    "Add docstrings and comments"
                ]
            },
            "improved_code": """
def fibonacci(n):
    """
    Calculate the nth Fibonacci number using dynamic programming.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    
    if n <= 1:
        return n
    
    # Use dynamic programming approach
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib[n]

def is_prime(n):
    """
    Check if a number is prime.
    
    Args:
        n: The number to check
        
    Returns:
        True if the number is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    
    # Check if n is divisible by 2 or 3
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Check all numbers of form 6k ± 1 up to sqrt(n)
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True
""",
            "feature_added_code": """
def fibonacci(n):
    """
    Calculate the nth Fibonacci number using dynamic programming.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    
    if n <= 1:
        return n
    
    # Use dynamic programming approach
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib[n]

def is_prime(n):
    """
    Check if a number is prime.
    
    Args:
        n: The number to check
        
    Returns:
        True if the number is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    
    # Check if n is divisible by 2 or 3
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Check all numbers of form 6k ± 1 up to sqrt(n)
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True

def gcd(a, b):
    """
    Calculate the greatest common divisor (GCD) of two numbers using Euclidean algorithm.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The greatest common divisor of a and b
    """
    # Ensure a and b are non-negative
    a, b = abs(a), abs(b)
    
    # Base case
    if b == 0:
        return a
    
    # Recursive case
    return gcd(b, a % b)
""",
            "refactored_code": """
def fibonacci(n):
    """
    Calculate the nth Fibonacci number using dynamic programming.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    
    if n <= 1:
        return n
    
    return _calculate_fibonacci_dp(n)

def _calculate_fibonacci_dp(n):
    """
    Helper function to calculate Fibonacci using dynamic programming.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
    """
    fib = [0, 1]
    for i in range(2, n + 1):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib[n]

def is_prime(n):
    """
    Check if a number is prime.
    
    Args:
        n: The number to check
        
    Returns:
        True if the number is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    
    if _is_divisible_by_small_primes(n):
        return False
    
    return _check_larger_divisors(n)

def _is_divisible_by_small_primes(n):
    """
    Check if n is divisible by 2 or 3.
    
    Args:
        n: The number to check
        
    Returns:
        True if n is divisible by 2 or 3, False otherwise
    """
    return n % 2 == 0 or n % 3 == 0

def _check_larger_divisors(n):
    """
    Check if n is divisible by any number of form 6k +/- 1 up to sqrt(n).
    
    Args:
        n: The number to check
        
    Returns:
        True if n is not divisible by any such number, False otherwise
    """
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    
    return True

def gcd(a, b):
    """
    Calculate the greatest common divisor (GCD) of two numbers using Euclidean algorithm.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The greatest common divisor of a and b
    """
    # Ensure a and b are non-negative
    a, b = abs(a), abs(b)
    
    # Base case
    if b == 0:
        return a
    
    # Recursive case
    return gcd(b, a % b)
""",
            "generated_code": """
def sort_list(numbers):
    """
    Sort a list of integers in ascending order without using built-in sort.
    
    Args:
        numbers: A list of integers to sort
        
    Returns:
        A new list with the integers sorted in ascending order
    """
    # Handle empty list or list with one element
    if not numbers:
        return []
    if len(numbers) == 1:
        return numbers.copy()
    
    # Implementation of merge sort
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr
        
        # Divide the list into two halves
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        
        # Merge the sorted halves
        return merge(left, right)
    
    def merge(left, right):
        result = []
        i = j = 0
        
        # Compare elements from both lists and add the smaller one to the result
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Add any remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    return merge_sort(numbers)
"""
        }
    
    def generate(self, prompt: str, max_tokens: int = 1000, 
                temperature: float = 0.7, stop: Optional[List[str]] = None) -> str:
        """
        Hasilkan teks dari LLM tiruan.
        
        Args:
            prompt: Prompt untuk LLM
            max_tokens: Jumlah maksimum token yang akan dihasilkan
            temperature: Temperatur sampling
            stop: Daftar string yang menghentikan generasi
            
        Returns:
            Teks yang dihasilkan
        """
        # Tentukan respons berdasarkan prompt
        if "Generate a Python function" in prompt:
            return self.responses["function"]
        elif "Generate a Python class" in prompt:
            return self.responses["class"]
        elif "Analyze the following problem" in prompt:
            return json.dumps(self.responses["problem_analysis"], indent=2)
        elif "Decompose the following problem" in prompt:
            return json.dumps(self.responses["problem_decomposition"], indent=2)
        elif "Solve the following problem" in prompt:
            return self.responses["problem_solution"]
        elif "Evaluate the following solution" in prompt:
            return json.dumps(self.responses["solution_evaluation"], indent=2)
        elif "Extract key concepts" in prompt:
            return json.dumps(self.responses["concepts"], indent=2)
        elif "Extract relationships" in prompt:
            return json.dumps(self.responses["relationships"], indent=2)
        elif "Extract a knowledge graph" in prompt:
            return json.dumps(self.responses["knowledge_graph"], indent=2)
        elif "Analyze the following code" in prompt:
            return json.dumps(self.responses["code_analysis"], indent=2)
        elif "Improve the following code" in prompt:
            return self.responses["improved_code"]
        elif "Add the following feature" in prompt:
            return self.responses["feature_added_code"]
        elif "Refactor the following code" in prompt:
            return self.responses["refactored_code"]
        elif "sort a list of integers" in prompt:
            return self.responses["generated_code"]
        else:
            return "I don't have a specific response for this prompt."


if __name__ == "__main__":
    main()