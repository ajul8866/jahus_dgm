"""
Implementasi kompleks integrasi LLM dengan Darwin-Gödel Machine.

Contoh ini mendemonstrasikan penggunaan LLM untuk:
1. Analisis kode dan pembuatan kode
2. Peningkatan agen dengan kemampuan pemecahan masalah kompleks
3. Generasi strategi evolusi dan operator genetik
4. Analisis hasil evolusi dan rekomendasi peningkatan
"""

import os
import random
import time
import json
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool
from simple_dgm.core.llm_integration_real import (
    LLMInterface, CodeGeneration, ProblemSolving, 
    KnowledgeExtraction, SelfModification
)

# Definisikan fungsi alat dasar
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y if y != 0 else 0

def power(x, y):
    return x ** y

def sqrt(x):
    return x ** 0.5 if x >= 0 else 0

def factorial(n):
    if n < 0:
        return 0
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, int(n) + 1):
        result *= i
    return result

def is_prime(n):
    """
    Periksa apakah n adalah bilangan prima.
    
    Args:
        n: Bilangan yang akan diperiksa
        
    Returns:
        True jika n adalah bilangan prima, False jika tidak
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

def fibonacci(n):
    """
    Hitung bilangan Fibonacci ke-n.
    
    Args:
        n: Indeks bilangan Fibonacci (0-indexed)
        
    Returns:
        Bilangan Fibonacci ke-n
    """
    if n < 0:
        return 0
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def gcd(a, b):
    """
    Hitung GCD (Greatest Common Divisor) dari a dan b.
    
    Args:
        a: Bilangan pertama
        b: Bilangan kedua
        
    Returns:
        GCD dari a dan b
    """
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """
    Hitung LCM (Least Common Multiple) dari a dan b.
    
    Args:
        a: Bilangan pertama
        b: Bilangan kedua
        
    Returns:
        LCM dari a dan b
    """
    return abs(a * b) // gcd(a, b) if a != 0 and b != 0 else 0

# Fungsi untuk pemecahan masalah matematika
def solve_equation(equation, llm_interface):
    """
    Selesaikan persamaan matematika menggunakan LLM.
    
    Args:
        equation: Persamaan yang akan diselesaikan
        llm_interface: Antarmuka LLM
        
    Returns:
        Solusi persamaan
    """
    prompt = f"""
    Solve the following mathematical equation:
    {equation}
    
    Please provide the step-by-step solution and the final answer.
    Your response should be in JSON format with the following structure:
    {{
        "steps": [
            "step1",
            "step2",
            ...
        ],
        "solution": "final_answer"
    }}
    """
    
    response = llm_interface.query_json(prompt)
    
    if "solution" in response:
        return response
    else:
        return {"steps": [], "solution": "Unable to solve the equation"}

def analyze_problem(problem, llm_interface):
    """
    Analisis masalah matematika menggunakan LLM.
    
    Args:
        problem: Masalah yang akan dianalisis
        llm_interface: Antarmuka LLM
        
    Returns:
        Analisis masalah
    """
    prompt = f"""
    Analyze the following mathematical problem:
    {problem}
    
    Please provide an analysis of the problem, including:
    1. Problem type
    2. Key concepts involved
    3. Potential approaches to solve it
    4. Difficulty level
    
    Your response should be in JSON format with the following structure:
    {{
        "problem_type": "type",
        "key_concepts": ["concept1", "concept2", ...],
        "approaches": [
            {{
                "name": "approach_name",
                "description": "approach_description"
            }}
        ],
        "difficulty": "level"
    }}
    """
    
    response = llm_interface.query_json(prompt)
    
    if "problem_type" in response:
        return response
    else:
        return {
            "problem_type": "Unknown",
            "key_concepts": [],
            "approaches": [],
            "difficulty": "Unknown"
        }

def generate_problem(topic, difficulty, llm_interface):
    """
    Hasilkan masalah matematika menggunakan LLM.
    
    Args:
        topic: Topik masalah
        difficulty: Tingkat kesulitan
        llm_interface: Antarmuka LLM
        
    Returns:
        Masalah matematika
    """
    prompt = f"""
    Generate a mathematical problem on the topic of {topic} with {difficulty} difficulty level.
    
    Please provide the problem statement and the solution.
    Your response should be in JSON format with the following structure:
    {{
        "problem": "problem_statement",
        "solution": "solution",
        "explanation": "explanation_of_solution"
    }}
    """
    
    response = llm_interface.query_json(prompt)
    
    if "problem" in response:
        return response
    else:
        return {
            "problem": f"Generate a problem about {topic} with {difficulty} difficulty",
            "solution": "No solution provided",
            "explanation": "No explanation provided"
        }

def implement_algorithm(algorithm_name, llm_interface):
    """
    Implementasikan algoritma menggunakan LLM.
    
    Args:
        algorithm_name: Nama algoritma
        llm_interface: Antarmuka LLM
        
    Returns:
        Implementasi algoritma
    """
    prompt = f"""
    Implement the {algorithm_name} algorithm in Python.
    
    Please provide a clean, efficient implementation with comments explaining the key steps.
    Your response should be in JSON format with the following structure:
    {{
        "code": "python_code",
        "explanation": "explanation_of_algorithm",
        "time_complexity": "time_complexity",
        "space_complexity": "space_complexity"
    }}
    """
    
    response = llm_interface.query_json(prompt)
    
    if "code" in response:
        return response
    else:
        return {
            "code": f"# Implementation of {algorithm_name}",
            "explanation": "No explanation provided",
            "time_complexity": "Unknown",
            "space_complexity": "Unknown"
        }

# Fungsi evaluasi untuk agen
def evaluate_agent(agent, task):
    """
    Evaluasi agen pada tugas pemecahan masalah matematika.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Skor evaluasi
    """
    if "problems" not in task or "llm_interface" not in task:
        return 0.0
    
    problems = task["problems"]
    llm_interface = task["llm_interface"]
    
    total_score = 0.0
    
    for problem in problems:
        problem_type = problem["type"]
        expected_solution = problem["solution"]
        
        # Cari alat yang sesuai
        tool = None
        
        if problem_type == "equation":
            problem_statement = problem.get("statement", "")
            for t in agent.tools:
                if t.name == "solve_equation":
                    tool = t
                    break
        elif problem_type == "analysis":
            problem_statement = problem.get("statement", "")
            for t in agent.tools:
                if t.name == "analyze_problem":
                    tool = t
                    break
        elif problem_type == "generation":
            for t in agent.tools:
                if t.name == "generate_problem":
                    tool = t
                    break
        elif problem_type == "implementation":
            problem_statement = problem.get("statement", "")
            for t in agent.tools:
                if t.name == "implement_algorithm":
                    tool = t
                    break
        
        # Jika tidak ada alat yang sesuai, coba alat dasar
        if tool is None:
            # Coba evaluasi dengan alat dasar
            if "expression" in problem:
                expression = problem["expression"]
                parts = expression.split()
                
                if len(parts) == 3:
                    a = float(parts[0])
                    op = parts[1]
                    b = float(parts[2])
                    
                    if op == "+":
                        tool_name = "add"
                    elif op == "-":
                        tool_name = "subtract"
                    elif op == "*":
                        tool_name = "multiply"
                    elif op == "/":
                        tool_name = "divide"
                    elif op == "^":
                        tool_name = "power"
                    else:
                        continue
                    
                    for t in agent.tools:
                        if t.name == tool_name:
                            tool = t
                            break
        
        # Jika ada alat yang sesuai, gunakan
        if tool:
            try:
                if problem_type == "equation":
                    result = tool.function(problem_statement, llm_interface)
                    if "solution" in result and result["solution"] == expected_solution:
                        total_score += 1.0
                    else:
                        total_score += 0.5  # Partial credit
                elif problem_type == "analysis":
                    result = tool.function(problem_statement, llm_interface)
                    if "problem_type" in result and result["problem_type"] == expected_solution:
                        total_score += 1.0
                    else:
                        total_score += 0.5  # Partial credit
                elif problem_type == "generation":
                    topic = problem.get("topic", "algebra")
                    difficulty = problem.get("difficulty", "medium")
                    result = tool.function(topic, difficulty, llm_interface)
                    if "problem" in result and "solution" in result:
                        total_score += 1.0
                    else:
                        total_score += 0.5  # Partial credit
                elif problem_type == "implementation":
                    result = tool.function(problem_statement, llm_interface)
                    if "code" in result and "time_complexity" in result:
                        total_score += 1.0
                    else:
                        total_score += 0.5  # Partial credit
                elif "expression" in problem:
                    expression = problem["expression"]
                    parts = expression.split()
                    
                    if len(parts) == 3:
                        a = float(parts[0])
                        op = parts[1]
                        b = float(parts[2])
                        
                        result = tool.function(a, b)
                        expected = float(expected_solution)
                        
                        if abs(result - expected) < 0.001:
                            total_score += 1.0
            except Exception as e:
                print(f"Error evaluating problem: {e}")
    
    return total_score / len(problems) if problems else 0.0

# Buat tugas matematika
def create_math_task(llm_interface):
    """
    Buat tugas matematika.
    
    Args:
        llm_interface: Antarmuka LLM
        
    Returns:
        Tugas matematika
    """
    problems = []
    
    # Persamaan
    equation_problems = [
        {
            "type": "equation",
            "statement": "2x + 5 = 15",
            "solution": "x = 5"
        },
        {
            "type": "equation",
            "statement": "3x - 7 = 2x + 4",
            "solution": "x = 11"
        },
        {
            "type": "equation",
            "statement": "x^2 - 5x + 6 = 0",
            "solution": "x = 2, x = 3"
        }
    ]
    problems.extend(equation_problems)
    
    # Analisis
    analysis_problems = [
        {
            "type": "analysis",
            "statement": "Find the derivative of f(x) = x^3 - 2x^2 + 4x - 7",
            "solution": "Calculus"
        },
        {
            "type": "analysis",
            "statement": "Solve the system of equations: 2x + y = 5, 3x - 2y = 4",
            "solution": "Linear Algebra"
        }
    ]
    problems.extend(analysis_problems)
    
    # Generasi
    generation_problems = [
        {
            "type": "generation",
            "topic": "probability",
            "difficulty": "medium",
            "solution": "Any valid probability problem"
        },
        {
            "type": "generation",
            "topic": "geometry",
            "difficulty": "hard",
            "solution": "Any valid geometry problem"
        }
    ]
    problems.extend(generation_problems)
    
    # Implementasi
    implementation_problems = [
        {
            "type": "implementation",
            "statement": "quicksort",
            "solution": "Any valid quicksort implementation"
        },
        {
            "type": "implementation",
            "statement": "binary search",
            "solution": "Any valid binary search implementation"
        }
    ]
    problems.extend(implementation_problems)
    
    # Operasi aritmatika
    for _ in range(5):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        operations = [
            (f"{a} + {b}", a + b),
            (f"{a} - {b}", a - b),
            (f"{a} * {b}", a * b),
            (f"{a} / {b}", a / b if b != 0 else 0),
            (f"{a} ^ {b}", a ** b)
        ]
        for expr, result in operations:
            problems.append({
                "type": "arithmetic",
                "expression": expr,
                "solution": str(result)
            })
    
    return {
        "type": "mathematical_problem_solving",
        "problems": problems,
        "llm_interface": llm_interface
    }

def main():
    print("=== Darwin-Gödel Machine: Complex LLM Integration ===")
    
    # Dapatkan API key dari environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return
    
    # Inisialisasi antarmuka LLM
    llm_interface = LLMInterface(api_key=api_key, model="gpt-3.5-turbo")
    
    # Inisialisasi komponen LLM
    code_generation = CodeGeneration(llm_interface)
    problem_solving = ProblemSolving(llm_interface)
    knowledge_extraction = KnowledgeExtraction(llm_interface)
    self_modification = SelfModification(llm_interface)
    
    # Buat agen dasar dengan beberapa alat
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    
    # Tambahkan alat dasar
    agent.add_tool(Tool(name="add", function=add, description="Add two numbers"))
    agent.add_tool(Tool(name="subtract", function=subtract, description="Subtract two numbers"))
    agent.add_tool(Tool(name="multiply", function=multiply, description="Multiply two numbers"))
    agent.add_tool(Tool(name="divide", function=divide, description="Divide two numbers"))
    
    # Buat tugas
    task = create_math_task(llm_interface)
    
    # Evaluasi agen awal
    initial_score = evaluate_agent(agent, task)
    print(f"Initial agent score: {initial_score:.4f}")
    
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
    Solve various mathematical problems including:
    1. Solving equations
    2. Analyzing mathematical problems
    3. Generating mathematical problems
    4. Implementing mathematical algorithms
    5. Performing basic arithmetic operations
    """
    
    improved_agent = llm_interface.improve_agent(agent, task_description)
    
    # Tambahkan alat yang mungkin belum ada
    if not any(tool.name == "power" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="power", function=power, description="Raise a number to a power"))
    
    if not any(tool.name == "sqrt" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="sqrt", function=sqrt, description="Calculate the square root of a number"))
    
    if not any(tool.name == "factorial" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="factorial", function=factorial, description="Calculate the factorial of a number"))
    
    if not any(tool.name == "is_prime" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="is_prime", function=is_prime, description="Check if a number is prime"))
    
    if not any(tool.name == "fibonacci" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="fibonacci", function=fibonacci, description="Calculate the nth Fibonacci number"))
    
    if not any(tool.name == "gcd" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="gcd", function=gcd, description="Calculate the GCD of two numbers"))
    
    if not any(tool.name == "lcm" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="lcm", function=lcm, description="Calculate the LCM of two numbers"))
    
    if not any(tool.name == "solve_equation" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="solve_equation", function=solve_equation, description="Solve a mathematical equation"))
    
    if not any(tool.name == "analyze_problem" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="analyze_problem", function=analyze_problem, description="Analyze a mathematical problem"))
    
    if not any(tool.name == "generate_problem" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="generate_problem", function=generate_problem, description="Generate a mathematical problem"))
    
    if not any(tool.name == "implement_algorithm" for tool in improved_agent.tools):
        improved_agent.add_tool(Tool(name="implement_algorithm", function=implement_algorithm, description="Implement a mathematical algorithm"))
    
    # Evaluasi agen yang ditingkatkan
    improved_score = evaluate_agent(improved_agent, task)
    print(f"\nImproved agent score: {improved_score:.4f}")
    
    # Cetak alat agen yang ditingkatkan
    print("\nImproved agent tools:")
    for tool in improved_agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen yang ditingkatkan
    print("\nImproved agent parameters:")
    print(f"  - Memory capacity: {improved_agent.memory_capacity}")
    print(f"  - Learning rate: {improved_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {improved_agent.exploration_rate:.4f}")
    
    # Demonstrasi pembuatan kode
    print("\n=== Code Generation Demo ===")
    function_code = code_generation.generate_function(
        function_name="solve_quadratic_equation",
        description="Solve a quadratic equation of the form ax^2 + bx + c = 0",
        parameters=[
            {"name": "a", "type": "float", "description": "Coefficient of x^2"},
            {"name": "b", "type": "float", "description": "Coefficient of x"},
            {"name": "c", "type": "float", "description": "Constant term"}
        ],
        return_type="tuple"
    )
    
    print("Generated function:")
    print(function_code)
    
    # Demonstrasi pemecahan masalah
    print("\n=== Problem Solving Demo ===")
    problem = "Find all positive integers n such that n^2 + 20 is a perfect square."
    
    print(f"Analyzing problem: {problem}")
    analysis = problem_solving.analyze_problem(problem)
    
    print("Problem analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Demonstrasi ekstraksi pengetahuan
    print("\n=== Knowledge Extraction Demo ===")
    text = """
    Mathematical problem-solving involves several key steps:
    
    1. Understanding the problem: Identify what is given and what needs to be found.
    2. Devising a plan: Select appropriate strategies or algorithms.
    3. Carrying out the plan: Apply the selected strategies correctly.
    4. Looking back: Verify the solution and reflect on the process.
    
    Different types of mathematical problems require different approaches:
    - Algebraic problems often involve manipulating equations and expressions.
    - Geometric problems deal with shapes, sizes, and properties of space.
    - Number theory problems focus on properties of integers.
    - Calculus problems involve rates of change and accumulation.
    
    Effective problem solvers use heuristics, visualization, and logical reasoning.
    """
    
    print("Extracting concepts...")
    concepts = knowledge_extraction.extract_concepts(text)
    
    print("Extracted concepts:")
    print(json.dumps(concepts[:2], indent=2))  # Tampilkan 2 konsep pertama
    
    # Demonstrasi modifikasi diri
    print("\n=== Self Modification Demo ===")
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
    
    print("Analyzing code...")
    analysis = self_modification.analyze_code(code)
    
    print("Code analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Demonstrasi generasi strategi evolusi
    print("\n=== Evolution Strategy Generation Demo ===")
    strategy_info = llm_interface.generate_evolution_strategy(task_description)
    
    print("Generated evolution strategy:")
    print(json.dumps(strategy_info, indent=2))
    
    # Demonstrasi generasi operator mutasi
    print("\n=== Mutation Operator Generation Demo ===")
    mutation_info = llm_interface.generate_mutation_operator(task_description)
    
    print("Generated mutation operator:")
    print(json.dumps(mutation_info, indent=2))
    
    # Demonstrasi generasi operator crossover
    print("\n=== Crossover Operator Generation Demo ===")
    crossover_info = llm_interface.generate_crossover_operator(task_description)
    
    print("Generated crossover operator:")
    print(json.dumps(crossover_info, indent=2))
    
    # Analisis hasil evolusi
    print("\n=== Evolution Results Analysis ===")
    evolution_results = {
        "initial_score": initial_score,
        "improved_score": improved_score,
        "best_score": improved_score,  # Gunakan improved_score sebagai best_score
        "evolution_time": 0.5,  # Nilai contoh
        "generations": 5,
        "population_size": 10,
        "best_agent_tools": [tool.name for tool in improved_agent.tools],
        "best_agent_parameters": {
            "memory_capacity": improved_agent.memory_capacity,
            "learning_rate": improved_agent.learning_rate,
            "exploration_rate": improved_agent.exploration_rate
        }
    }
    
    analysis = llm_interface.analyze_evolution_results(evolution_results)
    
    print("Evolution results analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Demonstrasi pembuatan kode kompleks
    print("\n=== Complex Code Generation Demo ===")
    class_code = code_generation.generate_class(
        class_name="MathematicalProblemSolver",
        description="A class for solving various mathematical problems",
        base_classes=["object"],
        methods=[
            {
                "name": "__init__",
                "description": "Initialize the problem solver",
                "parameters": [],
                "return_type": "None"
            },
            {
                "name": "solve_equation",
                "description": "Solve a mathematical equation",
                "parameters": [
                    {"name": "equation", "type": "str", "description": "The equation to solve"}
                ],
                "return_type": "str"
            },
            {
                "name": "analyze_problem",
                "description": "Analyze a mathematical problem",
                "parameters": [
                    {"name": "problem", "type": "str", "description": "The problem to analyze"}
                ],
                "return_type": "dict"
            },
            {
                "name": "generate_problem",
                "description": "Generate a mathematical problem",
                "parameters": [
                    {"name": "topic", "type": "str", "description": "The topic of the problem"},
                    {"name": "difficulty", "type": "str", "description": "The difficulty level"}
                ],
                "return_type": "dict"
            }
        ]
    )
    
    print("Generated class:")
    print(class_code)

if __name__ == "__main__":
    main()