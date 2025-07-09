"""
Contoh integrasi OpenAI API dengan Darwin-Gödel Machine.

Contoh ini mendemonstrasikan penggunaan OpenAI API untuk meningkatkan
kemampuan agen dalam DGM.
"""

import os
import random
import time
from typing import List, Dict, Any, Tuple

from simple_dgm.dgm import DGM
from simple_dgm.agents.base_agent import BaseAgent, Tool

# Coba impor OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not found. Installing...")
    os.system("pip install openai")
    try:
        import openai
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        print("Failed to install OpenAI package. Using mock LLM.")

# Definisikan fungsi alat
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    return x / y if y != 0 else 0

# Implementasi antarmuka LLM dengan OpenAI
class OpenAIInterface:
    """
    Antarmuka OpenAI untuk DGM.
    """
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """
        Inisialisasi antarmuka OpenAI.
        
        Args:
            api_key: API key OpenAI (opsional, jika tidak diberikan akan menggunakan environment variable)
            model: Model OpenAI yang akan digunakan
        """
        self.model = model
        
        # Dapatkan API key dari parameter atau environment variable
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Inisialisasi klien OpenAI
        if OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            self.available = True
        else:
            self.available = False
    
    def query(self, prompt: str) -> str:
        """
        Kirim kueri ke OpenAI.
        
        Args:
            prompt: Prompt untuk OpenAI
            
        Returns:
            Respons OpenAI
        """
        if not self.available:
            return "OpenAI API not available. Using mock response."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for a Darwin-Gödel Machine."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
            return f"Error: {str(e)}"
    
    def improve_agent(self, agent: BaseAgent) -> BaseAgent:
        """
        Tingkatkan agen menggunakan OpenAI.
        
        Args:
            agent: Agen yang akan ditingkatkan
            
        Returns:
            Agen yang ditingkatkan
        """
        # Buat prompt untuk OpenAI
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in agent.tools])
        
        prompt = f"""
        I have an agent with the following tools:
        {tools_description}
        
        The agent has the following parameters:
        - memory_capacity: {agent.memory_capacity}
        - learning_rate: {agent.learning_rate}
        - exploration_rate: {agent.exploration_rate}
        
        Please suggest improvements to this agent for mathematical tasks. 
        Specifically:
        1. What additional tools would be useful?
        2. What parameter values would be better?
        3. How can the agent's capabilities be enhanced?
        
        Format your response as follows:
        TOOLS: [list of additional tool names and descriptions]
        PARAMETERS: [suggested parameter values]
        ENHANCEMENTS: [other enhancement suggestions]
        """
        
        # Kirim prompt ke OpenAI
        response = self.query(prompt)
        print("\nOpenAI Response:")
        print(response)
        
        # Parse respons
        tools_section = ""
        parameters_section = ""
        enhancements_section = ""
        
        if "TOOLS:" in response:
            sections = response.split("PARAMETERS:")
            if len(sections) > 1:
                tools_section = sections[0].replace("TOOLS:", "").strip()
                remaining = sections[1]
                
                if "ENHANCEMENTS:" in remaining:
                    sections = remaining.split("ENHANCEMENTS:")
                    parameters_section = sections[0].strip()
                    enhancements_section = sections[1].strip()
                else:
                    parameters_section = remaining.strip()
        
        # Buat agen baru berdasarkan agen yang ada
        new_agent = BaseAgent(
            memory_capacity=agent.memory_capacity,
            learning_rate=agent.learning_rate,
            exploration_rate=agent.exploration_rate
        )
        
        # Salin alat dari agen lama
        for tool in agent.tools:
            new_agent.add_tool(tool)
        
        # Tambahkan alat baru berdasarkan respons OpenAI
        if "multiply" in tools_section.lower() and not any(t.name == "multiply" for t in new_agent.tools):
            new_agent.add_tool(Tool(
                name="multiply",
                function=multiply,
                description="Mengalikan dua angka"
            ))
        
        if "divide" in tools_section.lower() and not any(t.name == "divide" for t in new_agent.tools):
            new_agent.add_tool(Tool(
                name="divide",
                function=divide,
                description="Membagi dua angka"
            ))
        
        # Perbarui parameter berdasarkan respons OpenAI
        if "learning_rate" in parameters_section.lower():
            try:
                # Coba ekstrak nilai learning rate
                import re
                match = re.search(r"learning_rate:?\s*(0\.\d+)", parameters_section)
                if match:
                    new_agent.learning_rate = float(match.group(1))
            except:
                pass
        
        if "exploration_rate" in parameters_section.lower():
            try:
                # Coba ekstrak nilai exploration rate
                import re
                match = re.search(r"exploration_rate:?\s*(0\.\d+)", parameters_section)
                if match:
                    new_agent.exploration_rate = float(match.group(1))
            except:
                pass
        
        if "memory_capacity" in parameters_section.lower():
            try:
                # Coba ekstrak nilai memory capacity
                import re
                match = re.search(r"memory_capacity:?\s*(\d+)", parameters_section)
                if match:
                    new_agent.memory_capacity = int(match.group(1))
            except:
                pass
        
        return new_agent

# Fungsi evaluasi
def evaluate_agent(agent, task):
    """
    Evaluasi agen pada tugas.
    
    Args:
        agent: Agen yang akan dievaluasi
        task: Tugas untuk evaluasi
        
    Returns:
        Skor evaluasi
    """
    if "problems" not in task:
        return 0.0
    
    correct = 0
    total = len(task["problems"])
    
    for problem in task["problems"]:
        expression = problem["expression"]
        expected = problem["result"]
        
        try:
            # Parse expression
            parts = expression.split()
            if len(parts) == 3:
                a = float(parts[0])
                op = parts[1]
                b = float(parts[2])
                
                # Pilih alat berdasarkan operator
                if op == "+":
                    tool_name = "add"
                elif op == "-":
                    tool_name = "subtract"
                elif op == "*":
                    tool_name = "multiply"
                elif op == "/":
                    tool_name = "divide"
                else:
                    continue
                
                # Cari alat yang sesuai
                tool = None
                for t in agent.tools:
                    if t.name == tool_name:
                        tool = t
                        break
                
                if tool:
                    # Gunakan alat
                    result = tool.function(a, b)
                    
                    # Periksa apakah hasil mendekati yang diharapkan
                    if isinstance(result, (int, float)) and abs(result - expected) < 0.001:
                        correct += 1
            
        except Exception as e:
            print(f"Error evaluating {expression}: {e}")
    
    return correct / total if total > 0 else 0.0

# Buat tugas matematika
def create_math_task():
    """
    Buat tugas matematika.
    
    Returns:
        Tugas matematika
    """
    problems = []
    
    # Operasi aritmatika
    for _ in range(5):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        operations = [
            (f"{a} + {b}", a + b),
            (f"{a} - {b}", a - b),
            (f"{a} * {b}", a * b),
            (f"{a} / {b}", a / b if b != 0 else 0)
        ]
        problems.extend([{"expression": expr, "result": res} for expr, res in operations])
    
    return {
        "type": "arithmetic",
        "problems": problems
    }

def main():
    print("=== Darwin-Gödel Machine: OpenAI Integration Example ===")
    
    # Buat agen dasar dengan beberapa alat
    agent = BaseAgent(memory_capacity=5, learning_rate=0.01, exploration_rate=0.1)
    agent.add_tool(Tool(name="add", function=add, description="Tambahkan dua angka"))
    agent.add_tool(Tool(name="subtract", function=subtract, description="Kurangkan dua angka"))
    
    # Buat tugas
    task = create_math_task()
    
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
    
    # Inisialisasi DGM
    dgm = DGM(initial_agent=agent, population_size=10)
    
    # Inisialisasi antarmuka OpenAI
    openai_interface = OpenAIInterface()
    
    if not openai_interface.available:
        print("\nOpenAI API not available. Please check your API key.")
        return
    
    # Tingkatkan agen menggunakan OpenAI
    print("\nImproving agent using OpenAI...")
    improved_agent = openai_interface.improve_agent(agent)
    
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
    
    # Tambahkan agen yang ditingkatkan ke DGM
    agent_id = "improved_agent"
    dgm.archive[agent_id] = {
        "agent": improved_agent,
        "score": improved_score,
        "parent_id": None
    }
    
    # Jalankan evolusi
    print("\nEvolving for 5 generations...")
    dgm.evolve(generations=5, task=task)
    
    # Dapatkan agen terbaik
    best_agent = dgm.get_best_agent()
    
    # Evaluasi agen terbaik
    best_score = evaluate_agent(best_agent, task)
    print(f"\nBest agent score after evolution: {best_score:.4f}")
    
    # Cetak alat agen terbaik
    print("\nBest agent tools:")
    for tool in best_agent.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Cetak parameter agen terbaik
    print("\nBest agent parameters:")
    print(f"  - Memory capacity: {best_agent.memory_capacity}")
    print(f"  - Learning rate: {best_agent.learning_rate:.4f}")
    print(f"  - Exploration rate: {best_agent.exploration_rate:.4f}")

if __name__ == "__main__":
    main()