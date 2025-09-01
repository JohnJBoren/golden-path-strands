# 🚀 Golden Path Strands - Quick Start Guide

## Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
3. **gpt-oss:20b** model available

## Installation

### Automatic Setup

```bash
# Run the setup script
./setup.sh
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p logs datasets config

# Copy environment file
cp .env.example .env
```

## Start Ollama

```bash
# In a separate terminal, start Ollama server
ollama serve

# Pull the GPT-OSS model (20GB, one-time download)
ollama pull gpt-oss:20b
```

## Run the Framework

### Interactive Mode (Recommended for First Time)

```bash
python start.py
```

This launches an interactive menu where you can:
- Run research agents
- Generate code
- Analyze data
- Build custom workflows
- Discover golden paths

### Direct Mode Examples

#### Research Agent
```bash
python start.py --mode research --topic "Future of quantum computing"

# Or run the example directly
python examples/research_agent.py --topic "AI in healthcare" --depth comprehensive
```

#### Coding Agent
```bash
python start.py --mode coding --requirements "Create a REST API for user management"

# Or run the example directly
python examples/coding_agent.py --requirements "Sort algorithm" --language python
```

#### Interactive Research
```bash
python examples/research_agent.py --mode interactive
```

#### Interactive Coding
```bash
python examples/coding_agent.py --mode interactive
```

## Example Workflows

### 1. Research Workflow

```python
from golden_path import AgentOrchestrator

orchestrator = AgentOrchestrator()
result = await orchestrator.execute_research_workflow(
    topic="Machine learning trends",
    depth="comprehensive"
)
print(result["summary"])
```

### 2. Coding Workflow

```python
from golden_path import AgentOrchestrator

orchestrator = AgentOrchestrator()
result = await orchestrator.execute_coding_workflow(
    requirements="Binary search tree implementation",
    language="python",
    include_tests=True
)
print(result["code"])
```

### 3. Custom Agent Pipeline

```python
from golden_path import AgentOrchestrator

orchestrator = AgentOrchestrator()

# Define custom workflow
workflow = [
    {"agent": "planner", "task": "Design a web scraping system"},
    {"agent": "coder", "task": "Implement the scraper"},
    {"agent": "tester", "task": "Create test cases"},
    {"agent": "analyst", "task": "Optimize performance"}
]

results = await orchestrator.execute_sequential(workflow)
```

### 4. Golden Path Discovery

```python
from golden_path import GoldenPathStrands

gps = GoldenPathStrands()

tasks = [
    "Implement a caching system",
    "Design a microservices architecture",
    "Create a data pipeline"
]

results = await gps.run_complete_pipeline(tasks)
print(f"Discovered {results['discovered_paths']} optimal solutions")
```

## Available Agents

- **Research Agent**: Information gathering and analysis
- **Coding Agent**: Code generation and optimization
- **Analysis Agent**: Data analysis and insights
- **Planning Agent**: Task breakdown and project planning
- **Testing Agent**: Test case generation and quality assurance

## Configuration

Edit `.env` file to customize:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
DEFAULT_TEMPERATURE=0.7
MAX_TOKENS=4096
```

## Troubleshooting

### Ollama Not Running
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

### Model Not Found
```bash
# Pull the model
ollama pull gpt-oss:20b

# List available models
ollama list
```

### Python Dependencies Issues
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Next Steps

1. **Explore Examples**: Check the `examples/` directory
2. **Custom Agents**: Create your own specialized agents
3. **Fine-tuning**: Use discovered golden paths to train models
4. **Integration**: Connect with your existing workflows

## Support

- GitHub Issues: Report bugs or request features
- Documentation: See README.md for detailed documentation

---

Happy coding with Golden Path Strands! 🌟