# A Novel Expert-in-the-Loop Explainable Engineering Change Proposal Generation Framework: Multi-Agents for Cross-Departmental Information Aggregation and Consensus Reaching

This repository implements a knowledge-enhanced framework for generating engineering change proposals. Centered on multi-agent cross-departmental information aggregation and consensus reaching, the framework integrates interactive expert-in-the-loop mechanisms to enhance interpretability and practicality.


## System Architecture
The framework consists of two tightly integrated core modules:

1. **Intelligent Retrieval Module**: A knowledge graph retrieval system powered by LlamaIndex and Neo4j, enabling efficient knowledge management and reasoning.
2. **Dify Workflow Module**: A consensus-driven multi-agent framework that simulates realistic cross-departmental decision-making interactions.

## Core Functionalities
- **Knowledge Graph Construction**: Automatically extract entities and semantic relationships from engineering change order data to construct a structured knowledge graph.
- **Intelligent Search & Reasoning**: Support natural language-based graph queries, knowledge reasoning, and semantic retrieval.
- **Automated Cross-Departmental Evaluation Workflow**: A consensus-oriented multi-agent architecture designed to simulate the full lifecycle of cross-departmental evaluation. Domain-specific agents decompose complex decision-making processes into interpretable, role-based reasoning steps.

## Environmental Preparation

### 1. System Requirements
- Python 3.11 or higher
- Neo4j Graph Database
- Dify Platform v1.13.2 or higher

### 2. Install Dependencies
```bash
cd Intelligent\ Retrieval
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the environment template file:
```bash
cp .env.example .env
```

Edit the `.env` file to configure the following parameters:
```env
# Neo4j Configuration
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_URL=bolt://your_neo4j_host:7687

# Ollama Configuration
OLLAMA_MODEL=qwen3:30b-a3b-Instruct
OLLAMA_EMBEDDING_MODEL=bge-m3:latest
OLLAMA_REQUEST_TIMEOUT=3600

# Data Path Configuration
DATA_PATH=/path/to/your/change/order/data

# Cypher Query Configuration
CYPHER_QUERY="MATCH (comp:COMPONENT {name: $component_name})..."

# Schema Configuration
ENTITIES_LIST="CHANGE_ORDER,COMPONENT,DEPARTMENT,REASON,TIME_POINT"
RELATIONS_LIST="MODIFIES,SIGNED_BY,HAS_REASON,OCCURS_AT,PART_OF"
VALIDATION_SCHEMA="{\"CHANGE_ORDER\": [\"MODIFIES\", \"SIGNED_BY\", \"HAS_REASON\", \"OCCURS_AT\"], ...}"

# Prompt Configuration
EXTRACTION_PROMPT="You are a professional knowledge graph information extraction assistant..."
```

## Execution Steps

### Step 1: Build the Knowledge Graph
1. Prepare your engineering change order data and place it in the configured data path.
2. Run the knowledge graph construction script:
```bash
cd Intelligent\ Retrieval
python llamaindex.py
```

This script will:
- Load and parse change order data from the specified path
- Extract entities and relationships using a large language model
- Construct and persist the knowledge graph into the Neo4j database

### Step 2: Launch the API Service
Start the FastAPI-based knowledge retrieval service:
```bash
python run_app.py
```

Available API endpoints:
- `/query`: Execute natural language queries against the knowledge graph
- `/health`: Service health check

### Step 3: Configure the Dify Workflow
1. Import the `dify.yml` file via the **Import DSL** feature on the Dify platform.
2. Complete the workflow configuration:
   - Connect to the Ollama LLM service
   - Configure HTTP request nodes to interact with the knowledge graph API
   - Verify loop logic and conditional judgment rules


## Customization
- **Custom Cypher Query**: Modify the `CYPHER_QUERY` environment variable to define custom graph database query logic for specific business scenarios.
- **Custom Knowledge Schema**: Adjust `ENTITIES_LIST` and `RELATIONS_LIST` to define domain-specific entity and relationship types.
- **Custom Prompt**: Update `EXTRACTION_PROMPT` to optimize information extraction for specialized domains.


## Acknowledgments
This project is developed based on the following outstanding open-source projects. We sincerely thank the developers and communities for their contributions:
**LlamaIndex**: Provides core knowledge graph indexing, retrieval, and LLM orchestration functionalities.
- Repository: https://github.com/run-llama/llama_index
**Dify**: Supports the multi-agent workflow, consensus mechanism, and visual orchestration system.
- Official Website: https://dify.ai
- Repository: https://github.com/langgenius/dify
We fully comply with their respective open-source licenses and retain all original copyright statements.

## License Compliance
This project is licensed under the GPL-3.0 license.
The use of third-party libraries is subject to their own licenses:
LlamaIndex: MIT License
Dify: Apache License 2.0
