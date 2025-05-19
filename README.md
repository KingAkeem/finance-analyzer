# Financial Headlines Analyzer with Local LLM

## Features

- 📰 Fetches latest financial headlines from Yahoo Finance and Reuters
- 🔍 Analyzes headlines using Mistral 7B LLM via Ollama and performs research using SEC Filings with Llama3 LLM
- � Identifies key trends, sector categorization, and overall sentiment
- 🐳 Docker and Docker Compose support for easy deployment

## Prerequisites

- Docker
- Docker Compose
- NVIDIA GPU with ROCm support (for AMD) or CUDA (for NVIDIA)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/KingAkeem/financial-analyzer.git
cd financial-analyzer
```

### 2. Start the services with Docker Compose

```bash
docker-compose up -d ollama
```

```bash
docker exec ollama ollama pull mistral
```

```bash
docker exec ollama ollama pull llama3
```

This will start two services:
1. `ollama` - The Ollama service running Mistral 7B
2. `analyzer` - The financial headlines analyzer application

### 3. Run the analyzer

```bash
docker-compose up analyzer
```

### 4. View the output

The application will:
1. Fetch the latest financial headlines from Yahoo Finance and Reuters
2. Display the raw headlines.
3. Show the LLM's analysis including:
   - Key trends and insights
   - Sector categorization
   - Overall sentiment analysis
   - Provides research using SEC Filings

## Docker Setup Details

The project includes:

1. **Dockerfile**: Builds the Python environment with necessary dependencies
   - Based on `rocm/pytorch` image for GPU acceleration
   - Installs required Python packages

2. **docker-compose.yml**: Orchestrates two services:
   - `ollama`: Official Ollama service with automatic Mistral 7B model pull
   - `analyzer`: The headlines analyzer application

## Manual Installation (without Docker)

If you prefer to run locally:

1. Install Python 3.8+
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Ollama from [ollama.ai](https://ollama.ai) and pull Mistral:
```bash
ollama pull mistral
```

```bash
ollama pull llama3
```

```bash
ollama serve # if not running as a service
```

4. Run the application:
```bash
python main.py
```

## Troubleshooting

- **Ollama connection issues**: Ensure the Ollama service is running and accessible at `http://localhost:11434` (or the configured URL)
- **GPU acceleration**: Verify your GPU drivers (ROCm or CUDA) are properly installed