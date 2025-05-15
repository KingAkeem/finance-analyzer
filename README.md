Here's a comprehensive `README.md` for your GitHub repository:

# Financial Headlines Analyzer with Local LLM

## Features

- 📰 Fetches latest financial headlines from Yahoo Finance
- 🔍 Analyzes headlines using Mistral 7B LLM via Ollama
- � Identifies key trends, sector categorization, and overall sentiment
- 🐳 Docker and Docker Compose support for easy deployment

## Prerequisites

- Docker
- Docker Compose
- NVIDIA GPU with ROCm support (for AMD) or CUDA (for NVIDIA)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/financial-headlines-analyzer.git
cd financial-headlines-analyzer
```

### 2. Start the services with Docker Compose

```bash
docker-compose up -d ollama
```

This will start two services:
1. `ollama` - The Ollama service running Mistral 7B
2. `analyzer` - The financial headlines analyzer application

### 3. Run the analyzer

```bash
docker-compose exec analyzer python main.py
```

### 4. View the output

The application will:
1. Fetch the latest financial headlines from Yahoo Finance
2. Display the raw headlines
3. Show the LLM's analysis including:
   - Key trends and insights
   - Sector categorization
   - Overall sentiment analysis

## Configuration

You can modify the following aspects of the application:

- **LLM Model**: Change the model in `main.py` (default: "mistral")
- **Ollama URL**: Modify the base_url in `main.py` if needed
- **Headline Source**: Change the URL in `get_headlines()` function

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

4. Run the application:
```bash
python main.py
```

## Troubleshooting

- **Ollama connection issues**: Ensure the Ollama service is running and accessible at `http://localhost:11434` (or the configured URL)
- **GPU acceleration**: Verify your GPU drivers (ROCm or CUDA) are properly installed
- **Rate limiting**: If Yahoo Finance blocks requests, try modifying the User-Agent in `get_headlines()`

## License

MIT

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

This README provides:
1. Clear project description
2. Installation instructions for both Docker and manual setup
3. Usage examples
4. Configuration options
5. Troubleshooting tips
6. License and contribution information

You may want to customize:
- The repository URL
- License information
- Any additional configuration options specific to your setup
- Contribution guidelines if you want to open the project to contributorsGG