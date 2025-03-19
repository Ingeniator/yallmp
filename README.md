# YALLMP (Yet Another LLM Proxy)

A proxy server for Language Learning Models (LLMs) with built-in circuit breaking, rate limiting, and observability.

## Features

- **Proxy Service**
  - Configurable circuit breaker
  - Exponential backoff retry mechanism

- **Chain Hub**
  - Load and execute LangChain chains

- **Prompt Hub**
  - Centralized prompt management
  - Prompt templating
  - Variable substitution

- **Observability**
  - Health check endpoints
  - Structured logging
  - Request/Response metrics

## Quick Start

### Prerequisites
- Python 3.13+
- Make

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Ingeniator/yallmp.git
cd yallmp
```

2. Install dependencies:
```bash
make dev-init
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Running the Service

```bash
make run
```

The service will start on http://localhost:8000 by default.

## API Documentation

### Core Endpoints

- `GET /health` - Service health check
- `POST /proxy/{path}` - Proxy LLM requests

### Chain Hub

- `POST /chain/{name}` - Execute a chain
- `GET /chains` - List available chains

### Prompt Hub

- `POST /prompt/{name}` - Format a prompt
- `GET /prompts` - List available prompts

## Development

### Setup
```bash
make dev-init
```

### Running
```bash
make run
```

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code and lint
make lint
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests and linting
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
