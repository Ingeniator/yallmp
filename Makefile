# Makefile

# Default target
.DEFAULT_GOAL := help

# Application name
APP_NAME := LLMProxy

# Docker image name
DOCKER_IMAGE := $(APP_NAME)

# Kubernetes directory
K8S_DIR := k8s/

# Colors for output
CYAN  := \033[36m
RESET := \033[0m

## ---------- General Commands ----------

.PHONY: help
help:  ## Show available commands
	@echo "$(CYAN)Available commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'


## ---------- Local Development ----------
.PHONY: run
run:  ## Run application
	uv run entrypoint.py

## ---------- Code Quality ----------

.PHONY: lint
lint:  ## Run linting (Python example)
	uvx ruff check --fix .

.PHONY: commit
commit:  ## Auto-format code (Python example)
	uvx --from commitizen cz c

## ---------- Testing ----------

.PHONY: test
test:  ## Run tests
	pytest

.PHONY: test-coverage
test-coverage:  ## Run tests with coverage report
	pytest --cov=.

## ---------- Deployment ----------

.PHONY: deploy
deploy:  ## Deploy application using Kubernetes
	kubectl apply -f $(K8S_DIR)

.PHONY: undeploy
undeploy:  ## Remove deployment from Kubernetes
	kubectl delete -f $(K8S_DIR)

.PHONY: logs
logs:  ## Show logs of the running application
	docker logs -f $(shell docker ps -q --filter ancestor=$(DOCKER_IMAGE))

## ---------- Docker commands ----------

.PHONY: docker-build
docker-build:  ## Build the Docker image
	docker build -t $(DOCKER_IMAGE) .

.PHONY: docker-run
docker-run:  ## Run the Docker container
	docker run -p 8080:8080 $(DOCKER_IMAGE)

.PHONY: docker-clean
docker-clean:  ## Remove Docker images and containers
	docker system prune -f

.PHONY: docker-stop
docker-stop:  ## Stop all running Docker containers
	docker ps -q | xargs -r docker stop

.PHONY: docker-shell
docker-shell:  ## Open an interactive shell inside the running container
	docker exec -it $(shell docker ps -q --filter ancestor=$(DOCKER_IMAGE)) /bin/sh

.PHONY: docker-restart
docker-restart:  ## Restart the application
	docker restart $(shell docker ps -q --filter ancestor=$(DOCKER_IMAGE))

.PHONY: docker-status
docker-status:  ## Show running containers
	docker ps
