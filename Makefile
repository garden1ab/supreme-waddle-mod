# VGGT-Secure Makefile
# Quick commands for Docker-based workflow

IMAGE_NAME := vggt-secure
IMAGE_TAG  := latest
MODEL_DIR  := ./model
DATA_DIR   := ./data

.PHONY: help build download-model run run-cli reconstruct export audit stop logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Build ───────────────────────────────────────────────────────────────

build: ## Build the Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# ── Model Management ───────────────────────────────────────────────────

download-model: ## Download model weights to ./model/
	@mkdir -p $(MODEL_DIR)
	docker run --rm \
		-v $(abspath $(MODEL_DIR)):/model \
		-e VGGT_MODEL_PATH=/model/model.pt \
		$(IMAGE_NAME):$(IMAGE_TAG) download-model
	@echo ""
	@echo "Model downloaded to $(MODEL_DIR)/model.pt"
	@echo "Copy the SHA-256 hash above into your .env as VGGT_MODEL_HASH"

# ── Server ──────────────────────────────────────────────────────────────

run: ## Start the API server (requires .env)
	docker compose up -d
	@echo ""
	@echo "VGGT-Secure API starting..."
	@echo "Health check: curl http://localhost:$${VGGT_EXTERNAL_PORT:-8000}/api/v1/health"
	@echo "Logs: make logs"

stop: ## Stop the API server
	docker compose down

logs: ## Tail server logs
	docker compose logs -f --tail=100

# ── CLI Operations ──────────────────────────────────────────────────────

reconstruct: ## Reconstruct a scene (usage: make reconstruct SCENE=/path/to/scene OUTPUT=/path/to/output)
ifndef SCENE
	$(error Set SCENE=/path/to/scene/dir)
endif
	@mkdir -p $(or $(OUTPUT),$(SCENE)/results)
	docker run --rm --gpus all \
		-v $(abspath $(MODEL_DIR))/model.pt:/model/model.pt:ro \
		-v $(abspath $(SCENE)):/data/scene:ro \
		-v $(abspath $(or $(OUTPUT),$(SCENE)/results)):/data/output \
		-e VGGT_MODEL_PATH=/model/model.pt \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		reconstruct --scene_dir /data/scene --output /data/output

export: ## Export to SolidWorks format (usage: make export INPUT=/path/to/predictions.npz FMT=stl OUTPUT=/path/to/out)
ifndef INPUT
	$(error Set INPUT=/path/to/predictions.npz)
endif
	@mkdir -p $(or $(OUTPUT),./exports)
	docker run --rm --gpus all \
		-v $(abspath $(dir $(INPUT))):/data/in:ro \
		-v $(abspath $(or $(OUTPUT),./exports)):/data/out \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		export -i /data/in/$(notdir $(INPUT)) -f $(or $(FMT),stl) -o /data/out

audit: ## Print security audit summary
	docker run --rm \
		-v $(abspath $(MODEL_DIR))/model.pt:/model/model.pt:ro \
		-e VGGT_MODEL_PATH=/model/model.pt \
		-e VGGT_API_KEY=$${VGGT_API_KEY:-} \
		$(IMAGE_NAME):$(IMAGE_TAG) audit

# ── Maintenance ─────────────────────────────────────────────────────────

shell: ## Open a shell in the container (debugging only)
	docker run --rm -it --gpus all \
		--entrypoint /bin/bash \
		-v $(abspath $(MODEL_DIR)):/model:ro \
		$(IMAGE_NAME):$(IMAGE_TAG)

clean: ## Remove Docker image and dangling layers
	docker compose down --rmi local --volumes 2>/dev/null || true
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	docker image prune -f

setup: ## First-time setup: build image, download model, create .env
	@echo "=== VGGT-Secure First-Time Setup ==="
	@echo ""
	@test -f .env || (cp .env.example .env && echo "Created .env from template")
	@mkdir -p $(MODEL_DIR) $(DATA_DIR)
	$(MAKE) build
	$(MAKE) download-model
	@echo ""
	@echo "=== Setup Complete ==="
	@echo "1. Edit .env and set VGGT_API_KEY (run: openssl rand -hex 32)"
	@echo "2. Copy the model SHA-256 hash into .env as VGGT_MODEL_HASH"
	@echo "3. Run: make run"
	@echo "4. Test: curl http://localhost:8000/api/v1/health"
