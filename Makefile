# VGGT-Secure Makefile
#
# The Docker image is fully self-contained. 'make build' does everything:
# clones upstream, installs deps, downloads model, strips attack surface.

IMAGE := vggt-secure
TAG   := latest

.PHONY: help build run stop logs audit shell reconstruct export clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Build (one command, fully self-contained) ──────────────────────────

build: ## Build the image (downloads model, installs everything)
	docker build -t $(IMAGE):$(TAG) .

# ── Run ────────────────────────────────────────────────────────────────

run: ## Start API server via docker compose
	@test -f .env || (cp .env.example .env && echo "Created .env — edit VGGT_API_KEY first!")
	@mkdir -p data
	docker compose up -d
	@echo ""
	@echo "VGGT-Secure started."
	@echo "  Health: curl http://localhost:$$(grep VGGT_EXTERNAL_PORT .env 2>/dev/null | cut -d= -f2 || echo 8000)/api/v1/health"
	@echo "  Logs:   make logs"

run-direct: ## Run directly (no compose, no .env file needed)
ifndef VGGT_API_KEY
	$(error Set VGGT_API_KEY: make run-direct VGGT_API_KEY=your-key)
endif
	docker run --gpus all -p 8000:8000 \
		--read-only --tmpfs /tmp/vggt-secure:rw,noexec,nosuid,size=5g \
		--security-opt no-new-privileges --cap-drop ALL \
		-e VGGT_API_KEY=$(VGGT_API_KEY) \
		$(IMAGE):$(TAG)

stop: ## Stop the server
	docker compose down

logs: ## Tail server logs
	docker compose logs -f --tail=100

# ── CLI inside Docker ──────────────────────────────────────────────────

reconstruct: ## Reconstruct scene (SCENE=/path/to/images OUTPUT=/path/to/results)
ifndef SCENE
	$(error Usage: make reconstruct SCENE=/path/to/images OUTPUT=/path/to/results)
endif
	@mkdir -p $(or $(OUTPUT),./results)
	docker run --rm --gpus all \
		-v $(abspath $(SCENE)):/data/scene:ro \
		-v $(abspath $(or $(OUTPUT),./results)):/data/output \
		$(IMAGE):$(TAG) \
		reconstruct --scene_dir /data/scene --output /data/output

export: ## Export to SolidWorks (INPUT=/path/to/predictions.npz FMT=stl OUTPUT=/path)
ifndef INPUT
	$(error Usage: make export INPUT=predictions.npz FMT=stl OUTPUT=./exports)
endif
	@mkdir -p $(or $(OUTPUT),./exports)
	docker run --rm \
		-v $(abspath $(dir $(INPUT))):/data/in:ro \
		-v $(abspath $(or $(OUTPUT),./exports)):/data/out \
		$(IMAGE):$(TAG) \
		export -i /data/in/$(notdir $(INPUT)) -f $(or $(FMT),stl) -o /data/out

audit: ## Print security posture
	docker run --rm $(IMAGE):$(TAG) audit

# ── Maintenance ────────────────────────────────────────────────────────

shell: ## Debug shell inside container
	docker run --rm -it --gpus all --entrypoint /bin/bash $(IMAGE):$(TAG)

clean: ## Remove image and stopped containers
	docker compose down --rmi local --volumes 2>/dev/null || true
	docker rmi $(IMAGE):$(TAG) 2>/dev/null || true
	docker image prune -f
