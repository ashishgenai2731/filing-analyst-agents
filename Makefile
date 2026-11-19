.PHONY: up down logs test clean restart status model ask eval-compare

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

restart:
	docker compose down && docker compose up --build -d

status:
	docker compose ps

test:
	python -m pytest -v

clean:
	docker compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

model:
	ollama pull qwen2.5:7b-instruct

ask:
	python -m scripts.ask_agents "$(Q)"

eval-compare:
	python -m src.eval.comparison_golden
	python -m src.eval.compare_systems --system multiagent
	python -m src.eval.compare_systems --system workflow
	python -m src.eval.compare_systems --system multiagent_text
	python -m src.eval.comparison_golden --seed 23 --out data/eval/comparison_holdout_qa.jsonl
	python -m src.eval.compare_systems --system multiagent --qa data/eval/comparison_holdout_qa.jsonl --tag _holdout
