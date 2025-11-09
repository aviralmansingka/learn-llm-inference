set dotenv-load

# Serve the Modal app
serve:
    uv run modal serve deploy/serve.py

# Run FastAPI development server
dev:
    uv run fastapi dev src/modal_llm/web/fastapi.py

# Build the project (compile, type check, etc.)
build:
    uv run python -m py_compile src/
    uv run pyright src/ --outputjson || true

# Run tests
test:
    uv run pytest test/ -v

# Run all checks before deployment
check: build test

# Clean build artifacts
clean:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

req:
  xh POST ${APP_URL}/v1/chat/completions \
    model=gpt-4 \
    messages:='[{"role": "user", "content": "Hello"}]' \
    temperature:=0.7

# Show all available commands
help:
    @just --list
