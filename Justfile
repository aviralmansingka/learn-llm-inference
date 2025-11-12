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
    messages:='[{"role": "user", "content": "Hello, can you tell me a joke?"}]' \
    temperature:=0.7

# Multi-turn conversation request 1
req1:
    xh POST ${APP_URL}/v1/chat/completions \
        model=gpt-4 \
        messages:='[{"role": "system", "content": "You are a helpful assistant."}, \
                   {"role": "user", "content": "Hello, who won the World Cup in 2018?"}, \
                   {"role": "assistant", "content": "France won the World Cup in 2018."}, \
                   {"role": "user", "content": "Who was the top scorer?"}]' \
        temperature:=0.7

# Multi-turn conversation request 2
req2:
    xh POST ${APP_URL}/v1/chat/completions \
        model=gpt-4 \
        messages:='[{"role": "system", "content": "You are a helpful assistant."}, \
                   {"role": "user", "content": "What is the capital of Japan?"}, \
                   {"role": "assistant", "content": "The capital of Japan is Tokyo."}, \
                   {"role": "user", "content": "Can you tell me a fun fact about Tokyo?"}]' \
        temperature:=0.7


# Show all available commands
help:
    @just --list
