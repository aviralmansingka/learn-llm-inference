import pytest
from fastapi.testclient import TestClient
from modal_llm.web.web import asgi_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(asgi_app)


def test_chat_completions_basic_request(client):
    """Test that the /v1/chat/completions endpoint accepts a basic request."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
    )

    assert response.status_code == 200

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "gpt-4"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in data
    assert "id" in data


def test_chat_completions_with_temperature(client):
    """Test that temperature parameter is accepted."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.5
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4"


def test_chat_completions_with_max_tokens(client):
    """Test that max_tokens parameter is accepted."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4"


def test_chat_completions_missing_model(client):
    """Test that missing model field returns validation error."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
    )

    assert response.status_code == 422  # Unprocessable Entity


def test_chat_completions_missing_messages(client):
    """Test that missing messages field returns validation error."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4"
        }
    )

    assert response.status_code == 422  # Unprocessable Entity
