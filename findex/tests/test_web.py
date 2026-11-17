import pytest
from fastapi.testclient import TestClient
from findex.web.routes import app
from findex.web.state import app_state


def test_service_unavailable_without_index():
    """Перевірка 503 помилки, якщо індекс не завантажений."""
    app_state.index = None
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503


def test_full_web_api_flow():
    """Інтеграційна перевірка всіх ендпоінтів із завантаженим індексом."""
    with TestClient(app) as client:
        # 1. Перевірка стану сервера
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        # 2. Перевірка статистики корпусу
        stats = client.get("/stats")
        assert stats.status_code == 200

        # 3. Перевірка головної HTML-сторінки
        home = client.get("/")
        assert home.status_code == 200
        assert "findex" in home.text

        # 4. Перевірка JSON-пошуку (BM25 та TfIdf)
        search_bm25 = client.get("/search", params={"q": "test", "k": 5, "scorer": "bm25"})
        assert search_bm25.status_code == 200
        data = search_bm25.json()
        assert data["query"] == "test"
        assert data["scorer"] == "bm25"
        assert isinstance(data["results"], list)

        search_tfidf = client.get("/search", params={"q": "test", "scorer": "tfidf", "page": 1})
        assert search_tfidf.status_code == 200
        assert search_tfidf.json()["scorer"] == "tfidf"

        # 5. Перевірка валідації порожнього запиту (422)
        bad_search = client.get("/search", params={"q": ""})
        assert bad_search.status_code == 422

        # 6. Перевірка отримання документа за ID та 404 для неіснуючого
        doc_ok = client.get("/docs/0")
        assert doc_ok.status_code == 200
        assert "content" in doc_ok.json()

        doc_missing = client.get("/docs/99999")
        assert doc_missing.status_code == 404


def test_global_exception_handler():
    """Перевірка, що непередбачувана помилка повертає 500 JSON без витоку стека."""
    with TestClient(app, raise_server_exceptions=False) as client:
        # Ламаємо стан індексу після старту, щоб викликати виняток всередині /stats
        app_state.index = "broken_index_object"  # type: ignore
        response = client.get("/stats")
        assert response.status_code == 500
        body = response.json()
        assert body["detail"] == "Internal Server Error"
        assert "error_id" in body