"""FastAPI server tests. Mock scorer, CPU-only, no GPU."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bernoulli.labels import letter_token_ids
from bernoulli.server import create_app


class _FakeTokenizer:
    """Letter-token + chat-template mock sufficient for decide()."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if text.startswith(" ") and len(text) == 2 and text[1].isalpha():
            return [1000 + ord(text[1])]
        if text == " Yes":
            return [1200]
        if text == " No":
            return [1201]
        return [0, 0]

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> str:
        return str(next(m["content"] for m in conversation if m["role"] == "user"))


class _ConstantScorer:
    """Returns the same logits every call — strongly biased toward label index 0."""

    model_id = "test-model"
    revision = "test-rev"

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        # verify the fake tokenizer passes label-single-token checks
        letter_token_ids(self.tokenizer)

    def score(
        self,
        prompt: str,
        allowed_token_ids: list[int],
        *,
        images: list[str] | None = None,
    ) -> np.ndarray:
        logits = np.ones(len(allowed_token_ids), dtype=np.float32) * -5.0
        logits[0] = 10.0
        return logits

    def score_batch(
        self,
        prompts: list[str],
        allowed_token_ids_list: list[list[int]],
        *,
        images_list: list[list[str] | None] | None = None,
    ) -> list[np.ndarray]:
        return [self.score(p, a) for p, a in zip(prompts, allowed_token_ids_list, strict=True)]


@pytest.fixture
def client() -> TestClient:
    app = create_app(scorer=_ConstantScorer())
    return TestClient(app)


class TestHealthz:
    def test_returns_ok(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestModels:
    def test_lists_loaded_model(self, client: TestClient) -> None:
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["models"][0]["id"] == "test-model"
        assert body["models"][0]["revision"] == "test-rev"


class TestDecide:
    # debias=none disables permutation; the constant scorer is prompt-blind, so
    # the reverse-debiased average would collapse to uniform. Tests here are
    # shape/plumbing, not calibration — M3 covered that end.

    def test_choice_returns_shape(self, client: TestClient) -> None:
        r = client.post(
            "/v1/decide",
            json={
                "state": {"text": "customer email"},
                "questions": [
                    {
                        "id": "intent",
                        "type": "choice",
                        "prompt": "what?",
                        "options": ["refund", "exchange", "tracking"],
                    }
                ],
                "options": {"debias": "none", "calibrated": False},
            },
        )
        assert r.status_code == 200
        body = r.json()
        d = body["decisions"]["intent"]
        assert d["type"] == "choice"
        assert d["answer"] == "refund"  # constant scorer favors label[0]
        assert d["confidence"] > 0.9
        assert set(d["distribution"]) == {"refund", "exchange", "tracking"}
        assert body["model"] == "test-model"

    def test_binary_shape(self, client: TestClient) -> None:
        r = client.post(
            "/v1/decide",
            json={
                "state": {"text": "x"},
                "questions": [{"id": "urgent", "type": "binary", "prompt": "urgent?"}],
                "options": {"debias": "none", "calibrated": False},
            },
        )
        assert r.status_code == 200
        d = r.json()["decisions"]["urgent"]
        assert d["type"] == "binary"
        assert d["answer"] is True  # Yes is label[0] under the fake scorer
        assert d["probability"] > 0.9

    def test_rating_shape(self, client: TestClient) -> None:
        r = client.post(
            "/v1/decide",
            json={
                "state": {"text": "x"},
                "questions": [
                    {"id": "anger", "type": "rating", "prompt": "angry?", "scale": [1, 5]}
                ],
                "options": {"debias": "none", "calibrated": False},
            },
        )
        assert r.status_code == 200
        d = r.json()["decisions"]["anger"]
        assert d["type"] == "rating"
        assert set(d["distribution"]) == {"1", "2", "3", "4", "5"}
        # Constant scorer favors label 0 → integer 1 for scale (1, 5)
        assert d["expected"] < 1.5

    def test_multi_question(self, client: TestClient) -> None:
        r = client.post(
            "/v1/decide",
            json={
                "state": {"text": "x"},
                "questions": [
                    {"id": "a", "type": "choice", "prompt": "?", "options": ["x", "y"]},
                    {"id": "b", "type": "binary", "prompt": "?"},
                ],
                "options": {"debias": "none", "calibrated": False},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert set(body["decisions"]) == {"a", "b"}

    def test_invalid_payload_422(self, client: TestClient) -> None:
        # Missing required 'questions' field
        r = client.post("/v1/decide", json={"state": {"text": "x"}})
        assert r.status_code == 422

    def test_unknown_field_422(self, client: TestClient) -> None:
        # extra='forbid' on the pydantic models should reject unknown keys
        r = client.post(
            "/v1/decide",
            json={
                "state": {"text": "x"},
                "questions": [{"id": "a", "type": "binary", "prompt": "?"}],
                "options": {"debias": "reverse", "calibrated": False, "unknown_knob": 1},
            },
        )
        assert r.status_code == 422

    def test_images_501(self, client: TestClient) -> None:
        """Image state currently raises NotImplementedError → mapped to 501."""
        r = client.post(
            "/v1/decide",
            json={
                "state": {"text": "x", "images": ["base64fake"]},
                "questions": [{"id": "a", "type": "binary", "prompt": "?"}],
            },
        )
        assert r.status_code == 501
        assert "image" in r.json()["detail"].lower()


class _GeneratingScorer(_ConstantScorer):
    """ConstantScorer + a scripted .generate() so /v1/generate has something to parse."""

    def __init__(self, scripted: str = "A") -> None:
        super().__init__()
        self._scripted = scripted

    def generate(self, prompt: str, *, max_new_tokens: int = 10) -> str:
        return self._scripted


class TestGenerate:
    def test_choice_returns_shape(self) -> None:
        app = create_app(scorer=_GeneratingScorer(scripted="negative"))
        client = TestClient(app)
        r = client.post(
            "/v1/generate",
            json={
                "state": {"text": "whatever"},
                "questions": [
                    {
                        "id": "sentiment",
                        "type": "choice",
                        "prompt": "pos or neg?",
                        "options": ["negative", "positive"],
                    }
                ],
            },
        )
        assert r.status_code == 200, r.json()
        body = r.json()
        d = body["decisions"]["sentiment"]
        assert d["type"] == "choice"
        assert d["answer"] == "negative"
        assert d["distribution"] == {"negative": 1.0, "positive": 0.0}

    def test_501_when_scorer_lacks_generate(self, client: TestClient) -> None:
        """`_ConstantScorer` has no `.generate` → /v1/generate must 501, not 500."""
        r = client.post(
            "/v1/generate",
            json={
                "state": {"text": "x"},
                "questions": [{"id": "a", "type": "binary", "prompt": "?"}],
            },
        )
        assert r.status_code == 501
        assert "generate" in r.json()["detail"].lower()
