from __future__ import annotations

from graphkv_quality.provider import GeminiClient


class Response:
    ok = True
    status_code = 200
    headers = {}

    @staticmethod
    def json():
        return {
            "candidates": [{"content": {"parts": [{"text": "Paris"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1},
        }


class Session:
    @staticmethod
    def post(*args, **kwargs):
        return Response()

    @staticmethod
    def get(*args, **kwargs):
        response = Response()
        response.json = lambda: {
            "name": "models/test",
            "supportedGenerationMethods": ["generateContent"],
            "inputTokenLimit": 100,
            "outputTokenLimit": 20,
        }
        return response


def test_provider_parses_text_and_usage():
    client = GeminiClient("test", api_key="secret", session=Session())
    assert client.preflight()["name"] == "models/test"
    generation = client.generate("question", max_output_tokens=5)
    assert generation.text == "Paris"
    assert generation.input_tokens == 10
    assert generation.output_tokens == 1
