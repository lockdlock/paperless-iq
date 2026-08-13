"""External reranker provider."""

from __future__ import annotations

import httpx

from backend.providers.encryption import decrypt_credential


class ExternalRerankerProvider:
    """Reranker provider for an external HTTP API.

    The external reranker is intentionally independent from the LLM provider.
    It has its own URL, model, and API key.

    The expected API request is:

        {
            "model": "...",
            "query": "...",
            "documents": ["...", "..."]
        }

    The expected response is:

        {
            "results": [
                {
                    "index": 0,
                    "relevance_score": 0.91
                },
                ...
            ]
        }

    ``index`` refers to the original document position, so the returned
    scores are reconstructed in the same order as the input passages.
    """

    def __init__(
        self,
        api_key_enc: bytes | str,
        model: str,
        secret_key: str,
        base_url: str,
    ) -> None:
        self._api_key_enc = api_key_enc
        self._model = model
        self._secret_key = secret_key
        self._base_url = base_url

    async def rerank(
        self,
        query: str,
        passages: list[str],
    ) -> list[float]:
        """Rerank passages using the configured external API."""

        if not passages:
            return []

        api_key = decrypt_credential(
            self._api_key_enc,
            self._secret_key,
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self._base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "query": query,
                    "documents": passages,
                },
            )

        response.raise_for_status()

        data = response.json()

        try:
            results = data["results"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "External reranker API returned an invalid response: "
                "'results' is missing."
            ) from exc

        if not isinstance(results, list):
            raise RuntimeError(
                "External reranker API returned an invalid response: "
                "'results' must be a list."
            )

        scores = [0.0] * len(passages)

        for item in results:
            if not isinstance(item, dict):
                raise RuntimeError(
                    "External reranker API returned an invalid result item."
                )

            index = item.get("index")
            relevance_score = item.get("relevance_score")

            if not isinstance(index, int):
                raise RuntimeError(
                    "External reranker API returned an invalid result index."
                )

            if index < 0 or index >= len(passages):
                raise RuntimeError(
                    f"External reranker API returned an out-of-range index: {index}."
                )

            try:
                score = float(relevance_score)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "External reranker API returned an invalid relevance_score."
                ) from exc

            scores[index] = max(0.0, min(1.0, score))

        return scores

    async def health_check(self) -> bool:
        """Return True when the configured credentials are usable."""
        try:
            api_key = decrypt_credential(
                self._api_key_enc,
                self._secret_key,
            )
            return bool(api_key and api_key.strip())
        except Exception:
            return False
