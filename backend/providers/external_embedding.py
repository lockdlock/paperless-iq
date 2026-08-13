"""OpenAI-compatible external embedding provider."""

from __future__ import annotations

import httpx

from backend.providers.encryption import decrypt_credential


class ExternalEmbeddingProvider:
    """Embedding provider for OpenAI-compatible external APIs.

    This provider is intentionally independent from the LLM provider.
    It has its own URL, model, and API key.
    """

    def __init__(
        self,
        api_key_enc: bytes | str,
        model: str,
        secret_key: str,
        base_url: str,
        dimension: int | None = None,
    ) -> None:
        self._api_key_enc = api_key_enc
        self._model = model
        self._secret_key = secret_key
        self._base_url = base_url
        self._dimension = dimension

    async def embed(
        self,
        text: str,
        *,
        is_query: bool = False,
    ) -> list[float]:
        """Generate an embedding using an OpenAI-compatible API.

        ``is_query`` is accepted for provider-interface compatibility.
        OpenAI-compatible embedding APIs generally use the same endpoint
        for query and document embeddings.
        """
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
                    "input": text,
                    **(
                        {"dimensions": self._dimension}
                        if self._dimension is not None
                        else {}
                    ),
                },
            )

        response.raise_for_status()

        data = response.json()

        try:
            embedding = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "External embedding API returned an invalid response."
            ) from exc

        if self._dimension is not None and len(embedding) != self._dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch: "
                f"expected {self._dimension}, got {len(embedding)}."
            )

        return embedding

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
