"""AWS Bedrock embedding provider implementation for ChunkHound."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import boto3
import httpx
from loguru import logger

from chunkhound.core.utils import EMBEDDING_CHARS_PER_TOKEN
from chunkhound.interfaces.embedding_provider import EmbeddingConfig, RerankResult

# Known dims for common Bedrock embedding models; auto-detected otherwise.
_BEDROCK_MODEL_DIMS = {
    "amazon.titan-embed-text-v1": 1536,
    "amazon.titan-embed-text-v2:0": 1024,
    "cohere.embed-english-v3": 1024,
    "cohere.embed-multilingual-v3": 1024,
}


class BedrockEmbeddingProvider:
    """AWS Bedrock embedding provider using InvokeModel API."""

    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str = "us-east-1",
        model: str = "amazon.titan-embed-text-v2:0",
        batch_size: int = 100,
        timeout: int = 30,
        retry_attempts: int = 3,
        rerank_url: str | None = None,
        rerank_model: str | None = None,
        rerank_format: str = "auto",
        rerank_batch_size: int | None = None,
        rerank_ssl_verify: bool = True,
    ) -> None:
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_region = aws_region
        self._model = model
        self._batch_size = (
            1 if model.startswith("amazon.titan") else min(batch_size, 96)
        )
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._dims: int | None = _BEDROCK_MODEL_DIMS.get(model)
        self._client = None
        self._rerank_url = rerank_url
        self._rerank_model = rerank_model
        self._rerank_format = rerank_format
        self._rerank_batch_size = rerank_batch_size
        self._rerank_ssl_verify = rerank_ssl_verify
        self._usage_stats = {
            "requests_made": 0,
            "embeddings_generated": 0,
            "errors": 0,
        }

    def _get_client(self):
        if self._client is None:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._aws_region,
                aws_access_key_id=self._aws_access_key_id,
                aws_secret_access_key=self._aws_secret_access_key,
            )
        return self._client

    # --- Properties ---

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def model(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        if self._dims is None:
            raise RuntimeError(
                "Embedding dimensions not yet detected. Call initialize() first."
            )
        return self._dims

    @property
    def distance(self) -> str:
        return "cosine"

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_tokens(self) -> int | None:
        return 8192

    @property
    def config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider="bedrock",
            model=self._model,
            dims=self._dims or 1024,
            distance=self.distance,
            batch_size=self._batch_size,
            max_tokens=self.max_tokens,
            timeout=self._timeout,
            retry_attempts=self._retry_attempts,
        )

    # --- Lifecycle ---

    async def initialize(self) -> None:
        if self._dims is None:
            probe = await self.embed_single("probe")
            self._dims = len(probe)
            logger.debug(
                f"Bedrock provider initialized: model={self._model}, dims={self._dims}"
            )

    async def shutdown(self) -> None:
        self._client = None

    def is_available(self) -> bool:
        return bool(self._aws_access_key_id and self._aws_secret_access_key)

    async def health_check(self) -> dict[str, Any]:
        try:
            await self.embed_single("health check")
            return {"status": "ok", "provider": "bedrock", "model": self._model}
        except Exception as e:
            return {
                "status": "error",
                "provider": "bedrock",
                "model": self._model,
                "error": str(e),
            }

    # --- Core embedding ---

    def _invoke_titan(self, text: str) -> list[float]:
        client = self._get_client()
        body = json.dumps({"inputText": text})
        response = client.invoke_model(
            modelId=self._model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        self._usage_stats["requests_made"] += 1
        self._usage_stats["embeddings_generated"] += 1
        return result["embedding"]

    def _invoke_cohere(
        self, texts: list[str], input_type: str = "search_document"
    ) -> list[list[float]]:
        client = self._get_client()
        body = json.dumps({"texts": texts, "input_type": input_type})
        response = client.invoke_model(
            modelId=self._model,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        self._usage_stats["requests_made"] += 1
        self._usage_stats["embeddings_generated"] += len(texts)
        return result["embeddings"]

    def _is_titan(self) -> bool:
        return self._model.startswith("amazon.titan")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self.embed_batch(texts)

    async def embed_single(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        if self._is_titan():
            return await loop.run_in_executor(None, self._invoke_titan, text)
        else:
            results = await loop.run_in_executor(None, self._invoke_cohere, [text])
            return results[0]

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        if not texts:
            return []

        bs = batch_size or self._batch_size
        loop = asyncio.get_event_loop()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            if self._is_titan():
                for text in chunk:
                    emb = await loop.run_in_executor(None, self._invoke_titan, text)
                    all_embeddings.append(emb)
            else:
                embeddings = await loop.run_in_executor(
                    None, self._invoke_cohere, chunk
                )
                all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_streaming(self, texts: list[str]) -> AsyncIterator[list[float]]:
        for emb in await self.embed_batch(texts):
            yield emb

    # --- Reranking (via external HTTP service) ---

    def supports_reranking(self) -> bool:
        return self._rerank_url is not None

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        if not self._rerank_url:
            raise NotImplementedError(
                "Bedrock provider requires rerank_url to be configured for reranking"
            )
        return await self._rerank_via_http(query, documents, top_k)

    async def _rerank_via_http(
        self, query: str, documents: list[str], top_k: int | None
    ) -> list[RerankResult]:
        batch_limit = self.get_max_rerank_batch_size()

        if len(documents) <= batch_limit:
            results = await self._rerank_http_batch(query, documents, top_k)
            if top_k is not None:
                results = results[:top_k]
            return results

        all_results: list[RerankResult] = []
        for start in range(0, len(documents), batch_limit):
            batch = documents[start : start + batch_limit]
            batch_results = await self._rerank_http_batch(query, batch, top_k=None)
            for r in batch_results:
                all_results.append(RerankResult(index=r.index + start, score=r.score))

        all_results.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            all_results = all_results[:top_k]
        return all_results

    async def _rerank_http_batch(
        self, query: str, documents: list[str], top_k: int | None
    ) -> list[RerankResult]:
        assert self._rerank_url is not None
        payload = self._build_rerank_payload(query, documents, top_k)
        logger.debug(
            f"HTTP reranking {len(documents)} documents at {self._rerank_url} "
            f"(format={self._rerank_format})"
        )

        async with httpx.AsyncClient(
            timeout=self._timeout, verify=self._rerank_ssl_verify
        ) as client:
            response = await client.post(
                self._rerank_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

        if isinstance(data, list):
            data = {"results": data}
        if isinstance(data, dict) and "error" in data:
            raise ValueError(f"Rerank service error: {data['error']}")
        return self._parse_rerank_response(data, len(documents))

    def _build_rerank_payload(
        self, query: str, documents: list[str], top_k: int | None
    ) -> dict:
        fmt = self._rerank_format
        if fmt == "tei":
            return {"query": query, "texts": documents}
        elif fmt == "cohere":
            payload: dict = {"query": query, "documents": documents}
            if self._rerank_model:
                payload["model"] = self._rerank_model
            if top_k is not None:
                payload["top_n"] = top_k
            return payload
        else:  # auto: try Cohere if model provided, else TEI
            if self._rerank_model:
                payload = {
                    "query": query,
                    "documents": documents,
                    "model": self._rerank_model,
                }
                if top_k is not None:
                    payload["top_n"] = top_k
                return payload
            return {"query": query, "texts": documents}

    def _parse_rerank_response(
        self, data: dict, num_documents: int
    ) -> list[RerankResult]:
        if "results" not in data:
            raise ValueError(
                "Invalid rerank response: missing 'results' field. "
                f"Got: {list(data.keys())}"
            )
        results = []
        for item in data["results"]:
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict rerank result: {item!r}")
                continue
            idx = item.get("index")
            score = (
                item.get("relevance_score")
                if "relevance_score" in item
                else item.get("score")
            )
            if idx is None or score is None:
                logger.warning(f"Skipping malformed rerank result: {item}")
                continue
            if not (0 <= idx < num_documents):
                logger.warning(
                    f"Rerank index {idx} out of range ({num_documents} docs), skipping"
                )
                continue
            results.append(RerankResult(index=idx, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # --- Validation / token helpers ---

    def validate_texts(self, texts: list[str]) -> list[str]:
        return [t for t in texts if t and t.strip()]

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // EMBEDDING_CHARS_PER_TOKEN)

    def chunk_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        max_chars = max_tokens * EMBEDDING_CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return [text]
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    # --- Metadata ---

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self._model,
            "dims": self._dims,
            "distance": self.distance,
            "batch_size": self._batch_size,
            "max_tokens": self.max_tokens,
            "region": self._aws_region,
        }

    def get_usage_stats(self) -> dict[str, Any]:
        return dict(self._usage_stats)

    def reset_usage_stats(self) -> None:
        for key in self._usage_stats:
            self._usage_stats[key] = 0

    def update_config(self, **kwargs: Any) -> None:
        if "batch_size" in kwargs:
            self._batch_size = kwargs["batch_size"]
        if "timeout" in kwargs:
            self._timeout = kwargs["timeout"]

    def get_supported_distances(self) -> list[str]:
        return ["cosine"]

    def get_optimal_batch_size(self) -> int:
        return self._batch_size

    def get_max_tokens_per_batch(self) -> int:
        return (self.max_tokens or 8192) * self._batch_size

    def get_max_documents_per_batch(self) -> int:
        return self._batch_size

    def get_recommended_concurrency(self) -> int:
        return 4

    def get_max_rerank_batch_size(self) -> int:
        return self._rerank_batch_size if self._rerank_batch_size is not None else 32
