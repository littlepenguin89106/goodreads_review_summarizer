import ollama
import asyncio
from openai import AsyncOpenAI
import numpy as np
import re
from utils import logger


def build_llm_client(cfg):
    if cfg.client_type == "ollama":
        return OllamaClient(cfg)
    elif cfg.client_type == "openai":
        return OpenAIClient(cfg)
    else:
        raise ValueError(f"Invalid client type: {cfg.client_type}")


class BaseClient:
    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg
        self.max_concurrency = cfg.get("max_concurrency", 1)
        logger.info(f"max concurrency: {self.max_concurrency}")
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    async def chat(self, prompt):
        async with self.semaphore:
            return await self._chat(prompt)

    async def _chat(self, prompt):
        raise NotImplementedError

    async def embed(self, prompt):
        async with self.semaphore:
            return await self._embed(prompt)

    async def _embed(self, prompt):
        raise NotImplementedError


class OllamaClient(BaseClient):
    def __init__(
        self,
        cfg,
    ):
        super().__init__(
            ollama.AsyncClient(cfg.host),
            cfg,
        )

    async def _chat(self, prompt):
        response = await self.client.chat(
            messages=[{"role": "user", "content": prompt}], **self.cfg.llm.api_params
        )
        return response["message"]["content"]

    async def _embed(self, prompt):
        response = await self.client.embed(input=prompt, **self.cfg.embed.api_params)
        embeddings = np.array([res for res in response["embeddings"]])
        return embeddings


class OpenAIClient(BaseClient):
    def __init__(
        self,
        cfg,
    ):
        super().__init__(
            AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key),
            cfg,
        )

        think_start_tag = cfg.llm.get("think_start_tag", None)
        think_end_tag = cfg.llm.get("think_end_tag", None)

        if bool(think_start_tag) ^ bool(think_end_tag):
            raise ValueError(
                "You should either set both think_start_tag and think_end_tag, or unset both."
            )

        self.think = False
        if think_start_tag:
            self.think = True
            self.match_pattern = re.compile(
                rf"{re.escape(self.cfg.llm.think_start_tag)}(.*?){re.escape(self.cfg.llm.think_end_tag)}(.*)",
                flags=re.DOTALL,
            )

    async def _chat(self, prompt):
        response = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], **self.cfg.llm.api_params
        )
        content = response.choices[0].message.content
        if not self.think:
            return content

        match = re.match(self.match_pattern, content)
        if match:
            return match.group(2)

        logger.warning("Enable thinking extracting, but failed to extrant the content.")
        return content

    async def _embed(self, prompt):
        response = await self.client.embeddings.create(
            input=prompt, **self.cfg.embed.api_params
        )
        embeddings = np.array([res.embedding for res in response.data])
        return embeddings
