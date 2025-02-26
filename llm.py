import ollama
import asyncio
from omegaconf import OmegaConf
from openai import AsyncOpenAI


def build_client(client_cfg):
    cfg = OmegaConf.load(f"configs/clients/{client_cfg}")
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
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    async def chat(self, prompt):
        async with self.semaphore:
            return await self._chat(prompt)

    async def _chat(self, prompt):
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
            messages=[{"role": "user", "content": prompt}], **self.cfg.params
        )
        return response["message"]["content"]


class OpenAIClient(BaseClient):
    def __init__(
        self,
        cfg,
    ):
        super().__init__(
            AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key),
            cfg,
        )

    async def _chat(self, prompt):
        response = await self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], **self.cfg.params
        )

        return response.choices[0].message.content
