import ollama
import asyncio


class BaseClient:
    def __init__(self, client, model, temperature, max_tokens, num_ctx, max_requests=1):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.semaphore = asyncio.Semaphore(max_requests)

    async def chat(self, prompt):
        async with self.semaphore:
            return await self._chat(prompt)

    async def _chat(self, prompt):
        raise NotImplementedError

class OllamaClient(BaseClient):
    def __init__(
        self,
        host=None,
        model="llama3.2",
        temperature=0.0,
        num_ctx=32768,
        max_tokens=4096,
        max_requests=1,
    ):
        super().__init__(
            ollama.AsyncClient(host),
            model,
            temperature,
            max_tokens,
            num_ctx,
            max_requests,
        )

    async def _chat(self, prompt):
        response = await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        )
        return response["message"]["content"]
