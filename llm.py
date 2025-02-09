import ollama


class OllamaClient:
    def __init__(
        self,
        host=None,
        model="llama3.2",
        temperature=0.0,
        num_ctx=32768,
        max_tokens=4096,
    ):
        self.client = ollama.Client(host)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx

    def generate(self, prompt):
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        )
        return response["message"]["content"]
