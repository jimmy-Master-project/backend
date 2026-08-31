class BaseLLMClient:
    def generate(self, prompt: str):
        raise NotImplementedError
