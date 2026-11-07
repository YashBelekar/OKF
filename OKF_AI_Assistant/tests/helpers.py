from ai_engine.llm_client import LLMError


class FakeLLM:
    """Deterministic stand-in for Ollama."""

    model = "fake"

    def __init__(self, answer="Solar panels convert sunlight to electricity [Evidence 1].",
                 concepts=None, fail=False):
        self.answer = answer
        self.concepts = concepts if concepts is not None else (
            '{"concepts":[{"title":"Solar Energy","description":"Energy from sunlight.","tags":["energy"]}],'
            '"entities":[{"title":"Acme Corp","description":"A solar company.","tags":["company"]}]}'
        )
        self.fail = fail
        self.prompts = []

    def generate(self, system_prompt, user_prompt, json_mode=False):
        if self.fail:
            raise LLMError("Cannot reach Ollama")
        self.prompts.append(user_prompt)
        return self.concepts if json_mode else self.answer

    def is_available(self):
        return not self.fail

    def list_models(self):
        return [self.model]
