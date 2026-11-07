import re

from ai_engine.llm_client import LLMClient, LLMError
from ai_engine.prompt_builder import INSUFFICIENT_INFO, PromptBuilder


class AnswerGenerator:
    def __init__(self, llm_client=None, prompt_builder=None):
        self.llm_client = llm_client or LLMClient()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def answer(self, question, retrieved_chunks):
        if not retrieved_chunks:
            return {"answer": INSUFFICIENT_INFO, "sources": [], "error": None}

        context = self.prompt_builder.build_context(retrieved_chunks)
        try:
            answer = self.llm_client.generate(
                self.prompt_builder.build_system_prompt(),
                self.prompt_builder.build_user_prompt(question, context),
            )
        except LLMError as error:
            return {
                "answer": "",
                "sources": self._sources(retrieved_chunks, cited=set()),
                "error": str(error),
            }

        cited = {int(n) for n in re.findall(r"\[Evidence\s+(\d+)\]", answer)}
        return {
            "answer": answer or INSUFFICIENT_INFO,
            "sources": self._sources(retrieved_chunks, cited),
            "error": None,
        }

    @staticmethod
    def _sources(chunks, cited):
        return [
            {
                "evidence_number": number,
                "document": chunk["document_name"],
                "page": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "excerpt": chunk["text"][:400],
                "cited": number in cited,
            }
            for number, chunk in enumerate(chunks, start=1)
        ]
