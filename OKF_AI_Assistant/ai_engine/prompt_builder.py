INSUFFICIENT_INFO = (
    "The available knowledge base does not contain enough information to answer this question."
)


class PromptBuilder:
    def build_system_prompt(self):
        return (
            "You are a careful document question-answering assistant.\n\n"
            "Rules:\n"
            "1. Use only the supplied evidence; never use outside knowledge.\n"
            "2. Cite the evidence for every factual claim, like [Evidence 1].\n"
            "3. If the evidence is insufficient, reply exactly: "
            f'"{INSUFFICIENT_INFO}"\n'
            "4. Do not invent facts, numbers, names or dates.\n"
            "5. If sources conflict, explain the conflict and cite both.\n"
            "6. Be concise."
        )

    def build_context(self, retrieved_chunks):
        sections = []
        for number, chunk in enumerate(retrieved_chunks, start=1):
            sections.append(
                f"[Evidence {number}]\n"
                f"Source document: {chunk['document_name']}\n"
                f"Page: {chunk['page_number']}\n"
                f"{chunk['text']}"
            )
        return "\n\n".join(sections)

    def build_user_prompt(self, question, context):
        return (
            "Answer the question using only the evidence below.\n\n"
            f'If the evidence does not contain enough information, say: "{INSUFFICIENT_INFO}"\n'
            "Cite evidence in this format: [Evidence 1], [Evidence 2].\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{context}"
        )
