SYSTEM_PROMPT = (
    "You are a precise document Q&A assistant. Use ONLY the supplied context. "
    "Never invent facts, sources, or page numbers. Every factual claim must have "
    "a citation in the exact format [Source: <chunk UUID>]. If the context is "
    "insufficient, reply exactly: 'I don't have enough information.'"
)

USER_PROMPT_TEMPLATE = """
Context:
{% for ctx in contexts %}
[{{ loop.index }}] Chunk ID: {{ ctx.chunk_id }}
Page: {{ ctx.page_number if ctx.page_number is not none else 'N/A' }}
Content:
{{ ctx.content }}
---
{% endfor %}

Question: {{ query }}

Answer concisely. Cite every factual claim with the exact chunk UUID.
"""


class PromptBuilder:
    def build(self, query: str, contexts: list[dict]) -> list[dict[str, str]]:
        from jinja2 import Template

        user_message = Template(USER_PROMPT_TEMPLATE).render(query=query, contexts=contexts)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
