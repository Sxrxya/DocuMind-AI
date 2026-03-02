"""
DocuMind-AI — Prompt Builder

Constructs the final LLM prompt from retrieved context, conversation history,
and the user's question.
"""


SYSTEM_TEMPLATE = """You are DocuMind-AI, a helpful document assistant. Answer the user's question 
using ONLY the provided context. Follow these rules:

1. Base your answer strictly on the context below.
2. If the answer is not in the context, say: "I don't have enough information in the uploaded documents to answer that question."
3. Be concise but thorough.
4. If relevant, mention which part of the document your answer comes from.
"""

CONTEXT_TEMPLATE = """
--- CONTEXT ---
{context}
--- END CONTEXT ---
"""

HISTORY_TEMPLATE = """
--- CONVERSATION HISTORY ---
{history}
--- END HISTORY ---
"""


def build_prompt(
    query: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Build a chat-style message list for the LLM.

    Args:
        query:           The user's question.
        context_chunks:  Retrieved chunks from FAISS (each has "text" key).
        history:         Optional list of {"role": "user"|"assistant", "content": "..."}.

    Returns:
        List of message dicts suitable for OpenAI-style chat completion.
    """
    messages: list[dict] = []

    # System message
    system_content = SYSTEM_TEMPLATE.strip()

    # Add context
    if context_chunks:
        context_text = "\n\n---\n\n".join(
            f"[Chunk {i+1}] {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        )
        system_content += CONTEXT_TEMPLATE.format(context=context_text)

    # Add conversation history to system message
    if history:
        history_lines = []
        for msg in history[-6:]:  # keep last 3 exchanges
            role = msg.get("role", "user").capitalize()
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)
        system_content += HISTORY_TEMPLATE.format(history=history_text)

    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": query})

    return messages
