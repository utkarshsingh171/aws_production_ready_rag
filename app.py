import gradio as gr
from agent import get_agent_response_stream

# Hides the "Built with Gradio" footer and the generation timer text.
CSS = """
footer { display: none !important; }
.meta-text, .meta-text-center, .progress-text, .timer { display: none !important; }

/* three loading dots */
.dots span {
  display: inline-block; width: 8px; height: 8px; margin: 0 3px;
  border-radius: 50%; background: currentColor; opacity: .3;
  animation: blink 1.2s infinite;
}
.dots span:nth-child(2) { animation-delay: .2s; }
.dots span:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%, 80%, 100% { opacity: .3; } 40% { opacity: 1; } }
"""

DOTS = '<span class="dots"><span></span><span></span><span></span></span>'


def _user_id(request: gr.Request) -> str:
    """One id per browser session (limits are per session)."""
    return getattr(request, "session_hash", None) or "anonymous"


async def chat(message, history, request: gr.Request):
    chat_history = []

    for item in history or []:
        if item.get("role") in ("user", "assistant"):
            chat_history.append({
                "role": item["role"],
                "content": item["content"]
            })

    response = ""
    yield DOTS 
    async for chunk in get_agent_response_stream(
        question=message,
        chat_history=chat_history,
        user_id=_user_id(request),
    ):
        response += chunk
        yield response


demo = gr.ChatInterface(
    fn=chat,
    title="AI Engineering & Machine Learning Tutor",
    description="Ask questions about AI/ML concepts and AI engineering.",
    chatbot=gr.Chatbot(type="messages"),
    type="messages",
    show_progress="hidden",
    css=CSS,
)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=3000,
        show_api=False,  # removes the "Use via API" button
    )