
import gradio as gr
from agent import get_agent_response_stream


async def chat(message, history):
    chat_history = []

    for item in history or []:
        if item.get("role") in ("user", "assistant"):
            chat_history.append({
                "role": item["role"],
                "content": item["content"]
            })

    response = ""

    async for chunk in get_agent_response_stream(
        question=message,
        chat_history=chat_history
    ):
        response += chunk
        yield response


demo = gr.ChatInterface(
    fn=chat,
    title="AI Engineering & Machine Learning Tutor",
    description="Ask questions about AI/ML concepts and AI engineering.",
    chatbot=gr.Chatbot(),
    
)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=3000
    )
