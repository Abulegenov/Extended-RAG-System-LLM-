from dotenv import load_dotenv
import gradio as gr
import request

# Gradio interface functions
def query_chat(message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    """Handle chat queries for Gradio."""
    if not message.strip():
        return history, ""
    
    try:
        response = requests.post('http://localhost:8000/api/query', 
                               json={'query': message}, 
                               timeout=None)
        
        if response.status_code == 200:
            data = response.json()
            route = data.get('route', 'No route returned')
            answer = data.get('answer', 'No answer returned')
        bot_response = f"**[Route: {route}]**\n\n{answer}"
        # print(bot_response)
    except Exception as e:
        bot_response = f"Error: {str(e)}"
        print(f"Chat error: {str(e)}")
    
    # history.append({"role": "user", "content": message})
    # history.append({"role": "assistant", "content": bot_response})
    history.append((message, bot_response))
    return history, ""

def clear_chat():
    """Clear the chat history."""
    return [], ""

def main():
# Create Gradio interface
    with gr.Blocks(title="HIPAA Query Assistant", theme=gr.themes.Soft()) as gradio_app:
        gr.Markdown(
            """
            # HIPAA Query Assistant
            
            Ask questions about HIPAA documentation. The system will automatically route your query to:
            - **SQL Search**: For specific articles, parts, or structured queries
            - **Semantic Search**: For conceptual questions and meaning-based queries
            """
        )
        
        chatbot = gr.Chatbot(
            height=500,
            show_label=False,
            elem_id="chatbot",
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your question",
                placeholder="Ask about HIPAA articles, regulations, or concepts...",
                scale=4,
                container=False
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", scale=1)
            
        gr.Examples(
            examples=[
                "What is Article 164.502 about?",
                "Does HIPAA mention encryption best practices?",
                "Show me all article number and titles in Part 164",
                "What are the potential civil penalties for noncompliance?",
                "If a covered entity outsources data processing, which sections apply?",    
            ],
            inputs=msg,
            label="Example Questions"
        )
        
        # Event handlers
        msg.submit(query_chat, [msg, chatbot], [chatbot, msg])
        submit_btn.click(query_chat, [msg, chatbot], [chatbot, msg])
        clear_btn.click(clear_chat, outputs=[chatbot, msg])

    return gradio_app

if __name__ == "__main__":
    print("Starting Gradio interface...")
    
    # Test backend connection
    try:
        test_response = requests.get('http://localhost:8000/health', timeout=3)
        print("Backend server is accessible")
    except:
        print("Warning: Backend server not accessible on localhost:8000")
    
    # Launch interface
    demo = main()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
        share=True  # Set to True if you want public link
    )

