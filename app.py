import gradio as gr
from google import genai
import os
from dotenv import load_dotenv
from utils.sentiment_analyzer import analyze_sentiment
from gradio import ChatMessage
import time

# Load environment variables from .env file
load_dotenv()

# Set the Google API key from the environment variable
gemini_api_key = os.getenv("gemini_api_key")

# Authenticate with the Google Gemini API
client = genai.Client(api_key=gemini_api_key)

# Set up the Gemini model
model = "gemini-2.5-flash"

def process_chat(message, history):
    # Create context from history
    # conversation_history = "\n".join([f"User: {user}\nAI: {ai}" for user, ai in history])
    
    # Create system prompt to instruct the model
    system_prompt = """You are an AI assistant that can determine sentiment in text. 
    If the user asks about sentiment in a sentence or provides a statement for sentiment analysis, 
    use the analyze_sentiment tool to determine if it's positive or negative, then respond accordingly.
    For other queries, respond helpfully as a general AI assistant."""
    
    # Check if user is asking about sentiment
    sentiment_prompt = f"""
    {system_prompt}
    
    Conversation history:
    conversation_history
    
    User's latest message: {message}
    
    First, determine if the user is asking about sentiment analysis or wants you to analyze the sentiment of a sentence.
    If yes, respond only yes.
    If no, respond to the query directly.
    """
    
    # Get response from Gemini
    # response = client.models.generate_content(model = model, contents = sentiment_prompt)
    response = "test"  # Placeholder for actual API call
    
    # Check if the request is for sentiment analysis
    if "sentiment" in message.lower() or any(phrase in message.lower() for phrase in ["how do i feel", "is this positive", "is this negative"]):
        # Extract text to analyze - assume the entire message for simplicity
        # In a more advanced implementation, you could use Gemini to extract the relevant text
        sentiment = analyze_sentiment(message)
        
        # Generate response about sentiment
        if sentiment == "positive":
            time.sleep(0.5)  # Simulate processing time
            history.append(
                ChatMessage(
                    role="assistant",
                    content="Sentence sentiment is positive.",
                    metadata={"title": "Using Tool 'BERT'"}
                )
            )
            yield history
            time.sleep(0.5)  # Simulate processing time
            history.append(
                ChatMessage(
                    role="assistant",
                    content="I analyzed the sentiment in your message, and it seems positive! 😊 The text has an upbeat or favorable tone."
                )
            )
            yield history
            return
        else:
            time.sleep(0.5)
            history.append(
                ChatMessage(
                    role="assistant",
                    content="Sentence sentiment is negative.",
                    metadata={"title": "Using Tool 'BERT'"}
                )
            )
            yield history
            time.sleep(0.5)  # Simulate processing time
            history.append(
                ChatMessage(
                    role="assistant",
                    content="I analyzed the sentiment in your message, and it seems negative. 😞 The text has a somewhat downbeat or unfavorable tone."
                )
            )
            yield history
            return
    
    # For non-sentiment queries, return the Gemini response
    history.append(
        ChatMessage(
            role="assistant",
            content=response
        )
    )
    yield history

# Set up the Gradio interface with chat history
demo = gr.ChatInterface(
    fn=process_chat,
    type = "messages",
    title="Gemini AI Assistant with Sentiment Analysis",
    description="Chat with Gemini AI. Ask about sentiment in text or any other questions!",
    examples=[
        ["Can you analyze the sentiment in this sentence: I love this product!"],
        ["What's the sentiment of: This movie was disappointing."],
        ["Tell me about sentiment analysis"],
        ["What's the weather like today?"]
    ],
    run_examples_on_click=False,
    save_history=True,
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()