import gradio as gr
from utils.sentiment_analyzer import analyze_sentiment
from utils.authenticate import authenticate_gemini
from utils.user_intent import intent_detection
import time
import json
import re

# Authenticate with Gemini API
client = authenticate_gemini()

# Set up the Gemini model
model = "gemini-2.0-flash"

def process_chat(message, history):
    # Determine the user's intent using Gemini
    response = intent_detection(message, client, model)

    # Print the response for debugging
    print(f"Gemini response: {response}")

    # Extract JSON from markdown code block if present
    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(json_pattern, response)

    if match:
        json_str = match.group(1).strip()
        response_data = json.loads(json_str)
        
        # Check if the request is for sentiment analysis
        if response_data.get("intent") == "yes":
            sentiment = analyze_sentiment(message)
            
            # Generate response about sentiment
            if sentiment == "positive":
                return "The sentiment is positive."
            else:
                return "The sentiment is negative."
        
        # For non-sentiment queries, return the response field
        return response_data['response']
    else:
        # If no JSON found, return the original response
        return response

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value["value"])
    else:
        print("You downvoted this response: " + data.value["value"])

# Set up the Gradio interface with chat history
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(placeholder="<h1>How can I help you today?</h1>")
    chatbot.like(vote, None, None)
    gr.ChatInterface(
        chatbot=chatbot,
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