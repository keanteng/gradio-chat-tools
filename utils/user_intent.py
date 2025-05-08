from google import genai

def intent_detection(message, client, model):
    prompt = f"""
        User's message: {message}
        
        First, determine if the user is asking about sentiment analysis or wants you to analyze the sentiment of a sentence.
        If yes, respond in the format
        {{
            "intent": "yes",
            "sentence": "<sentence to analyze>"
        }}

        If no, respond in the format with your normal response.
        {{
            "intent": "no",
            "response": "<your response>"
        }}
    """
    # Get response from Gemini
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text