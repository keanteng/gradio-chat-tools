def analyze_sentiment(sentence):
    positive_keywords = ["good", "great", "happy", "love", "excellent", "fantastic"]
    negative_keywords = ["bad", "sad", "hate", "terrible", "awful", "poor"]

    sentence_lower = sentence.lower()

    if any(keyword in sentence_lower for keyword in positive_keywords):
        return "positive"
    elif any(keyword in sentence_lower for keyword in negative_keywords):
        return "negative"
    else:
        return "neutral"