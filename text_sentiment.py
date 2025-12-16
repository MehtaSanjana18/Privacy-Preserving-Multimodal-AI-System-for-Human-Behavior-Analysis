from textblob import TextBlob

def analyze_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        return "Positive", polarity
    elif polarity < -0.2:
        return "Negative", polarity
    else:
        return "Neutral", polarity
