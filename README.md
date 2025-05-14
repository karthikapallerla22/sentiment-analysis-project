from textblob import TextBlob

texts = [
    "I love this movie!",
    "This was a terrible experience.",
    "It's okay, not great.",
    "Absolutely wonderful!",
    "I wouldn't recommend this.",
    "Mediocre acting, not impressed.",
    "That was the worst!",
    "A true masterpiece.",
    "It was average.",
    "Brilliant work!"
]

for text in texts:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"Text: {text}")
    print(f"Polarity: {polarity}, Sentiment: {sentiment}\n")
