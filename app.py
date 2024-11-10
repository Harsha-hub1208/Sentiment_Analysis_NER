import openai
import spacy
import streamlit as st

# Load Spacy NER model
nlp = spacy.load("en_core_web_sm")

# Set OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Function to perform Named Entity Recognition (NER)
def extract_entities(review):
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function to analyze sentiment with word-level contributions using GPT
def analyze_sentiment_with_words(review, category):
    prompt = f"Analyze the sentiment of the following {category} review and provide sentiment contributions for each word (percentage):\n\nReview: {review}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    sentiment_analysis = response['choices'][0]['message']['content']
    return sentiment_analysis.strip()

# Streamlit main function
st.title("Sentiment Analysis and Named Entity Recognition")

# Get input from user via Streamlit
category = st.text_input("Enter the category of the review (e.g., Food, Product, Place, Other):").capitalize()
review = st.text_area("Enter your review:")

if st.button("Analyze"):
    if review:
        # Analyze the sentiment of the review with word-level contributions
        st.write("Performing Sentiment Analysis...")
        sentiment_with_contributions = analyze_sentiment_with_words(review, category)
        st.text_area("Sentiment Analysis with Word-Level Contributions:", sentiment_with_contributions, height=300)

        # Perform Named Entity Recognition (NER)
        st.write("Performing Named Entity Recognition (NER)...")
        entities = extract_entities(review)
        if entities:
            st.write("Named Entities in the Review:")
            for entity, label in entities:
                st.write(f"Entity: {entity}, Label: {label}")
        else:
            st.write("No named entities found.")
    else:
        st.warning("Please enter a valid review.")
