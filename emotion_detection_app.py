import streamlit as st
from transformers import pipeline

# Load emotion detection pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

emotion_pipeline = load_pipeline()

# Function to detect emotion in text
def detect_emotion(text):
    results = emotion_pipeline(text)
    return results[0]

# Main application function
def main():
    st.title("Emotion Detection in Text")
    st.write("Enter your text below to detect the emotions.")

    user_input = st.text_area("Your Text:", "")
    
    if st.button("Detect Emotion"):
        if user_input:
            emotion_results = detect_emotion(user_input)
            st.write("### Emotion Scores")
            for result in emotion_results:
                st.write(f"**{result['label']}**: {result['score']:.4f}")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
