import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def preprocess_text(text):
    """Remove numbers and extra spaces"""
    text = ''.join([i for i in text if not i.isdigit()])
    text = ' '.join(text.split())
    return text.lower()

def find_top_resumes(job_description, resumes, top_n=10):
    """Compute similarity and return top N resumes"""
    vectorizer = TfidfVectorizer(stop_words='english')
    docs = [job_description] + resumes
    tfidf_matrix = vectorizer.fit_transform(docs)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(resumes[i], similarities[i]) for i in top_indices]

# Streamlit UI
st.title("Resume Matching System")

# Upload resumes
uploaded_files = st.file_uploader("Upload Resume Files (Text Format)", accept_multiple_files=True, type=["txt"])
resumes = []

if uploaded_files:
    for file in uploaded_files:
        content = file.read().decode("utf-8")
        resumes.append(preprocess_text(content))

# Input job description
job_description = st.text_area("Enter Job Description")

if st.button("Find Top 10 Resumes"):
    if job_description and resumes:
        results = find_top_resumes(preprocess_text(job_description), resumes)
        st.subheader("Top Matching Resumes:")
        for idx, (resume, score) in enumerate(results):
            st.write(f"**Rank {idx + 1}:** Similarity Score: {score:.4f}")
    else:
        st.warning("Please upload resumes and enter a job description.")
