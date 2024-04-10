import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

# Load the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def calculate_similarity(query_embedding, stored_embedding):
    stored_embedding_np = np.array(stored_embedding)
    query_embedding = query_embedding.reshape(1, -1)
    stored_embedding_np = stored_embedding_np.reshape(1, -1)
    return cosine_similarity(query_embedding, stored_embedding_np)[0][0]

def load_json_data():
    data = []
    with open('storage.json', 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    return data

def generate_embeddings(text):
    embeddings = model.encode(text)
    return embeddings


json_data = load_json_data()
st.write(json_data[0])
st.set_page_config(page_title="LEGAL EASE ", layout="wide")
st.title("Legal Ease: ")
st.write("Enter your query:")
input_text = st.text_area("Query:", height=100)

if st.button("Proceed") and input_text:
    query_embedding = generate_embeddings(input_text)
    max_similarity = 0
    most_similar_article = None

    for article in json_data:
        similarities = []
        for section_key in ['citation_title', 'date_bench', 'Introduction', 'prev_judgement', 'background', 'argument', 'Judgement', 'concluding']:
            section_data = article.get(section_key, {})
            section_embedding = section_data.get('embedding', [])
            if section_embedding:
                similarity = calculate_similarity(query_embedding, section_embedding)
                similarities.append(similarity)

        # Calculate the average similarity for this article
        if similarities:
            avg_similarity = np.mean(similarities)
            if avg_similarity > max_similarity:
                max_similarity = avg_similarity
                most_similar_article = article

    if most_similar_article:
        st.write(f"**Similarity:** {max_similarity * 100:.2f}%")
        for key, value in most_similar_article.items():
            if key != "embedding":  # Skip embedding for display
                st.write(f"**{key.replace('_', ' ').title()}:** {value['text']}")
    else:
        st.write("No similar articles found.")
else:
    st.error("Please enter a query.")




import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json


def load_json_data():
    with open('storage.json', 'r') as file:
        return json.load(file)

json_data = load_json_data()

# Initialize the model for generating embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(text):
    embeddings = model.encode(text)
    return embeddings

# Streamlit App UI
st.set_page_config(page_title="Legal Expert", layout="wide")
st.title("Legal News")

st.write("Enter your query and find the most similar article from the database:")

input_text = st.text_area("Query:", height=100)

if st.button("Check"):
    if input_text:
        query_embedding = generate_embeddings(input_text)
        
        max_similarity = 0
        most_similar_article = None
        
        for article in json_data.values():
            news_content = article['content']
            news_embedding = generate_embeddings(news_content)
            similarity = cosine_similarity([query_embedding], [news_embedding])[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_article = article
        
        if most_similar_article:
            st.write(f"**Most Similar Article:** {most_similar_article['headline']}")
            st.write(f"**Source:** {most_similar_article['source']}")
            st.write(f"**Similarity:** {max_similarity * 100:.2f}%")
            st.write(f"**Content:** {most_similar_article['content']}")
            
            
            
        else:
            st.write("No similar articles found.")
    else:
        st.error("Please enter a query.")
