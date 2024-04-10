import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_json_data():
    with open('storage.json', 'r') as file:
        return json.load(file)
json_data = load_json_data()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def generate_embeddings(text):
    embeddings = model.encode(text)
    return embeddings
st.set_page_config(page_title="Legal Expert", layout="wide")
st.title("Legal Cases")
st.write("Enter your query and find the most similar article from the database:")
input_text = st.text_area("Query:", height=100)
if st.button("Check"):
    if input_text:
        query_embedding = generate_embeddings(input_text)
        max_similarity = 0 
        most_similar_article = None
        for article in json_data: 
            citation_title_embedding =article['citation_title']['emdedding']
            date_bench_embedding =article['date_bench']['embedding']
            Introduction_embedding =article['Introduction']['embedding']
            prev_judgement_embedding =article['prev_judgement']['embedding']
            background_embedding =article['background']['embedding']
            Judgement_embedding =article['Judgement']['embedding']
            concluding_embedding =article['concluding']['embedding']

#Put  drop down to search from the domain  citation_title_embedding , ate_bench_embedding etc. and allow the user to search on its behalf.

            similarity1 = cosine_similarity([query_embedding], [citation_title_embedding])
            similarity2 = cosine_similarity([query_embedding], [date_bench_embedding])
            similarity3 = cosine_similarity([query_embedding], [Introduction_embedding])
            similarity4 = cosine_similarity([query_embedding], [prev_judgement_embedding])
            similarity5 = cosine_similarity([query_embedding], [background_embedding])
            similarity6 = cosine_similarity([query_embedding], [Judgement_embedding])
            similarity7 = cosine_similarity([query_embedding], [concluding_embedding])
            similarity=(similarity1+similarity2+similarity3+similarity4+similarity5+similarity6+similarity7)

                
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_article = article
        
        if most_similar_article:
            st.write(f"**Citation_title Case:** {most_similar_article['citation_title']['text']}")
            st.write(f"**Date Bench Case:** {most_similar_article['date_bench']['text']}")
            st.write(f"**Introduction Case:** {most_similar_article['Introduction']['text']}")
            st.write(f"**Previous Judgement Case:** {most_similar_article['prev_judgement']['text']}")
            st.write(f"**Background Case:** {most_similar_article['background']['text']}")
            st.write(f"**Judgement Case:** {most_similar_article['Judgement']['text']}")
            # st.write(f"**Similarity:** {max_similarity * 100:.2f}%")
        else:
            st.write("No similar articles found.")
    else:
        st.error("Please enter a query.")
