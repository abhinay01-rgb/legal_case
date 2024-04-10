import json
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

def load_json_data(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("JSON file not found.")
        return []

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embeddings(text):
    embeddings = model.encode(text)  # This returns a NumPy array
    return embeddings.tolist()  # Convert the NumPy array to a list

def evaluate_relevance(query, section_text):
    try:
        payload = {
            "inputs": {
                "prompt": f"Query: {query}\nDocument Section: {section_text}",
                "parameters": {
                    "temperature": 0.5
                }
            }
        }
        bedrock = boto3.client(service_name="bedrock-runtime", aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        body = json.dumps(payload)
        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        print(description(response_body))  # Assuming description() processes and returns something meaningful
        relevance_score = response_body.get('score', 0)  # Assuming 'score' is in response_body
        return relevance_score
    except (NoCredentialsError, ClientError) as e:
        print("AWS API error: Make sure your AWS credentials are set up correctly.")
        return 0

def main():
    print("Legal Document Relevance Tool")
    input_query = input("Enter your query here: ")
    input_query=generate_embeddings(input_query)
    documents = load_json_data('storage.json')
    highest_score = -1
    most_relevant_document = None
    if documents:
        for document in documents:
            total_score = 0
            count = 0
            for key, value in document.items():
                print(value)
                score = evaluate_relevance(input_query, value)
                total_score += score
                count += 1
            if count > 0:
                avg_score = total_score / count
                if avg_score > highest_score:
                    highest_score = avg_score
                    most_relevant_document = document

        if most_relevant_document:
            print("Most Relevant Document Found:")
            for key, value in most_relevant_document.items():
                print(f"**{key}:** {value}")
            print(f"**Relevance Score:** {highest_score}")
        else:
            print("No relevant documents found.")

if __name__ == "__main__":
    main()
