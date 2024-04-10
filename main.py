import json
import os
import boto3
import fitz 


def citation_title_fn(content):
    prompt_data=""" As an AI distinguished for its proficiency in legal analysis, you are tasked to thoroughly scrutinize the text extracted from a legal document.
      Your mission is to identify the Case Citation and Title, crucial elements that encapsulate the identity and reference of the case at hand. 
      Structure your insights into numbered points for a streamlined presentation. It's imperative that your summary is both accurate and articulated in layman's terms, making it accessible and understandable to individuals outside the legal profession. 
    Ensure that your report is concise, eliminating any repetition, and focuses solely on pinpointing the vital details of the case's citation and title.
    """
    # Concatenate the transcription with the prompt_data
    prompt = "[INST]" + prompt_data + content + "[/INST]"

    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body['outputs'][0]['text']

def date_bench_fn(content):
    prompt_data=""" As an AI with exceptional expertise in legal matters, you're assigned the task of meticulously examining the text from a provided legal document.
      Your objective is to uncover the Date of the proceeding and the Composition of the Bench involved in the case. 
      Organize your findings in a numbered format, ensuring each point is clear and concise. Aim for precision in your summary, presenting the date and bench composition in plain language that is easily graspable by any reader, irrespective of their familiarity with legal terminology. 
    Make certain that your summary avoids redundancy and directly addresses the essential details regarding the case's date and bench composition.
    """
    # Concatenate the transcription with the prompt_data
    prompt = "[INST]" + prompt_data + content + "[/INST]"

    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body['outputs'][0]['text']


def Introduction_fn(content):
    prompt_data=""" As an AI deeply versed in the legal domain, you are tasked with rigorously analyzing the provided text from a legal document. 
    Your focus is to delineate the Introduction to the Case, capturing its essence and preliminary context. 
    Present your findings in a structured manner, utilizing numbered points for enhanced readability. 
    Your analysis should be both concise and precise, ensuring that the introduction is relayed in clear, straightforward language accessible to all readers, regardless of their legal knowledge. 
    It is crucial that your summary eliminates any repetition and succinctly conveys the foundational aspects of the case's introduction.
    """
    # Concatenate the transcription with the prompt_data
    prompt = "[INST]" + prompt_data + content + "[/INST]"

    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body['outputs'][0]['text']

def prev_judement_fn(content):
    prompt_data=""" As an AI proficient in legal analysis, you are tasked with conducting a detailed review of the enclosed text from a legal document. 
    Your primary goal is to unearth the Legal Provisions and Previous Judgments referenced or implied within the case. 
    Structure your findings into numbered points for clarity and coherence. Ensure that your summary is both precise and articulated in a manner that is comprehensible to a broad audience, including those without a legal background. 
    Focus on delivering a succinct overview that omits redundancies and highlights the key legal provisions and notable precedents.
    """
    # Concatenate the transcription with the prompt_data
    prompt = "[INST]" + prompt_data + content + "[/INST]"

    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body['outputs'][0]['text']

def background_fn(content):
    prompt_data="""As an AI with advanced expertise in legal analysis, your mission is to meticulously examine the provided text from a legal document. 
    Focus specifically on identifying the Factual Background of the case. Present your findings in a clear, numbered format to ensure ease of understanding. 
    Strive for conciseness and accuracy in your summary, and express the factual background in plain language, making it accessible to individuals without a legal background. 
    Ensure your presentation is streamlined, avoiding redundancies, and captures only the essential elements of the case's factual background.
    """
    # Concatenate the transcription with the prompt_data
    prompt = "[INST]" + prompt_data + content + "[/INST]"

    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body['outputs'][0]['text']

def argument_fn(content):
    prompt_data=""" As an AI endowed with deep expertise in legal matters, your assignment involves a comprehensive review of the enclosed text from a legal document.
      Your objective is to discern the core Arguments and Analysis presented within the case. Summarize your insights in a structured format, 
      using numbered points for straightforward comprehension. Aim for succinctness and precision in your exposition, ensuring the arguments and analyses are communicated in straightforward language, accessible to both legal professionals and laypersons alike.
      It is essential that your summary avoids redundancy and encapsulates the fundamental aspects of the legal arguments and analytical perspectives.
    """
    # Concatenate the transcription with the prompt_data
    prompt = "[INST]" + prompt_data + content + "[/INST]"

    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body['outputs'][0]['text']

def Judgement_fn(content):
    prompt_data="""As an AI with specialized expertise in legal analysis, your task involves a thorough examination of the text provided from a legal document. 
    Your focus should be on uncovering the Judgment and Decision made in the case. Present your findings succinctly, using numbered points for clarity. 
    Your analysis should be precise and articulate the judgment and decision in a manner that is easily understandable to a layperson, without any legal jargon. 
    Ensure that your summary is concise, avoiding any repetition, and captures the critical elements of the case's final judgment and decision.
    """
    # Concatenate the transcription with the prompt_data
    prompt = "[INST]" + prompt_data + content + "[/INST]"

    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())

    return response_body['outputs'][0]['text']

def concluding_fn(content):
    prompt_data=""" As a highly skilled AI with expertise in legal analysis, your task is to meticulously review the provided text from a legal document. 
    Focus on identifying the Concluding Remarks section, which encapsulates the essence and final decisions of the case. 
    Present your findings concisely, listing them in numbered points for ease of understanding. Aim for precision in your analysis and communicate the outcomes in clear, accessible language that any individual, regardless of their legal background, can comprehend.
    Ensure your summary is direct and avoids any repetition, capturing only the most critical elements of the Concluding Remarks..
    """
    # Concatenate the transcription with the prompt_data
    prompt = "[INST]" + prompt_data + content + "[/INST]"

    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "prompt": prompt,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body['outputs'][0]['text']

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def generate_embeddings(text):
    embeddings = model.encode(text)  # This returns a NumPy array
    return embeddings.tolist()  # Convert the NumPy array to a list

import json

def read_pdf_content(pdf_path):
    document = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    document.close()
    #return the data in chunks 32768 tokens
    return text

def read_all_pdfs_in_folder(folder_path):
    files = os.listdir(folder_path)
    pdf_files = [file for file in files if file.lower().endswith('.pdf')]
    json_file_path = os.path.join(folder_path, "output_file_with_embeddings.ndjson")  # Using .ndjson for clarity
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Reading {pdf_file}...")
        
        text = read_pdf_content(pdf_path)
        print(pdf_path)

        citation_title = citation_title_fn(text)
        date_bench = date_bench_fn(text)
        Introduction = Introduction_fn(text)
        prev_judgement = prev_judement_fn(text)
        background = background_fn(text)
        argument = argument_fn(text)
        Judgement = Judgement_fn(text)
        concluding = concluding_fn(text)


        
        citation_title_em = generate_embeddings(citation_title)
        date_bench_em = generate_embeddings(date_bench)
        Introduction_em = generate_embeddings(Introduction)
        prev_judgement_em = generate_embeddings(prev_judgement)
        background_em = generate_embeddings(background)
        argument_em = generate_embeddings(argument)
        Judgement_em = generate_embeddings(Judgement)
        concluding_em = generate_embeddings(concluding)

        pdf_data= {
            "citation_title": {
                "text": citation_title,
                "embedding": citation_title_em
            },
            "date_bench": {
                "text": date_bench,
                "embedding": date_bench_em
            },
            "Introduction": {
                "text": Introduction,
                "embedding": Introduction_em
            },
            "prev_judgement": {
                "text": prev_judgement,
                "embedding": prev_judgement_em
            },
            "background": {
                "text": background,
                "embedding": background_em
            },
            "argument": {
                "text": argument,
                "embedding": argument_em
            },
            "Judgement": {
                "text": Judgement,
                "embedding": Judgement_em
            },
            "concluding": {
                "text": concluding,
                "embedding": concluding_em
            }
        }

        # Convert pdf_data to match the desired output format
        output_pdf_data = {
        "citation_title":{ "text": pdf_data["citation_title"]["text"],"emdedding":pdf_data["citation_title"]["embedding"]},
        "date_bench": {"text":pdf_data["date_bench"]["text"],"embedding":pdf_data["date_bench"]["embedding"]},
        "Introduction":{"text":pdf_data["Introduction"]["text"],"embedding":pdf_data["Introduction"]["embedding"]},
        "prev_judgement":{"text": pdf_data["prev_judgement"]["text"],"embedding":pdf_data["prev_judgement"]["embedding"]},
        "background":{"text": pdf_data["background"]["text"],"embedding":pdf_data["background"]["embedding"]},
        "argument": {"text":pdf_data["argument"]["text"],"embedding":pdf_data["argument"]["embedding"]},
        "Judgement":{"text":pdf_data["Judgement"]["text"],"embedding":pdf_data["Judgement"]["embedding"]},
        "concluding":{"text": pdf_data["concluding"]["text"],"embedding":pdf_data["Judgement"]["embedding"]}
        }

        # Now output_pdf_data is ready to be saved in the desired format
        json_file_path = "storage.json"  # Set your file path here
        save_pdf_data(output_pdf_data, json_file_path)

import json
import os

def save_pdf_data(pdf_data, json_file_path):
    """Appends PDF data to a JSON file as part of a list."""
    # Check if the file exists and is not empty
    if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
        # File exists and has data
        with open(json_file_path, 'r+', encoding='utf-8') as f:
            # Move to the beginning of the file
            f.seek(0)
            # Load the existing data
            try:
                data = json.load(f)
                # Ensure the data is a list
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                # In case the file is not empty but doesn't contain valid JSON
                data = []
            
            # Append the new pdf_data
            data.append(pdf_data)
            
            # Move back to the start of the file to overwrite
            f.seek(0)
            # Convert the list back to JSON and write it to the file
            json.dump(data, f, ensure_ascii=False, indent=4)
            # Truncate the file in case the new data is shorter than the old data
            f.truncate()
    else:
        # File doesn't exist or is empty, start a new list
        with open(json_file_path, 'w', encoding='utf-8') as f:
            # Write the pdf_data as the first element of a list
            json.dump([pdf_data], f, ensure_ascii=False, indent=4)



folder_path = 'cases'
read_all_pdfs_in_folder(folder_path)
 