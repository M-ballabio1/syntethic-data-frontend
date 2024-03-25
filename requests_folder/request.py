import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
import streamlit as st

def get_transactions(get_transaction_method):
    response = requests.get(get_transaction_method)
    if response.status_code == 200:
        return response, response.status_code
    else:
        return None, response.status_code

def process_train_ctgan(file, server_url, API_KEY, epochs):
    # Leggi il contenuto del file
    file_content = file.read()
    filename = file.name
    
    # Costruisci il payload della richiesta
    m = MultipartEncoder(
        fields={
            "epochs": str(epochs),
            "file_training_data": (filename, file_content, "text/csv"),
            "api_key": API_KEY
        }
    )
    
    # Invia la richiesta al server
    response = requests.post(
        server_url, 
        data=m, 
        headers={"Content-Type": m.content_type}, 
        timeout=8000
    )
    return response

def process_inference_ctgan(server_url, API_KEY, model_id, sample_num):
    
    sample_num = int(sample_num)

    #attesa
    with st.spinner("Calling API..."):
        # Fai la chiamata POST
        response = requests.post(
            server_url,
            params={"model_id": model_id, "sample_num": sample_num, "api_key": API_KEY}
        )
        if response.status_code == 200:
            return response, response.status_code
        else:
            return None, response.status_code
    

def process_train_tvae(server_url, API_KEY, epochs):
    # Costruisci il payload della richiesta

    # Fai la chiamata POST
    response = requests.post(
        server_url,
        params={"epochs": epochs, "api_key": API_KEY}
    )
    if response.status_code == 200:
        return response, response.status_code
    else:
        return []


def inference_tvae(API_KEY, server_url, unique_id, num_rows):
    # Costruisci il payload della richiesta

    num_rows = int(num_rows)

    #attesa
    with st.spinner("Calling API..."):
        # Fai la chiamata POST
        response = requests.post(
            server_url,
            params={"unique_id": unique_id, "num_rows": num_rows, "api_key": API_KEY}
        )
        if response.status_code == 200:
            return response, response.status_code
        else:
            return None, response.status_code
    
@st.cache_data()
def get_models(get_models_method):
    response = requests.get(get_models_method)
    if response.status_code == 200:
        return response.json()["models"]
    else:
        return []