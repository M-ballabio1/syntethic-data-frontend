#libraries
import streamlit as st
import base64
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
import os
from io import StringIO
import tempfile

from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata

from requests_folder.request import process_train_ctgan, get_models, inference_tvae, process_train_tvae
from visual.visualization import compare_vis

# interact with FastAPI endpoint

# Recupera la chiave API dalle variabili d'ambiente
try:
    endopoint = os.environ['URL']
except KeyError:
    endopoint = "http://127.0.0.1:8000"

get_models_method = endopoint+"/get_models"
training_ctgan_method = endopoint+"/training_model_ctgan"
training_tvae_method = endopoint+"/train_model_tvae_adults_dataset"
inference_ctgan_url = endopoint+"/inference_ctgan_metrics"
inference_tvae_url = endopoint+"/inference_tvae"
#backend = "https://api-ultrasound-classificator-cloud-run-pa6vji5wfa-ew.a.run.app/classification"

#open images
img_icon = Image.open('img/sdv2.png')
img_pipe = Image.open('img/mod_ctgan.png')

#starting application
st.set_page_config(page_title="SYN-GEN", page_icon="img/tools.svg", layout="wide")

#sidebar
with st.sidebar:
    a,b,c=st.columns([0.1,1,0.1])
    b.image(img_icon, width=150)
    st.header("")
    selected = st.radio("Menu Choice", ['Predictor', 'Report Bug'])

    st.write("")

    st.markdown("""<hr style="height:5px;border:none;color:#027953;background-color:#2FA27D;" /> """, unsafe_allow_html=True)

    # Chiedi all'utente di inserire l'API key come password
    api_key = st.text_input("Add API Key to use tool", type="password")

    st.sidebar.header("Info & Support")
    st.info(
    """
    Questa è una webapp che consente di interagire con l'api sulla cloud run.
    
    API URL: https://api-ultrasound-classificator-cloud-run-pa6vji5wfa-ew.a.run.app/

    Per eventuali problemi nell'utilizzo app rivolgersi a: matteoballabio99@gmail.com
    """)

    st.sidebar.header("Test the tool here! ")
    with open("data_example/adult_example_test.csv", "rb") as file:
        csv_data = file.read()
        st.download_button(
            label="Download csv ⬇️",
            data=csv_data,
            file_name="adult_example_test.csv",
            mime="text/csv"
        )

hide_img_fs = '''
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        '''
st.markdown(hide_img_fs, unsafe_allow_html=True)

### documentation
if selected=="Predictor":
    st.title("SYN-GEN: Integrated platform to generate new synthetic data 🚀")

    with st.expander("**Explanation of the HLD of the platform SYN-GEN**"):
        a,b,c=st.columns([0.05,1.5,0.05])
        b.info("""
        **Project Implementation of a Synthetic Data Generation Platform!**

        _**Goal:**_ The project aims to implement a web application named 'Synthetic Data Generation Platform', 
               engineering two generative AI models (transformers) for synthetic data generation and creating a user-friendly local 
               or cloud-based API infrastructure. The entire engineering process includes code validation of the provided models, 
               creation of APIs for basic functionalities, and implementation of a simple interface to utilize the models. The implementation 
               of the infrastructure container will be considered an additional advantage.
        """, icon="ℹ️")

        with b.container():
            b.image(img_pipe)
            b.caption(""":black[Figure illustrating the architecture of the CT-GAN generator and discriminator. The generator transforms random noise into synthetic data samples, while the discriminator evaluates the authenticity of the generated samples compared to real data.]""")
        st.header("")
    

    st.subheader("Syntethic data generator model ")
    st.write("In this section is possible to train the models with CT-GAN or T-VAE. Then, it's possible to test the model and obtain the metrics between real and sytethic data.")

    main_choice = st.selectbox("Do you want to train your new models or use an existing one to generate data?", ["Training", "Evaluation"])

    ### Evaluation section
    if main_choice=="Evaluation":

        # Ottieni i modelli utilizzando la cache
        models = get_models(get_models_method)

        # Lista per memorizzare le informazioni sui modelli
        model_info = []
        for model in models:
            split = model.split("_")
            
            type_model = split[0]
            id_model = split[2]
            epochs = split[3]
            training_data = split[4]
            tr = int(training_data.split(".")[0])

            # Aggiungi una tupla contenente le informazioni del modello alla lista
            model_info.append((type_model, id_model, epochs, tr))

        # Creazione del DataFrame
        df_models = pd.DataFrame(model_info, columns=["Model Type", "Model ID", "Epochs", "Training Data"])
        st.table(df_models)

        choice = st.selectbox("Select your model",["CT-GAN", "T-VAE"])

        if choice == "T-VAE":

            unique_id = st.text_input("Insert ID of model to use")

            # Parametri da inserire nella richiesta
            num_rows = st.number_input("Number of data to generate", value=200, min_value=50, max_value=2000)

            if st.button("Testing T-VAE and create new data"):
                if unique_id and num_rows is not None:
                    # Visualizza l'API Key inserita dall'utente
                    st.write("API Key:", api_key)

                    response, status_code = inference_tvae(api_key, inference_tvae_url, unique_id, num_rows)

                    if status_code==200:
                        st.success("Ecco i risultati!")

                        # Leggi i dati sintetici dalla risposta
                        synthetic_data = pd.read_csv(StringIO(response.content.decode('utf-8')))
                        
                        # Salva i dati sintetici in un file CSV temporaneo
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                            synthetic_data.to_csv(tmp_file, index=False)
                            tmp_file_path = tmp_file.name
                        
                        # Crea un pulsante di download per il file
                        st.download_button(
                            label="Download Synthetic Data CSV",
                            data=open(tmp_file_path, 'rb'),
                            file_name="synthetic_data.csv",
                            mime="text/csv"
                        )
                    
                        # genearate dashboard
                        metadata = SingleTableMetadata()

                        # Scarica il dataset di esempio
                        real_data, metadata = download_demo(modality='single_table', dataset_name='adult')

                        #plotly
                        compare_vis(real_data, synthetic_data)

                    else:
                        st.error("Ops, Qualcosa è andato storto")

    ### Training section
    elif main_choice=="Training":

        choice = st.selectbox("Select your model",["CT-GAN", "T-VAE"])

        if choice == "CT-GAN":

            # Caricamento dei file di training
            file_training_data = st.file_uploader('Insert data for training', type=['csv'])

            # Parametri da inserire nella richiesta
            epochs = st.number_input("Number of epochs", value=3, min_value=1, max_value=10)

            # Invio della richiesta al server
            if st.button("Train CTGAN"):
                if file_training_data and api_key is not None:
                    # Visualizza l'API Key inserita dall'utente
                    st.write("API Key:", api_key)

                    response = process_train_ctgan(file_training_data, training_ctgan_method, api_key, epochs)
                    
                    # Verifica della risposta
                    if response.status_code == 200:
                        response_json = response.json()
                        status_code = response_json.get('status')
                        unique_id = response_json.get('uuid')
                        message = response_json.get('message')
                        st.success(f"Starting {message} with code: {unique_id}")
                    else:
                        st.error(f"Error: {response.status_code}")
                else:
                    st.warning("Please upload a file first and set API KEY")

        elif choice=="T-VAE":

            st.warning("Per il training di questo modello verrà utilizzato di default il dataset adults.csv (32561 records)")
            # Parametri da inserire nella richiesta
            epochs = st.number_input("Number of epochs", value=3, min_value=1, max_value=30)
            butt=st.button("Train TVAE")

            # Invio della richiesta al server
            if butt:
                try:
                    if epochs and api_key is not None:
                        # Visualizza l'API Key inserita dall'utente
                        st.write("API Key:", api_key)
                        print(training_tvae_method, epochs)
                        response, status_code = process_train_tvae(training_tvae_method, api_key, epochs)
                        
                        # Verifica della risposta
                        if response.status_code == 200:
                            response_json = response.json()
                            status_code = response_json.get('status')
                            unique_id = response_json.get('uuid')
                            message = response_json.get('message')
                            st.success(f"Starting {message} with code: {unique_id}. Otterrai il modello utilizzabile nella sezione evaluation.")
                        else:
                            st.error(f"Error: {response.status_code}")
                    else:
                        st.warning("Please upload a file first and set API KEY")
                except Exception as e:
                    st.error(f"An error occurred: Probabilmente non hai settato api key")


elif selected=="Report Bug":
    st.title("Bug reporting App")
    form = st.form(key="annotation", clear_on_submit=True)
    with form:
        cols = st.columns((1, 1))
        author = cols[0].text_input("Report author:")
        bug_type = cols[1].selectbox(
            "Bug type:", ["Front-end", "Back-end", "Data related", "404"], index=2
        )
        comment = st.text_area("Comment:")
        cols = st.columns(2)
        date = cols[0].date_input("Bug date occurrence:")
        bug_severity = cols[1].slider("Bug severity:", 1, 5, 2)
        submitted = st.form_submit_button(label="Submit")

        if submitted:
            st.success("Bug sent")