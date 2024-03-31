#libraries
import streamlit as st
import base64
from PIL import Image
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "seaborn"

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
import os
from io import StringIO
import tempfile

from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata

from requests_folder.request import process_train_ctgan, process_inference_ctgan, get_models, get_transactions, inference_tvae, process_train_tvae
from visual.visualization import compare_vis, sistemazione_modelli

# interact with FastAPI endpoint

# Recupera la chiave API dalle variabili d'ambiente
try:
    endopoint = os.environ['URL']
except KeyError:
    endopoint = "http://127.0.0.1:8000"

get_models_method = endopoint+"/get_models"
get_transaction_method = endopoint+"/transactions"
training_ctgan_method = endopoint+"/training_model_ctgan"
training_tvae_method = endopoint+"/train_model_tvae_adults_dataset"
inference_ctgan_tvae_url = endopoint+"/inference_ctgan_tvae_metrics"
inference_tvae_url = endopoint+"/inference_tvae_gpu"

#open images
img_icon = Image.open('img/sdv2.png')
img_pipe = Image.open('img/mod_ctgan.png')

# Verifica se il file Excel esiste già
excel_file = 'latency_data.xlsx'

if not os.path.exists(excel_file):
    # Se il file non esiste, crea un nuovo DataFrame vuoto
    data = {'url': [], 'Latency': []}
    df_latency = pd.DataFrame(data)
else:
    # Se il file esiste, carica il DataFrame dal file Excel
    df_latency = pd.read_excel(excel_file)

#starting application
st.set_page_config(page_title="SYN-GEN", page_icon="img/tools.svg", layout="wide")

#sidebar
with st.sidebar:
    a,b,c=st.columns([0.1,1,0.1])
    b.image(img_icon, width=150)
    st.header("")
    selected = st.radio("Menu Choice", ['Predictor', 'Dashboard API analytics', 'Report Bug'])

    st.write("")

    st.markdown("""<hr style="height:5px;border:none;color:#027953;background-color:#2FA27D;" /> """, unsafe_allow_html=True)

    # Chiedi all'utente di inserire l'API key come password
    api_key = st.text_input("Add API Key to use tool", type="password")

    st.sidebar.header("Info & Support")
    st.info(
    """
    Questa è una webapp che consente di interagire con l'api sulla cloud run.
    
    API URL: https://syntethic-data-backend-gxk724njya-ew.a.run.app/

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

    main_choice = st.selectbox("Do you want to train your new models or use an existing one to generate data?", ["Training", "Evaluation"])

    ### Evaluation section
    if main_choice=="Evaluation":

        col1, col2 = st.columns([3, 1.5])

        with col1:
            butt_model_list = st.button("Show the available trained model 🔥")
            if butt_model_list:
                # Ottieni i modelli utilizzando la cache
                models = get_models(get_models_method)
                model_info = sistemazione_modelli(models)
            
                # Creazione del DataFrame
                df_models = pd.DataFrame(model_info, columns=["Model Type", "Extension file", "Model ID", "Epochs", "Training Data"])
                # Mostra il DataFrame come editor dati e assegna la funzione di refresh al suo cambio
                st.data_editor(df_models, width=1200)

        with col2:
            st.info("If you trained a model, after 2/3 minutes max, you should see in the table. Otherwise, click this button to refresh!")


        st.markdown("""<hr style="height:3px;border:none;color:#027953;background-color:#2FA27D;" /> """, unsafe_allow_html=True)

        choice = st.selectbox("Select your model",["CT-GAN and T-VAE", "T-VAE"])
        st.info("You can use the 'CT-GAN and T-VAE' section for inference of all models (ctgan or tvae). Instead, the other section works with the accelerator of CUDA and so works only in locally with gpu.")

        if choice == "T-VAE":
            
            st.write("In this section, you can load only PKL files and don't works in cloud without GPU")
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

        elif choice == "CT-GAN and T-VAE":
            
            st.write("In this section, you can load only PyTorch (.pt) files")
            unique_id = st.text_input("Insert ID of model to use")

            # Parametri da inserire nella richiesta
            num_rows = st.number_input("Number of data to generate", value=200, min_value=50, max_value=2000)

            # Start counter
            start_time = time.time()

            if st.button("Testing CT-GAN and T-VAE and create new data"):
                if unique_id and num_rows is not None:
                    # Visualizza l'API Key inserita dall'utente
                    st.write("API Key:", api_key)

                    response, status_code = process_inference_ctgan(inference_ctgan_tvae_url, api_key, unique_id, num_rows)

                    if status_code==200:
                        # Calcolo latenza
                        end_time = time.time()
                        latency = end_time - start_time
                        df_latency = df_latency._append({'url': '/inference_ctgan_metrics', 'Latency': latency}, ignore_index=True)
                        df_latency.to_excel(excel_file, index=False)

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

        # Start counter
        start_time = time.time()

        if choice == "CT-GAN":

            # Caricamento dei file di training
            file_training_data = st.file_uploader('Insert data for training', type=['csv', 'xlsx'])

            # Parametri da inserire nella richiesta
            epochs = st.number_input("Number of epochs", value=3, min_value=1, max_value=10)

            # Invio della richiesta al server
            if st.button("Train CTGAN"):
                if file_training_data and api_key is not None:
                    # Visualizza l'API Key inserita dall'utente
                    st.write("API Key:", api_key)

                    response = process_train_ctgan(file_training_data, training_ctgan_method, api_key, epochs)
                    
                    # Calcolo latenza
                    end_time = time.time()
                    latency = end_time - start_time
                    df_latency = df_latency._append({'url': '/training_model_ctgan', 'Latency': latency}, ignore_index=True)
                    df_latency.to_excel(excel_file, index=False)

                    # Verifica della risposta
                    if response.status_code == 200:
                        response_json = response.json()
                        status_code = response_json.get('status')
                        unique_id = response_json.get('uuid')
                        message = response_json.get('message')
                        st.success(f"Starting {message} with code: {unique_id}")
                    else:
                        st.error(f"Error: {response.status_code} - Please upload a file first and set API KEY")
                else:
                    st.warning("Please upload a file first and set API KEY")

        elif choice=="T-VAE":

            st.warning("Per il training di questo modello verrà utilizzato di default il dataset adults.csv (32561 records)")
            # Parametri da inserire nella richiesta
            epochs = st.number_input("Number of epochs", value=3, min_value=1, max_value=30)
            butt=st.button("Train TVAE")

            # Start counter
            start_time = time.time()

            # Invio della richiesta al server
            if butt:
                try:
                    if epochs and api_key is not None:
                        # Visualizza l'API Key inserita dall'utente
                        st.write("API Key:", api_key)
                        print(training_tvae_method, epochs)
                        response, status_code = process_train_tvae(training_tvae_method, api_key, epochs)
                        
                        # Calcolo latenza
                        end_time = time.time()
                        latency = end_time - start_time
                        df_latency = df_latency._append({'url': '/train_model_tvae_adults_dataset', 'Latency': latency}, ignore_index=True)
                        df_latency.to_excel(excel_file, index=False)

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

elif selected=="Dashboard API analytics":
    st.title("Dashboard API analytics")

    # Creazione delle liste vuote per ogni attributo
    urls = []
    unique_ids = []
    model_ids = []
    status_codes = []
    methods = []
    timestamps = []

    transactions_response, status_code = get_transactions(get_transaction_method)
    if status_code == 200:
        estrazione = transactions_response.json()
        # Creazione del DataFrame
        df = pd.DataFrame(estrazione)
        num_tot = len(df)

        # Filtraggio dei dati dove status code == 200
        successfully = df[df['status_code'] == 200]
        num_success = len(successfully)
        #print(successfully)

        # Filtraggio dei dati dove status code =! 200
        wrong = df[df['status_code'] != 200]
        num_wrong = len(wrong)
        #print(wrong)

        perc_succ = round((num_success / num_tot)*100, 3)
        perc_wrong = round((num_wrong / num_tot)*100, 3)

        # Calcola la media della colonna1
        media_latency = round(df_latency['Latency'].mean(), 2)

    st.write("In this section, it's possible analyze the API call and discover some problems about the backend of application.")
    a, b, c, d = st.columns(4)
    a.metric("Number of API call",num_tot, )
    b.metric(":green[Successful] API call (%)",perc_succ)
    c.metric(":red[Wrong] API call (%)",perc_wrong)
    d.metric("Latency of requests (sec)",media_latency)

    # Converti la colonna 'timestamp' in formato datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Estrai il giorno dalle date
    df['day'] = df['timestamp'].dt.date

    # Raggruppa i dati per giorno e conteggia il numero di chiamate con diversi status code
    daily_counts = df.groupby(['day', 'status_code']).size().unstack(fill_value=0)

    # Plotly bar chart
    fig = px.bar(daily_counts, x=daily_counts.index, y=[200, 500], 
                labels={'value': 'Numero di chiamate', 'day': 'Giorno', 'status_code': 'Status Code'},
                title='Numero di chiamate API per giorno', barmode='group')

    # Impostazioni grafico
    fig.update_layout(xaxis_tickangle=-45)

    # Visualizza il grafico su Streamlit
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.table(df)


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