import streamlit as st
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Funzione per il confronto tra dati reali e sintetici
def compare_vis(real_data, synthetic_data):

    st.title("Quick overview of data generated")
    # Analizza i dati per determinare il tipo di dati
    numeric_columns = real_data.select_dtypes(include='number').columns
    categorical_columns = real_data.select_dtypes(exclude='number').columns

    # Se i dati sono numerici
    if len(numeric_columns) > 0:
        
        # Calcola medie e deviazioni standard per entrambi i set di dati
        numeric_stats_real = real_data.describe().loc[['mean', 'std']]
        numeric_stats_synthetic = synthetic_data.describe().loc[['mean', 'std']]

        show_results(real_data, synthetic_data)

        st.divider()

        # Distribuzioni dei dati numerici
        num_plots = len(numeric_columns)
        cols = st.columns(3)  # Numero di colonne per la disposizione dei grafici

        st.subheader("Real (blue) and Syntethic Data (red) comparison")

        for i, column in enumerate(numeric_columns):
            # Istogramma per i dati reali
            fig_real = px.histogram(real_data, x=column, histnorm='probability density', opacity=0.5)
            # Istogramma per i dati sintetici
            fig_synthetic = px.histogram(synthetic_data, x=column, histnorm='probability density', opacity=0.5, color_discrete_sequence=['indianred'])
            # Aggiungi entrambi gli istogrammi al grafico
            fig = fig_real.add_traces(fig_synthetic.data)
            fig.update_layout(barmode='overlay', title=f"Distribution of {column}")
            cols[i % 3].plotly_chart(fig)  # Distribuisci i grafici su tre colonne

    # Se i dati sono categorici
    if len(categorical_columns) > 0:
        # Frequenze delle categorie
        num_plots = len(categorical_columns)
        cols = st.columns(3)  # Numero di colonne per la disposizione dei grafici

        for i, column in enumerate(categorical_columns):
            # Istogramma per i dati reali
            fig_real = px.bar(real_data[column].value_counts(), x=real_data[column].value_counts().index, y=real_data[column].value_counts(), opacity=0.5, title=f"Frequencies of {column}")
            # Istogramma per i dati sintetici
            fig_synthetic = px.bar(synthetic_data[column].value_counts(), x=synthetic_data[column].value_counts().index, y=synthetic_data[column].value_counts(), opacity=0.5, color_discrete_sequence=['indianred'])
            # Aggiungi entrambi gli istogrammi al grafico
            fig = fig_real.add_traces(fig_synthetic.data)
            cols[i % 3].plotly_chart(fig)  # Distribuisci i grafici su tre colonne
    return "ok"

# Funzione per calcolare il log-likelihood sintetico
def compute_synthetic_log_likelihood(real_data, synthetic_data):
    try:
        # Calcola le probabilità di ogni valore nei dati sintetici
        synthetic_probs = synthetic_data.value_counts(normalize=True)

        # Seleziona solo i valori presenti sia nei dati reali che sintetici
        common_values = real_data[real_data.isin(synthetic_data.unique())].unique()

        # Calcola il logaritmo delle probabilità dei valori comuni
        log_likelihoods = np.log(synthetic_probs[common_values].values)

        # Somma dei log-likelihoods
        synthetic_log_likelihood = log_likelihoods.sum()

        return synthetic_log_likelihood / len(real_data)
    except Exception as e:
        print(f"Error in compute_synthetic_log_likelihood: {e}")
        return None

# Funzione per calcolare le statistiche e eseguire il test t
def compare_statistics_and_t_test(real_data, synthetic_data):
    # Seleziona solo le colonne continue
    real_numeric = real_data.select_dtypes(include=['float64', 'int64'])
    synthetic_numeric = synthetic_data.select_dtypes(include=['float64', 'int64'])

    # Calcola le statistiche descrittive
    real_stats = real_numeric.describe().loc[['mean', 'std']]
    synthetic_stats = synthetic_numeric.describe().loc[['mean', 'std']]

    # Calcola il t-test
    t_test_results = ttest_ind(real_numeric, synthetic_numeric)

    # Crea il DataFrame per la tabella di comparazione
    comparison_table = pd.concat([real_stats, synthetic_stats], axis=1)
    
    # Assegna i nomi delle colonne in base alle colonne selezionate
    comparison_table.columns = [f'Real Data ({col})' for col in real_stats.columns] + [f'Synthetic Data ({col})' for col in synthetic_stats.columns]
    
    # Crea un DataFrame per i risultati del test t
    t_test_results_df = pd.DataFrame(t_test_results)
    # Assegna i nomi delle colonne in base alle colonne selezionate
    t_test_results_df.index = ['T-test Statistic', 'P-value']
    t_test_results.columns = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    return comparison_table, t_test_results_df

# Funzione per mostrare i risultati in Streamlit
def show_results(real_data, synthetic_data):

    # Confronta le statistiche e il test t
    comparison_table, t_test_results_df = compare_statistics_and_t_test(real_data, synthetic_data)

    st.write("Comparison Table:")
    st.table(comparison_table)

    st.write("T-Test Results:")
    st.table(t_test_results_df)
    return "ok"