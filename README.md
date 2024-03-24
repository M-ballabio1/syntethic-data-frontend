# SYN-GEN Web Application Documentation

## Overview

SYN-GEN is a web application designed to interact with an API backend for synthetic data generation. It provides an intuitive user interface for training generative AI models, generating synthetic data, and reporting bugs. The application integrates with backend services to perform these tasks efficiently.

## Getting Started

To use SYN-GEN, follow these steps:

1. Clone the repository to your local machine.
2. Install the required Python dependencies using `pip install -r requirements.txt`.
3. Set up the backend API endpoint URL in the environment variable `URL`.
4. Run the frontend application using `streamlit run app.py`.

## Features

### Synthetic Data Generation

- **Training Models**: Train generative AI models (CT-GAN and T-VAE) using your dataset.
- **Inference**: Generate synthetic data using trained models.
- **Model Evaluation**: Evaluate the performance of trained models using real and synthetic data.

### Bug Reporting

- **Bug Types**: Report bugs related to front-end, back-end, data, or 404 errors.
- **Severity Level**: Assign a severity level to reported bugs for prioritization.

## Usage

### Synthetic Data Generation

1. **Choose Mode**: Select whether you want to train a new model or use an existing one.
2. **Model Selection**: Choose the model type (CT-GAN or T-VAE) and provide necessary inputs.
3. **Training/Inference**: Train the model or generate synthetic data using the selected model.
4. **Visualization**: Visualize the synthetic data and compare it with real data using interactive charts.

### Bug Reporting

1. **Bug Reporting Form**: Fill out the bug reporting form with details such as author, bug type, severity, and comments.
2. **Submission**: Submit the bug report using the "Submit" button.

## Support

For any issues or queries related to SYN-GEN, please contact the project maintainer at [matteoballabio99@gmail.com](mailto:matteoballabio99@gmail.com).