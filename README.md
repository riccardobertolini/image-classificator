# Felidae Image Classification Project

This project aims to classify images of different species within the Felidae family. It consists of a machine learning model for training and a web server for handling user requests.

## Getting Started

Follow these steps to set up and run the project:

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Train the machine learning model:
```commandline
python main.py
```



This script trains the model on the dataset and saves the trained model in the models directory. You can modify the script to adjust the model architecture, hyperparameters, or training data.

Start the web server:

```commandline
python app.py
```

This command starts the server, which listens for user requests to classify images. The server uses the trained model to make predictions and returns the results to the user.
Project Structure

* main.py: The main script for training the machine learning model.
* app.py: The script to start the web server for handling user requests.
* models: A directory containing the trained models.
* data: A directory containing the dataset, organized into subdirectories for each species.
