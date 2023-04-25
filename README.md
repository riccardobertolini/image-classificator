# Felidae Image Classification Project

This project aims to classify images of different species within the Felidae family. It consists of a machine learning model for training and a web server for handling user requests.

## Demo
It's possible to test the model uploading an image for the prediction using [this link](https://image-classification.herokuapp.com/form).
The result will be a json with the possible category prediction probability.
For security reason, the image once interpolated it's discarded. 

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
* training: A directory containing the trained models.
* my_model.h5: The auto generated model, updated using the training material provided.
* Procfile/runtime.txt: files used as config for Heroku deployment
