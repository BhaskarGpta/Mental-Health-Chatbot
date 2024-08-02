# Mental Health Chatbot

This project focuses on emotion classification using various machine learning techniques to support a mental health chatbot.

## Table of Contents

- [Project Overview](#project-overview)
- [Files and Directories](#files-and-directories)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This repository contains code and data for building and evaluating emotion classification models. The models are trained using different techniques and datasets to classify emotions from text data, which can be used to enhance the functionality of a mental health chatbot.

## Files and Directories

- `.gitignore`: Specifies files and directories to be ignored by Git.
- `emotion.ipynb`: Jupyter Notebook for emotion classification.
- `emotion_classifer.py`: Python script for emotion classification.
- `lstm_vader.ipynb`: Jupyter Notebook for LSTM and VADER sentiment analysis.
- `mindmate.ipynb`: Jupyter Notebook for another emotion classification model.
- `my_model.h5`: Pre-trained model file.
- `my_model_five.h5`: Another pre-trained model file.
- `text.csv`: Dataset containing text data for training and evaluation.
- `tokenizer.pkl`: Tokenizer object for preprocessing text data.
- `tokenizer_five.pkl`: Another tokenizer object for preprocessing text data.
- `.venv`: Virtual environment directory (ignored by Git).
- `.ipynb_checkpoints`: Jupyter Notebook checkpoints (ignored by Git).
- `data.ipynb`: Data exploration and preprocessing notebook.
- `emotion.csv`: Dataset file (ignored by Git).
- `.env`: Environment variables file (ignored by Git).
- `docs`: Documentation directory (ignored by Git).
- `main.py`: Main script for running the application.
- `models`: Directory containing model files (ignored by Git).
- `__pycache__`: Python cache directory (ignored by Git).

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt

Usage
To run the emotion classification model, use the following command:
python emotion_classifer.py

For detailed analysis and model training, refer to the Jupyter Notebooks provided in the repository.

License
This project is licensed under the MIT License. See the LICENSE file for more details.
