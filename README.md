# [Fake News Detection](#fake-news-detection)

[![GitHub license](https://img.shields.io/github/license/deepikagandla7456/Fake-News-Detection)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/deepikagandla7456/Fake-News-Detection)]()
[![GitHub contributors](https://img.shields.io/github/contributors/deepikagandla7456/Fake-News-Detection)]()
[![GitHub last-commit](https://img.shields.io/github/last-commit/deepikagandla7456/Fake-News-Detection)]()

This project provides a command-line tool that helps to detect whether a news article is real or fake. It leverages a machine learning model trained on a large dataset to analyze text and predict its authenticity with a high degree of accuracy.

## Table of Contents
- [Fake News Detection](#fake-news-detection)
  - [About](#about)
  - [Features](#features)
  - [Future Work](#future-work)
  - [Requirements](#requirements)
  - [Setup and Installation](#setup-and-installation)
  - [Usage](#usage)
  - [How It Works](#how-it-works)
  - [Screenshots](#screenshots)
  - [License](#license)

---

## About

The primary goal of this project is to prevent the spread of misleading news by classifying articles as either ***Fake*** or ***Real***. It addresses this by using Natural Language Processing (NLP) and a classic machine learning model to differentiate between real and false content.

The model was trained on a substantial dataset consisting of **44,898 news articles**. The content of this dataset is primarily from **2017**, providing a robust, albeit time-specific, foundation for the model's classifications. The project successfully demonstrates the viability of this approach, achieving an accuracy of over **98.5%** on its test data.

---

## Features

- **High-Accuracy Predictions**: Classifies articles as `REAL` or `FAKE` with a proven accuracy of over 98.5%.
- **Prediction Confidence**: Provides a confidence score (e.g., `99.18%`) with each prediction, indicating the model's certainty.
- **Modular Structure**: The project is cleanly divided into a training script (`train_model.py`) and a prediction application (`app.py`).
- **Efficient Workflow**: The training process only needs to be run once. It saves a reusable model and vectorizer, allowing the prediction app to load and run instantly.

---

## Future Work

Potential improvements and future directions for the project include:

- **Real-Time News Analysis**: Integrate with live news APIs (e.g., NewsAPI, Google's Fact check API key) to fetch and classify current events, expanding its relevance beyond the static 2017 dataset.
- **Web Interface**: Develop a user-friendly web application using a framework like Flask or Streamlit to make the tool accessible to a broader audience.
- **Advanced Models**: Explore more complex deep learning models, such as LSTMs or pre-trained transformers (e.g., BERT), to potentially improve nuance and accuracy.

---

## Requirements

The following libraries are required to run this project and are listed in `requirements.txt`:

- `python 3.8 or higher version`
- `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `joblib`

---

## Setup and Installation

Follow these steps to get the project running locally.

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/deepikagandla7456/Fake-News-Detection.git
    cd Fake-News-Detection
    ```

2.  **Install the required packages:**
    It is recommended to use a virtual environment.
    ```shell
    pip install -r requirements.txt
    ```

3.  **Train the model:**
    This is a crucial step that must be run first. This script will process the data and create the `saved_models` directory.
    ```shell
    python train_model.py
    ```
---

## Usage

After the model has been trained and saved, the application can be used for predictions.

1.  **Run the application script:**
    ```shell
    python app.py
    ```

2.  **Enter News Text:**
    The program will prompt you to paste the text of a news article.

3.  **Receive the Prediction:**
    The model will return its classification (`REAL` or `FAKE`) and its confidence level. Type `quit` or `exit` to close the application.

---

## How It Works

1.  **Data Processing**: The `train_model.py` script loads the `True.csv` and `False.csv` datasets.
2.  **Feature Engineering**: The text from the articles is converted into numerical vectors using a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer. This technique helps the model understand which words are most important in distinguishing between the two classes.
3.  **Model Training**: A **Logistic Regression** classifier is trained on the vectorized data.
4.  **Application**: The `app.py` script loads the pre-trained model and vectorizer to perform live predictions on new, user-provided text.

---

## Screenshots

<<<<<<< HEAD
**Model Training and Evaluation Output**
![Image](https://github.com/user-attachments/assets/ba0074db-d20d-486b-b0f7-268ae22e9089)

**Live Article Prediction Output**
![Image](https://github.com/user-attachments/assets/0a1a611e-26ee-4f4f-9e1a-fb67a770e07a)
=======
**Model Evaluation Output**
<img width="1753" height="541" alt="Image" src="https://github.com/user-attachments/assets/d3881b74-db65-42a8-b20f-0408836bbf0b" />
<img width="1756" height="466" alt="Image" src="https://github.com/user-attachments/assets/64f10264-8935-444b-9f14-4d74962781fb" />
**Article Prediction Output**
<img width="1756" height="479" alt="Image" src="https://github.com/user-attachments/assets/e652c02f-1d60-4c3b-b1f3-84272a40de52" />
>>>>>>> b9af10e42afd10bda0d5b21b125f6e050c9811b8

---

## License
This project is licensed under the [MIT](LICENSE) - see the [LICENSE](LICENSE) file for details.