# [Fake News Detection](#fake-news-detection)

[![GitHub license](https://img.shields.io/github/license/deepikagandla7456/Fake-News-Detection)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/deepikagandla7456/Fake-News-Detection)]()
[![GitHub contributors](https://img.shields.io/github/contributors/deepikagandla7456/Fake-News-Detection)]()
[![GitHub last-commit](https://img.shields.io/github/last-commit/deepikagandla7456/Fake-News-Detection)]()

This project helps to detect whether a news article is real or fake by analyzing the content using machine learning techniques. It is built using Python and a few popular libraries for data processing and model building.



## Table of Contents
- [Fake News Detection](#fake-news-detection)
  - [About](#about)
  - [Future Work](#future-work)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Setup and Installation](#setup-and-installation)
  - [Usage](#usage)
  - [Screenshots](#screenshots)
  - [License](#license)

---

## About

The goal of this project is to reduce the spread of misleading news by classifying articles as either ***Fake*** or ***Real.*** A dataset containing labeled news headlines and content was used to train a machine learning model. The text data is preprocessed using NLTK, converted into numerical format using TF-IDF, and then passed into a Logistic Regression model for prediction.

The final model performs well and gives accuracy score on the test data.

---

## Future Work

- Add a user-friendly web interface.
- Explore advanced models like LSTM or BERT for better performance.
- Support classification in multiple languages.
- Optimize the model for faster predictions.

---

## Features

- Detects fake news based on content.
- Preprocesses text using standard NLP techniques.
- Uses TF-IDF vectorizer and Logistic Regression model.
- Displays accuracy and confusion matrix for evaluation.

---

## Requirements

- Python 3.10 or higher version
- pandas  
- numpy  
- scikit-learn  
- nltk  

---

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/deepikagandla7456/Fake-News-Detection.git
cd Fake-News-Detection
```
2. Install required packages:

```shell
pip install -r requirements.txt
```

## Usage
To run the script:
```shell
python app.py
```

## Screenshots

**Model Evaluation Output**
<img width="1753" height="541" alt="Image" src="https://github.com/user-attachments/assets/d3881b74-db65-42a8-b20f-0408836bbf0b" />
<img width="1756" height="466" alt="Image" src="https://github.com/user-attachments/assets/64f10264-8935-444b-9f14-4d74962781fb" />
**Article Prediction Output**
<img width="1756" height="479" alt="Image" src="https://github.com/user-attachments/assets/e652c02f-1d60-4c3b-b1f3-84272a40de52" />

## License

This project is licensed under the [MIT](LICENSE) - see the [LICENSE](LICENSE) file for details.
