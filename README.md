# [Fake News Detection](Fake-News-Detection)

[![GitHub license](https://img.shields.io/github/license/deepikagandla7456/Fake-News-Detection)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/deepikagandla7456/Fake-News-Detection)]()
[![GitHub contributors](https://img.shields.io/github/contributors/deepikagandla7456/Fake-News-Detection)]()
[![GitHub last-commit](https://img.shields.io/github/last-commit/deepikagandla7456/Fake-News-Detection)]()

This project helps to detect whether a news article is real or fake by analyzing the content using machine learning techniques. It is built using Python and a few popular libraries for data processing and model building.

---

## Table of Contents

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

- Python 3.x  
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
Try typing a news sentence to see if it’s classified as real or fake.
## Screenshots
Sample Input:

Output 


## License

This project is licensed under the [MIT](LICENSE) - see the [LICENSE](LICENSE) file for details.
