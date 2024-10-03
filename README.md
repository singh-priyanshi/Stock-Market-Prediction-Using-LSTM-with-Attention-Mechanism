# Stock Market Prediction Using LSTM with Attention Mechanism

## Project Overview

This project implements a machine learning pipeline to predict stock market trends using **Long Short-Term Memory (LSTM)** networks enhanced with an **Attention Mechanism**. The solution integrates **BERT-based sentiment analysis** of market news to improve the accuracy of the predictions.

Stock market data is highly volatile and non-linear, making traditional methods of forecasting inadequate. This project aims to leverage deep learning techniques to capture temporal patterns, sentiment from external sources, and market indicators, thereby improving prediction performance.

## Architecture

The project follows a modular pipeline with the following key stages:

### 1. **Data Collection and Preprocessing**
   - Historical stock price data from Yahoo Finance and other APIs.
   - News articles and social media feeds are processed using **BERT** for sentiment analysis to capture market sentiment.
   - Preprocessing involves normalization of stock data and tokenization of text for BERT.

### 2. **Feature Engineering**
   - Stock price trends, volume, and other technical indicators (moving averages, RSI, etc.) are included as features.
   - Sentiment scores from news articles are aggregated and appended as additional features to enhance predictive capability.

### 3. **Model Architecture**
   The core model of the system is built using an **LSTM network** to handle sequential time-series data and predict future stock prices. The architecture includes:
   - **LSTM layers**: Responsible for capturing temporal dependencies from the stock data.
   - **Attention Mechanism**: Introduced to focus on key timesteps or features that significantly impact prediction, ensuring the model adapts dynamically to important events or changes.
   - **Dense layers**: After the LSTM outputs, dense layers are applied to refine predictions.
   - **BERT-based Sentiment Analysis Module**: This module provides a sentiment score that feeds into the LSTM model as an external feature.

### 4. **Training and Optimization**
   - The LSTM model is trained using **mean squared error (MSE)** as the loss function and **Adam optimizer** to minimize the prediction error.
   - Sentiment features are combined with stock data features to improve predictions by integrating market mood with technical patterns.
   - Hyperparameter tuning (e.g., number of LSTM units, learning rate) is done using **Grid Search**.

### 5. **Evaluation**
   - The model is evaluated using **root mean square error (RMSE)** and **mean absolute error (MAE)** on the test data.
   - Additional evaluation metrics include accuracy of trend direction (whether the stock moves up or down).
   - Sentiment analysis performance is evaluated using **F1-score** and **accuracy** on a labeled dataset of market news.

### 6. **Deployment**
   - The model is packaged for deployment in a cloud environment, utilizing AWS Lambda and S3 for model serving and storage.
   - An API is provided to get real-time stock predictions using recent data.

## Technologies Used
- **LSTM Networks** with attention mechanism for time series forecasting.
- **BERT** for sentiment analysis on market-related news and social media data.
- **PyTorch** as the primary deep learning framework for model implementation.
- **Scikit-learn** for feature engineering and hyperparameter tuning.
- **Pandas** and **NumPy** for data processing.
- **Matplotlib** and **Seaborn** for visualizing trends and model performance.
- **AWS Lambda** and **S3** for cloud-based model deployment.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- PyTorch
- Transformers (for BERT model)
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### Installation

Clone this repository:
```bash
git clone https://github.com/yourusername/stock-market-prediction-lstm.git
cd stock-market-prediction-lstm
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Data Preprocessing**: Prepare the data by running the data preprocessing scripts.
2. **Training**: Train the LSTM model with sentiment features by running the training script:
   ```bash
   python train_model.py
   ```
3. **Evaluation**: Run the evaluation script to test the model on the test dataset:
   ```bash
   python evaluate_model.py
   ```

### Deployment
You can deploy the model on AWS Lambda for real-time predictions. Refer to the `deployment/` folder for setup instructions.

## Results
<img width="581" alt="image" src="https://github.com/user-attachments/assets/ebca5cb7-61ad-46f4-89f2-5935cb22ee2b">

The model demonstrates significant improvement in predictive performance by incorporating sentiment analysis with stock data. With the attention mechanism, the model better identifies important events influencing the market, leading to:
- Improved trend prediction accuracy.
- More robust handling of volatile periods.

## Future Work

The current model can be extended by:
- Incorporating more advanced sentiment analysis techniques, such as topic modeling.
- Using reinforcement learning to optimize stock trading strategies based on model predictions.
- Expanding to other financial markets and including more diverse data sources.

