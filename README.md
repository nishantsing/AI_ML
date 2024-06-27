# AI_ML

## Integrating your local large language model to VSCode
- [Ollama - Download](https://ollama.com/)
- ollama --help in terminal to check if its installed
- [To check Models ranking](https://evalplus.github.io/leaderboard.html)
- go to ollama website and search for CodeQwen(4GB) or DeepSeek code v2(8GB) models because they are free unlike GPT-4
- ollama run codeqwen
- continue(own AI copilot) extension in VSCode
- create a customCommands
```json
{
  "name":"step",
  "prompt":"{{{input}}}\n\nExplain the selected code step by step",
  "description":"Code explanation"
}

```
- Ctrl + I -> ask the model to do generate some code
- "Select the code "Ctrl + L
- 

## 15 Beginner ML Projects (Learn by doing)

1. Iris Flower (classification)
    - scikit learn / Tensorflow (DataSet)
    - Jupyter Notebook, Google collab, vscode
    - split data into training and testing samples. Perform scailing on it.
    - Classification Algos - (Linear/Logistic Regression, SVM, KNN, Decision Trees)
    - Train model on training data.
    - Evaluate performance on testing data(Precision, Recall, Accuracy, F-1 Score, Confusion Matrix)

2. Predicting the price of house based on its features using supervise learning
    - Kaggle, UCI repository (DataSet)
    - Data Preprocessing(cleaning)
    - Logistic regression, decision tree, random forests
    - Split data (Make training set larger)
    - Training Data, Also perform cross-validation
    - Evaluate Performance (MSE, MAE, R-squared)
  
     
3. Classify Emails as Spam or Not Spam
    - Extract features from text(use techniques such as bag-of-words and TF-IDF)
    - Naive Bayes, SVM, Random Forest
    -  Evaluate performance on testing data(Precision, Recall, Accuracy, F-1 Score)
    -  Hyperparameter Tuning & Model Evaluation

4. Predicting Customer Churn for subscription-based business
    - kaggle
5. Building Recommendation Engine to Suggest Products (collaborating filtering)
    - Use Collaborative Filtering Algorithm(User based, item based)
    - Integrating to e-commerce site
6. Predicting Sentiments of Movie Reviews using NLP (NLP)
7. Building a Chatbot to Answer Customer Queries Using NLP
    - NLP
        - Process Data
        - Tokenization, lemmatization, part-of-speech tagging, and named entity recognition.
        - Learn frameworks such as NLTK, CoreNLP, GenSim
8. Predicting Customer will default on their loan using classification
    - Data Visulaization
    - Imbalance dataset
    - Overfitting (Cross-validation & regularization)
    - Bias Variance trade off

9. Building face recognition system using deep learning (CV, Neural Networks, DL)
    - Image processing(Computer Vision), Face Recognition
    - OpenCV, Data Augmentation
10. Classify Images Based on their content(CNN)
    - Image Preprocessing (Resizing to standard size, Grayscale conversion)
    -  Start building CNN model(use prebuilt architectures such as VGG, ResNet)
11. Detecting Credit Card fraud using anomaly detection
12. Building a model to predict probability of medical diagnosis
13. Classify Customer Complaints into different categories
    - clustering is unsupervised learning technique used in ML to group similar data points
    - K Means, Hierarichal clustering
    - Evaluate performance(Dunn Index, silhoutte score)
14. Predicting Stock Prices based on historical performance
    - Time series Forecasting
    - kaggle
    - smoothing, trained analysis, season decomposition, auto correlative analysis.
    - ML algos(ARIMA, SARIMA, LSTM)
15. Speech recognition
    - Building Speech to Text app using Python and google cloud speec to text api.
    - Voice assistant using raspberry pi
    - speaker identification
