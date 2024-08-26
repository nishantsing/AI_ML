# AI_ML

## Deploy AI apps Free(Local First development)
- [host ALL your AI locally](https://youtu.be/Wjrdr0NU4Sk)
- Ollama to download open source AI model(phi3 - chatmodel), (llava-llama3 - Image) on machine.(private and dont share your data like chatgpt or gemini )
    - ollama run phi3
    - ollama list
    - ollama rm  
- [Vercel AI SDK](https://sdk.vercel.ai/docs/introduction)
- libraries(npm)
    - ai
    - ollama
    - ollama-ai-provider
- create frontend using react/next
- see the cheap-ai code to understand more.

## Roadmap
- Python
- version control git
- DSA
- SQL
- Maths(probability, linear, calculus) and statistics
- Data Handling(pandas, numpy, matplotlib, seaborn)
- ML Algos (supervised(labeled data), unsupervised(unlabeled data)) [Tensorflow, PyTorch, Scikit Learn]
- Advance (Ensemble, deep learning(NLP, Computer Vision))
- Deployment(flask, django, docker)

## Gen AI Resources

- [Gen AI](https://github.com/genieincodebottle/generative-ai)

## SDK/ Frameworks
[Lang Chain](https://js.langchain.com/v0.2/docs/introduction/)
[vercel AI SDK](https://sdk.vercel.ai/docs/introduction)
[Haystack](https://haystack.deepset.ai/)
[Github Copilot](https://github.com/features/copilot)
[Codium](https://www.codium.ai/)

- spine cone
- vector databases


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

or can use codeium, github copilot for VSCode but they are not private.

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


## AI Projects | python libraries

1. Sentiment Analysis(nltk, pandas, scikit learn)
2. Image Classifier (tenserflow, keras, pandas, numpy, matplotlib)
3. AI Voice Assistant (speech recognition module, pyttsx3, llm(openai, ollama))
4. Recommendation System (suprise, scikit learn, tensorflow, pytorch, pandas, numpy)
5. AI Agent (Langchain, ollama(free), gpt(paid))

## Text to speech

```py
from pygame import mixer
from gtts import gTTS

def main():
    tts = gTTS('WOW')
    tts.save('output.mp3')
    mixer.init()

mixer.music.load('output.mp3')
mixer.music.play()

if __name__ = "__main__":
    main()
```
## ChatBot 

1. Create it locally with ollama
2. Create it with some online llm model like gemini, openai
3. Create it using your own training data


## RAG(Retrieval Augmented Generation)
- We are providing some extra data to the model, and can use that instead of its on training data.

## Private AI and connecting it to our docs

- [Hugging Face](https://huggingface.co/models) - you can download the llmodels(which are pre trained) using ollama [Models you can run](https://ollama.com/library)
- use WSL for windows. wsl --install
- sudo apt update
- sudo apt upgrade -y
- curl -fsSL https://ollama.com/install.sh | sh
- localhost:11434
- ollama run llama2(mistral ... many more)

- how to teach them on our data? the process is called fine tuning, but fine tuning such a model on new data can be very expensive at it will require gpus.
- VMWare private AI can help you do that. [VMWare private AI](https://www.vmware.com/products/cloud-infrastructure/vsphere/ai-ml) VMare collaborating with Nvidia, IBM, Intel.

or we can tach them using prompts which is not as extensive and takes less time.
or RAG(Retrieval Augmented Generation) connecting LLM to a vector database.

- [Private GPT](https://docs.privategpt.dev/overview/welcome/introduction)
- [Private GPT by Martinez](https://github.com/joz-it/imartinez-privateGPT)
- [Installing PrivateGPT on WSL with GPU support](https://medium.com/@docteur_rs/installing-privategpt-on-wsl-with-gpu-support-5798d763aa31)



## Octo AI

[PDF Summarizer - Adrian](https://youtu.be/OIYevBOSMxY)



## Local AI with Open Web UI
- [Open Web UI GUI](https://docs.openwebui.com/)
- Setup WSL (Windows)
    Install WSL and Ubuntu
    wsl --install

    Connect to a WSL Instance in a new window
    wsl -d Ubuntu

    Install Ollama
    https://ollama.com/download

    Add a model to Ollama
    ollama pull llama2


- "Watch" GPU Performance in Linux - watch -n 0.5 nvidia-smi

- Get Dockers GPG key
    #Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    #Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update

- Install Docker
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

- Run Open WebUi Docker Container
sudo docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main

localhost:8080

- Stable Diffusion Install(Creating Images)
Prereqs
Pyenv
- Install Pyenv prereqs
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git

#Install Pyenv

    curl https://pyenv.run | bash

#Install Python 3.10

pyenv install 3.10

#Make it global

pyenv global 3.10


Install Stable Diffusion
wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh

- Make it executable

chmod +x webui.sh

#Run it

./webui.sh --listen --api

## Using Fabric

- [Fabric - Network Chunk](https://youtu.be/UbDyjIIGaxQ)
- [Fabric](https://github.com/danielmiessler/fabric?tab=readme-ov-file#environmental-variables)
