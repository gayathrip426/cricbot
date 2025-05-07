# 🏏 CRICBOT - AI-Powered Cricket Assistant Chatbot

CRICBOT is an intelligent chatbot designed to provide, historical statistics, trivia, game rules, and personalized content. It combines Natural Language Processing (NLP) techniques with cricket data sources to offer an engaging and informative experience for cricket fans.

---

## 🚀 Deployment

Follow these steps to deploy and run the CRICBOT locally.

### 📋 Prerequisites

Make sure you have the following installed:

- **Node.js** (v14 or above)
- **npm**
- **Python** (3.8 or above)
- **uvicorn** and **FastAPI** (`pip install fastapi uvicorn`)
- Any other backend Python dependencies (install using `pip install -r requirements.txt` if applicable)

### 🛠️ Steps to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/gayathrip426/cricbot.git
   cd cricbot
2. **Download pretrained models and cricket database**
   - Pretrained, finetuned models are available [here](https://drive.google.com/drive/folders/161_Tc9hq45wVkzgcFmCsQJdA7gA7Va5n?usp=sharing)
   - Cricket database built from [Cricsheet](https://cricsheet.org/downloads/), reference script to parse the JSON files [here](https://drive.google.com/file/d/1yJKPoiuarmCd9sy7BVMkZKTcyduZ8XfU/view?usp=sharing)
   - Prebuilt database [here](https://drive.google.com/file/d/1uvPzsCimeqRzB8VMgZqkOprXggF-IOWr/view?usp=sharing)
   - Scripts and Datasets used to train NER and Intent recognition pipeline [here](https://drive.google.com/drive/folders/1cHM1R2woh-kgiY--x_4oJdj65iismOe5?usp=sharing)
   - After setting up the models and the database, make sure the directory structure look like this:
   ```text
      project-root/
      └── chatbot_backend/
          ├── cricket_stats.db
          └── fine_tuned_models/
              ├── cricket_ner_model/
              └── fine_tuned_intent_model/
                  └── model-best/
                      └── config.json
      ```
3. **Build and run frontend**
   ```bash
   npm install
   npm run dev
4. **In another terminal Build and run backend**
   ```bash
   uvicorn controller:app --reload
5. **Go to the url set by the variable [origins](https://github.com/gayathrip426/cricbot/blob/main/chatbot_backend/controller.py#L22) in [controller.py](https://github.com/gayathrip426/cricbot/blob/main/chatbot_backend/controller.py)**
