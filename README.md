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
2. **Build and run frontend**
   ```bash
   npm install
   npm run dev
3. **Build and run backend**
   ```bash
   uvicorn controller:app --reload
