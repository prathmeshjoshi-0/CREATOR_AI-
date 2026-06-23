# 🎬 Creator AI
### YouTube Analytics Platform for Smart Creators

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://aagttuyfdd2eqnnray2lds.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![MySQL](https://img.shields.io/badge/MySQL-Aiven%20Cloud-4479A1?style=for-the-badge&logo=mysql)](https://aiven.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## What is Creator AI?

Creator AI is a data-driven YouTube analytics platform that helps content creators **predict video virality, track competitors, and optimize content strategy** — all powered by Machine Learning.

Think of it as a low-cost alternative to VidIQ or TubeBuddy, built for early-stage creators who want real insights without the expensive subscription.

> **91% AUC** on virality prediction using Random Forest — trained on ~4,373 real YouTube video records across multiple regions and categories.

---

## 🚀 Live Demo

👉 **[Try Creator AI here](https://aagttuyfdd2eqnnray2lds.streamlit.app/)**

---

## ✨ Features

| Page | What it does |
|------|-------------|
| 🏠 **Home** | Live scoreboard of top-performing videos |
| 📊 **My Channel Analyzer** | Deep-dive analytics on your own channel |
| 🔍 **Competitor Tracker** | Monitor what's working for rivals |
| 🔮 **Virality Predictor** | ML model predicts if your video will go viral + AI Title Scorer |
| 📈 **Trend Analysis** | Spot rising topics before they peak |
| 🔒 **Retention Analyzer** | Coming soon |

---

## 🧠 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **Database:** MySQL on Aiven Cloud
- **ML Models:** Random Forest (91% AUC), XGBoost
- **AI Layer:** Groq API (LLaMA 3)
- **Data Source:** YouTube Data API v3
- **Dataset:** ~4,373 YouTube video records across multiple regions & categories

---

## 📁 Project Structure

```
CREATOR_AI/
│
├── app.py                  # Main Streamlit app entry point
├── pages/
│   ├── home.py             # Live scoreboard
│   ├── channel_analyzer.py # My Channel Analyzer
│   ├── competitor.py       # Competitor Tracker
│   ├── virality.py         # Virality Predictor + AI Title Scorer
│   └── trends.py           # Trend Analysis
│
├── models/
│   └── random_forest.pkl   # Trained ML model
│
├── utils/
│   ├── db.py               # Aiven MySQL connection + caching
│   └── youtube_api.py      # YouTube Data API v3 helpers
│
├── requirements.txt
└── .env.example
```

---

## ⚙️ Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/prathmeshjoshi-0/CREATOR_AI-.git
cd CREATOR_AI-
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
GROQ_API_KEY=your_groq_api_key_here
DB_HOST=your_aiven_mysql_host
DB_PORT=your_port
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🤖 ML Model Details

| Model | AUC Score | Notes |
|-------|-----------|-------|
| Random Forest | **91%** | ✅ Production model |
| XGBoost | ~88% | Challenger model |

**Features used for virality prediction:**
- View count, like ratio, comment rate
- Video duration, publish hour/day
- Category, region, title characteristics
- Channel size and historical performance

---

## 🗃️ Dataset

- ~4,373 YouTube video records
- Multiple regions (US, IN, GB, CA, and more)
- Multiple categories (Entertainment, Education, Tech, Music, etc.)
- Sourced via YouTube Data API v3

---

## 🔮 Roadmap

- [ ] Retention Analyzer (in progress)
- [ ] 24-hour auto-retrain pipeline
- [ ] Power BI integration


---

## 👤 Author

**Prathmesh Joshi**
- GitHub: [@prathmeshjoshi-0](https://github.com/prathmeshjoshi-0)
- LinkedIn: https://www.linkedin.com/in/prathmesh-joshi-data/

---

## 📄 License

MIT License — feel free to fork and build on this.

---

*Built as a Final Year Capstone Project | Data Analysis*
