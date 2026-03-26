# 🎓 College Predictor ML

A full-stack machine learning web app that recommends the best engineering colleges in India based on your budget, state preference, college type, gender preference, minimum rating, and required facilities.

Built with **Flask** + **Random Forest** + a custom dark-mode UI.

---

## 📁 Project Structure

```
college-predictor-ml/
│
├── app.py                  ← Flask web server
├── requirements.txt        ← Python dependencies
├── README.md
│
├── data/
│   └── colleges.csv        ← College dataset (40+ colleges)
│
├── templates/
│   └── index.html          ← Frontend UI (dark theme)
│
└── model/
    ├── train_model.py      ← Train & save the ML model
    ├── predictor.py        ← Inference logic (used by app.py)
    └── model.pkl           ← Auto-generated after training
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/college-predictor-ml.git
cd college-predictor-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python model/train_model.py
```
This reads `data/colleges.csv`, trains a Random Forest classifier, and saves `model/model.pkl`.

### 4. Start the web app
```bash
python app.py
```
Open your browser at **http://localhost:5000**

---

## 🧠 How It Works

1. **Data** — `colleges.csv` contains 40+ engineering colleges with features like rating, fees, campus size, faculty count, facilities, state, and college type.

2. **Training (`train_model.py`)** — Features are engineered (facility flags, label encoding, MinMax scaling) and a `RandomForestClassifier` (300 trees) is trained to predict/rank colleges. The full artifact (model + encoders + scaler + dataframe) is pickled.

3. **Inference (`predictor.py`)** — At request time, user preferences are used to:
   - Hard-filter the college dataframe
   - Build feature vectors for surviving colleges
   - Score them with `predict_proba` (confidence per college)
   - Rank by confidence + rating and return top-N results

4. **Web App (`app.py`)** — Flask serves the UI and exposes a `/predict` POST endpoint that returns JSON results rendered dynamically in the browser.

---

## 🎛️ Filter Options

| Filter | Options |
|---|---|
| Annual Budget | ₹30,000 – ₹6,00,000 (slider) |
| Minimum Rating | 0.0 – 5.0 (slider) |
| Preferred State | Any / specific state |
| College Type | Any / Public-Government / Private |
| Gender | Any / Co-Ed |
| Required Facilities | Gym, Sports, Wi-Fi, Labs, Cafeteria, Library, Hostels |
| Number of Results | Top 3 / 5 / 10 |

---

## 📊 Dataset

`data/colleges.csv` — 40+ colleges with columns:

- `College Name`, `State`, `City`
- `College Type` (Public/Government or Private)
- `Rating` (out of 5)
- `Average Fees` (annual, INR)
- `Campus Size`
- `Total Faculty`
- `Courses` (semicolon-separated)
- `Facilities` (semicolon-separated)
- `Genders Accepted`
- `Established Year`

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask 3.x
- **ML**: scikit-learn (Random Forest, LabelEncoder, MinMaxScaler)
- **Data**: pandas, numpy
- **Frontend**: Vanilla HTML/CSS/JS (no frameworks, dark theme)
- **Fonts**: Syne + Manrope (Google Fonts)

---

## Author

Sami Noor Saifi
