# 🎯 PKM2Pred: AI-Powered Tool for Predicting PKM2 Modulators and Potency

## 🛠️ Tech Stack

### ⚙️ Backend
- **Python 3.10**
- **Flask** – RESTful API to serve predictions
- **Gunicorn** – WSGI HTTP server for Flask deployment
- **PaDELPy** – Python wrapper for molecular descriptor extraction via PaDEL-Descriptor
- **scikit-learn** – Machine learning models (Random Forest, Decision Tree)
- **Pandas & NumPy** – Data manipulation and statistical operations

### 🌐 Frontend
- **Next.js** – React-based framework for building UI
- **Tailwind CSS** – Utility-first CSS framework for styling
- **Chart.js** – For dynamic pie/bar/AC50 visualizations
- **Axios** – HTTP client for API communication

### ☁️ Deployment
- **Frontend**: Deployed on [Vercel](https://vercel.com) for seamless CI/CD and hosting
- **Backend**: Hosted via **Gunicorn** and exposed publicly using **Ngrok** on a private server (BIT Mesra)

### 🧪 Descriptor Generation
- **PaDELPy** – Used to interface with PaDEL-Descriptor in Python

---

## 🚀 Overview

**PKM2Pred** is an open-source, AI-powered web server designed to:
- 🧪 Classify unknown chemical compounds as **activators**, **inhibitors**, or **decoys** of the PKM2 enzyme.
- 📉 Predict the **bioactivity (AC50)** range of **activators** using a regression model with confidence intervals.
- 🧬 Identify key molecular descriptors (e.g., **WTPT-5**, **SRW9**, **nHeteroRing**) essential in PKM2 activity.

PKM2 is a glycolytic enzyme critical in cancer metabolism, making it a promising drug target in oncology. PKM2Pred speeds up **early drug discovery** by allowing rapid in-silico screening of compounds.

---

## 🧠 Machine Learning Architecture

### 🔹 Classification Pipeline
- **Goal**: Classify molecules as *Activator*, *Inhibitor*, or *Decoy*
- **Algorithm Used**: Random Forest Classifier (RFC)
- **Input**: Top 28 statistically significant molecular descriptors
- **Output**: Compound class (Activator / Inhibitor / Decoy)
- **Accuracy**: 94%
- **MCC**: 90.02%

### 🔹 Regression Pipeline
- **Goal**: Predict AC50 range for Activators
- **Algorithm**: Bootstrapped Decision Tree Regressor
- **Method**:
  - 100 model iterations (bootstrapped)
  - Predict range based on selected **confidence interval** (50%, 75%, 95%)
  - AC50 is shown with median value and bounds
- **Key Insight**: Higher confidence → wider but more reliable range

---

## 🧪 Molecular Descriptor Engineering

- **Initial Descriptors**: 1875 via PaDELPy
- **Refined**: 28 statistically significant via:
  - Null/removal
  - Correlation filtering (>0.95)
  - Kruskal-Wallis H-test (p < 0.05)

**Top 3 descriptors:**
| Descriptor     | Description                                                       |
|----------------|-------------------------------------------------------------------|
| WTPT-5         | Path length from nitrogen atoms — indicates structural complexity |
| SRW9           | Self-returning walk of order 9 — molecular connectivity           |
| nHeteroRing    | # of rings with heteroatoms — relates to reactivity and potency   |

---

## 🌐 Web Application

### 🔧 Frontend
- Built using **Next.js**
- Deployed on **Vercel**
- Handles user input and visualizations (charts, plots)

### 🔧 Backend
- Built using **Python Flask**
- Deployed via **Gunicorn** + **Ngrok Tunnel** (BIT Mesra-hosted server)
- Computes descriptor values and makes predictions
- Returns data as JSON → parsed and displayed on frontend

### 🔍 Features
1. Upload molecules via **SMILES**
2. Select **confidence interval** (50%/75%/95%) for AC50
3. View **classification** results in pie chart
4. View **AC50 prediction** in bar + scatter plot
5. **Download** results as CSV

---

## 📁 Project Structure
```bash
PKM2Pred/
│
├── backend/ # Flask backend
│ ├── model/ # Pickled models (classifier, regressor)
│ ├── descriptors/ # PaDEL descriptor generator
│ ├── app.py # Flask app
│ └── utils.py # Helper functions
│
├── frontend/ # Next.js frontend
│ ├── components/ # Reusable React components
│ ├── pages/ # Routes (index.js, about.js, etc.)
│ ├── public/ # Static files
│ └── styles/ # Tailwind CSS styles
│
├── data/ # Dataset (SMILES, descriptor CSVs)
├── requirements.txt # Backend Python dependencies
├── padel.sh # Shell script for PaDEL descriptor calculation
└── README.md # You are here!
```


## ⚙️ How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Arya-Chakraborty/PKM2Pred.git
cd PKM2Pred
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```

### 4. Ngrok Tunneling (Backend to Public)
```bash
ngrok http 5000
```
Set the frontend API URL in .env as the forwarded ngrok address.


## 📈 Results Summary
- Classification Accuracy: 94%

- Interactive Results: Classification Pie Chart, AC50 Range Plot

- User Configurable: Select AC50 confidence range

- Exportable: Download as CSV for further use

## 🌐 Try it Live
https://pkm2pred.vercel.app ➡️


## 📄 Citation
If you use this tool in your work, please cite:
```bash
Accelerating Anticancer Drug Discovery with PKM2Pred: A Scalable AI Tool for Rapid Identification and Potency Estimation of PKM2-Targeting Compounds
Aryan Raj Saxena, Palak Singla, Arya Chakraborty, Archit Mukherjee, Mrityunjay Nigam, Alok Jain.
Advanced BioComputing Lab, BIT Mesra.
```

## 🧪 License
This project is open-source under the MIT License.
