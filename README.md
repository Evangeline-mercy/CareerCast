# CareerCast — Milestone 1

AI-powered career path prediction system.

## Milestone 1 Deliverables

- Resume parsing using spaCy NER (skills, education, institutions)
- Structured user-profile form with validation
- Baseline Logistic Regression classifier trained on historical career dataset
- Career prediction report with accuracy and coverage metrics

## How to run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python train_model.py
streamlit run app.py
```

## Application Screenshots

### 1. Home Page

![Home](images/1_home.png)

---

### 2. User Profile Input

![Form](images/2_form.png)

---

### 3. Resume Parsing

![Resume Parsing](images/3_parsed_resume.png)

---

### 4. Career Prediction Report

![Career Prediction](images/4_prediction.png)