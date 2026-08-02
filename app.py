import streamlit as st
import joblib
from parser import parse_resume

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="CareerCast", page_icon="🔮", layout="centered")

# ---- Purple Family Theme ----
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #1A0E2E 0%, #2D1B4E 100%);
        color: #F3E8FF;
    }
    h1 {
        color: #E9D5FF;
        font-weight: 800;
        text-shadow: 0px 0px 12px rgba(168, 85, 247, 0.5);
    }
    h2, h3, h4 { color: #D8B4FE; }
    p, span, label, .stMarkdown, .stCaption { color: #F3E8FF !important; }

    /* Form inputs */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stTextArea textarea {
        background-color: #3B2364;
        color: #FFFFFF;
        border: 1px solid #A855F7;
        border-radius: 8px;
    }
    .stTextInput label, .stNumberInput label, .stTextArea label {
        color: #E9D5FF !important;
        font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #7C3AED, #A855F7);
        color: #FFFFFF;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.6em 2em;
        border: none;
        box-shadow: 0px 4px 14px rgba(124, 58, 237, 0.5);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #A855F7, #C084FC);
        color: #1A0E2E;
    }

    /* Divider */
    hr { border-color: #7C3AED; }

    /* Metric boxes */
    div[data-testid="stMetric"] {
        background-color: #2D1B4E;
        border: 1px solid #7C3AED;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stMetricLabel"] { color: #D8B4FE !important; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; }

    /* Career prediction cards */
    .career-card {
        background: linear-gradient(90deg, #3B2364, #2D1B4E);
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #C084FC;
    }
    .career-name { font-size: 18px; font-weight: 700; color: #FFFFFF; }
    .career-pct { font-size: 16px; color: #D8B4FE; font-weight: 600; }

    /* Expander */
    .streamlit-expanderHeader { color: #E9D5FF !important; }

    /* Success/Error boxes */
    .stAlert { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔮 CareerCast</h1>", unsafe_allow_html=True)
st.caption("Predicting Career Paths with AI-Powered Skill Intelligence")
st.divider()

st.markdown("### 📝 Structured Profile Input")

with st.form("profile_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name *")
        education = st.text_input("Highest Education (e.g. B.Tech, MBA) *")
    with col2:
        experience_years = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
        skills = st.text_input("Skills (comma separated) *")

    resume_text = st.text_area("📄 Paste full resume text (optional, for NER parsing)", height=150)
    submitted = st.form_submit_button("🚀 Submit & Predict")

if submitted:
    errors = []
    if not name.strip():
        errors.append("Name is required.")
    if not skills.strip():
        errors.append("Skills field is required.")
    if not education.strip():
        errors.append("Education field is required.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        st.success(f"✅ Profile submitted for {name}")

        combined_text = skills
        if resume_text.strip():
            parsed = parse_resume(resume_text)
            st.markdown("### 🔍 Parsed Resume Data (spaCy NER)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Skills Found", len(parsed["skills"]))
            c2.metric("Education Tags", len(parsed["education"]))
            c3.metric("Institutions", len(parsed["institutions"]))
            with st.expander("View parsed details"):
                st.write(parsed)
            combined_text = skills + " " + " ".join(parsed["skills"])

        vec = vectorizer.transform([combined_text])
        probs = model.predict_proba(vec)[0]
        classes = model.classes_

        st.markdown("### 📊 Career Prediction Report")
        results = sorted(zip(classes, probs), key=lambda x: -x[1])
        for career, p in results[:5]:
            st.markdown(f"""
                <div class="career-card">
                    <span class="career-name">{career}</span><br>
                    <span class="career-pct">{p*100:.1f}% match</span>
                </div>
            """, unsafe_allow_html=True)
            st.progress(float(p))