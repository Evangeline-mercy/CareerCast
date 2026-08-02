import streamlit as st
import joblib
from parser import parse_resume

# Page configuration (must come before any Streamlit output)
st.set_page_config(page_title="CareerCast - Milestone 1")

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.title("CareerCast - AI Career Path Prediction System")

st.write("Fill in your details below to predict a suitable career path.")

# -----------------------------
# User Profile Form
# -----------------------------
with st.form("career_form"):

    name = st.text_input("Full Name")

    skills = st.text_area("Skills (comma separated)")

    education = st.text_input("Education")

    experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=50,
        value=0
    )

    resume = st.text_area("Paste Resume (Optional)")

    submit = st.form_submit_button("Predict Career")

# -----------------------------
# Validation and Prediction
# -----------------------------
if submit:

    if not name.strip() or not skills.strip() or not education.strip():

        st.error("Please fill all the required fields.")

    else:

        st.success("Profile Submitted Successfully!")

        # Resume Parsing
        if resume.strip():

            parsed = parse_resume(resume)

            st.subheader("Resume Parsing Results")

            st.write("Skills Found:", parsed["skills"])
            st.write("Education Found:", parsed["education"])
            st.write("Institutions:", parsed["institutions"])
            st.write("Roles:", parsed["roles"])

            final_text = skills + " " + " ".join(parsed["skills"])

        else:

            final_text = skills

        # Convert input into vector
        vector = vectorizer.transform([final_text])

        # Predict Career
        prediction = model.predict(vector)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(vector)[0]

        st.subheader("Career Prediction Report")

        st.success(f"Predicted Career: {prediction}")

        st.write("### Confidence Scores")

        results = sorted(
            zip(model.classes_, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        for career, score in results:
            st.write(f"**{career}** : {score * 100:.2f}%")