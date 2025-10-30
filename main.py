import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import re
from dotenv import dotenv_values
import os

# ------------------- Load API Key -------------------
config = dotenv_values(".env")
api_key = config.get("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file. Please check your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI()

# ------------------- Initialize session state -------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

# ------------------- Themes -------------------
THEMES = {
    "Light": {
        "background": "#f5f7fa",
        "text": "#000000",
        "card": "#ffffff",
        "accent": "#0078ff",
    },
    "Dark": {
        "background": "#0e1117",
        "text": "#ffffff",
        "card": "#1c1c1c",
        "accent": "#00bfff",
    },
    "Glassmorphic": {
        "background": "linear-gradient(135deg, rgba(20,20,20,0.9), rgba(40,40,40,0.8))",
        "text": "#ffffff",
        "card": "rgba(255,255,255,0.15)",
        "accent": "#00ffff",
    },
}

theme = THEMES[st.session_state.theme]

# ------------------- Inject CSS -------------------
st.markdown(f"""
    <style>
        .stApp {{
            background: {theme['background']};
            color: {theme['text']};
        }}
        .theme-toggle {{
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: {theme['card']};
            color: {theme['text']};
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            box-shadow: 0 0 15px rgba(0,0,0,0.3);
            font-size: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .theme-toggle:hover {{
            transform: scale(1.1);
            background-color: {theme['accent']};
            color: white;
        }}
        .glass-card {{
            background: {theme['card']};
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin: 15px 0;
        }}
        .stDownloadButton > button {{
            background-color: {theme['accent']};
            color: white;
            border-radius: 10px;
            border: none;
        }}
    </style>
""", unsafe_allow_html=True)

# ------------------- Theme Toggle Icon -------------------
toggle_icon = "🌙" if st.session_state.theme == "Light" else ("🪩" if st.session_state.theme == "Dark" else "☀️")

# Place the toggle button
if st.button(toggle_icon, key="theme_toggle", help="Switch Theme", use_container_width=False):
    if st.session_state.theme == "Light":
        st.session_state.theme = "Dark"
    elif st.session_state.theme == "Dark":
        st.session_state.theme = "Glassmorphic"
    else:
        st.session_state.theme = "Light"
    st.rerun()

# ------------------- Header -------------------
st.markdown(
    f"<h1 style='text-align:center; color:{theme['accent']}'>ResuMate 🧠 - AI Resume Analyzer</h1>",
    unsafe_allow_html=True
)

# ------------------- Functions -------------------
def extract_pdf_text(uploaded_file):
    try:
        return extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""


def calculate_similarity_bert(text1, text2):
    ats_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    emb1 = ats_model.encode([text1])
    emb2 = ats_model.encode([text2])
    return cosine_similarity(emb1, emb2)[0][0]


def get_report(resume, job_desc):
    prompt = f"""
    You are an AI Resume Analyzer.
    Compare the following resume and job description.

    Resume: {resume}
    ---
    Job Description: {job_desc}

    Provide:
    - A scored evaluation (1–5) for each key requirement.
    - A detailed reason with ✅ / ❌ / ⚠️.
    - Suggestions for improvement at the end.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content


def extract_scores(text):
    pattern = r'(\d+(?:\.\d+)?)/5'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]


# ------------------- UI -------------------
if not st.session_state.form_submitted:
    with st.form("resume_form"):
        resume_file = st.file_uploader("📄 Upload Resume (PDF)", type="pdf")
        st.session_state.job_desc = st.text_area("💼 Paste Job Description", placeholder="Enter job description here...")

        submitted = st.form_submit_button("🔍 Analyze Resume")

        if submitted:
            if st.session_state.job_desc and resume_file:
                st.info("Extracting text from resume...")
                st.session_state.resume = extract_pdf_text(resume_file)
                st.session_state.form_submitted = True
                st.rerun()
            else:
                st.warning("Please upload both Resume and Job Description.")

if st.session_state.form_submitted:
    score_place = st.info("Analyzing Resume...")

    ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)
    report = get_report(st.session_state.resume, st.session_state.job_desc)
    report_scores = extract_scores(report)
    avg_score = sum(report_scores) / (5 * len(report_scores)) if report_scores else 0

    score_place.success("✅ Analysis Complete!")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ATS Similarity Score", f"{round(ats_score, 2)}")
    with col2:
        st.metric("AI Evaluation Avg", f"{round(avg_score, 2)}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("AI Generated Resume Feedback 💡")
    st.markdown(report, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        label="⬇️ Download Report",
        data=report,
        file_name="AI_Resume_Report.txt",
    )
