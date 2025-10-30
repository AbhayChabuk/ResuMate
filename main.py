import streamlit as st                      # For Web Interface (Front-End)
import time # Needed for simulating processing delay
from pdfminer.high_level import extract_text      # To Extract Text from Resume PDF
from sentence_transformers import SentenceTransformer       # To generate Embeddings of text
from sklearn.metrics.pairwise import cosine_similarity     # To get Similarity Score of Resume and Job Description
from openai import OpenAI                             # API to use LLM's
import re                                     # To perform Regular Expression Functions
from dotenv import load_dotenv              # Loading API Key from .env file
import os


# Load environment variables from .env
load_dotenv()

# Fetch the key from the environment
api_key = os.getenv("OPENAI_API_KEY")


# --- 1. THEME DEFINITIONS ---
# Define the colors for each theme based on your specifications.
THEMES = {
    'white': {
        'background': '#E4FBFF',
        'title_color': '#7868E6',
        'placeholder_bg': '#EDEEF7', 
        'input_bg': '#B8B5FF',       
        'input_text_color': '#000000', 
        'report_bg': '#F0F8FF',       # Background for the final report box
        'report_text_color': '#000000',
        'button_style': 'background-color: #7868E6; color: white; border: none; padding: 10px 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'
    },
    'neon': {
        'background': '#1C1427',
        'title_color': '#7ECA9C',
        'placeholder_bg': '#CCFFBD', 
        'input_bg': '#40394A',       
        'input_text_color': '#FFFFFF', 
        'report_bg': '#292131',       # Background for the final report box
        'report_text_color': '#CCFFBD',
        'button_style': 'background-color: #7ECA9C; color: #1C1427; border: 2px solid #CCFFBD; padding: 10px 20px; border-radius: 5px; box-shadow: 0 0 10px #7ECA9C;'
    }
}

# --- 2. THEME STATE MANAGEMENT ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'white'

def toggle_theme():
    """Switches the theme between 'white' and 'neon'."""
    st.session_state.theme = 'neon' if st.session_state.theme == 'white' else 'white'
    # st.rerun() is removed here. Streamlit automatically reruns the script 
    # when a widget (button) in the main body is clicked.

# --- 3. CUSTOM CSS GENERATION & INJECTION ---
def apply_custom_theme():
    """Generates and injects the custom CSS for the current theme."""
    current_theme = THEMES[st.session_state.theme]
    
    # Define primary colors
    bg_color = current_theme['background']
    title_color = current_theme['title_color']
    placeholder_bg = current_theme['placeholder_bg']
    input_bg = current_theme['input_bg']
    input_text_color = current_theme['input_text_color']
    
    # Theme-specific toggle button styling
    # Use the Title color for the icon/border for simplicity
    toggle_color = title_color
    toggle_border = f'1px solid {toggle_color}'

    # CSS to override Streamlit defaults and apply custom colors
    custom_css = f"""
    <style>
    /* 1. Main Background */
    .stApp {{
        background-color: {bg_color};
    }}

    /* 2. Main Title/Header Color and all subheadings/text (st.subheader, st.write, st.info etc.) */
    .app-title {{
        color: {title_color};
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 20px;
    }}
    
    /* Target all standard text elements (labels, headers, general text) */
    h1, h2, h3, h4, .stMarkdown, p, .st-emotion-cache-10qik0c, .st-emotion-cache-13s6zbd {{
        color: {title_color} !important;
    }}
    
    /* Overriding specific Streamlit text areas for theme consistency */
    [data-testid="stForm"] > div > div:first-child h3 {{
        color: {title_color} !important;
    }}
    
    /* 3. Text Area Containers (Resume and Job Description placeholder) */
    [data-testid="stTextarea"], [data-testid="stTextInput"], [data-testid="stFileUploader"] {{
        background-color: {placeholder_bg};
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* 4. Input Text Area Background */
    [data-testid="stTextarea"] textarea, [data-testid="stTextInput"] input {{
        background-color: {input_bg};
        color: {input_text_color} !important; 
        border: 1px solid {placeholder_bg};
        border-radius: 6px;
    }}
    
    /* 5. Custom button styles (Analyze/Download) */
    .stButton > button {{
        {current_theme['button_style']}
        transition: all 0.2s ease-in-out;
    }}
    
    .stButton > button:hover {{
        opacity: 0.8;
    }}
    
    /* 7. Theme Toggle Button Styling (Simplified) */
    /* Target the button by its key/ID for fine-tuning */
    [data-testid="stKey-theme_toggle_button"] > button {{
        background: none !important; /* Remove background color */
        border: {toggle_border} !important; /* Subtle border using title color */
        color: {toggle_color} !important; /* Icon color using title color */
        box-shadow: none !important; /* Remove shadow */
        font-size: 1.25rem; /* Make icon slightly larger */
        padding: 0.25rem 0.5rem; /* Smaller padding */
    }}
    
    [data-testid="stKey-theme_toggle_button"] > button:hover {{
        background-color: rgba(120, 104, 230, 0.1) !important; /* Subtle hover effect */
        opacity: 1;
    }}


    /* 6. Report box styling */
    .report-container {{
        text-align: left; 
        background-color: {current_theme['report_bg']}; 
        color: {current_theme['report_text_color']} !important;
        padding: 15px; 
        border-radius: 10px; 
        margin: 5px 0;
        white-space: pre-wrap; /* Ensure text formatting from LLM is preserved */
    }}
    
    /* Adjust general Streamlit element backgrounds */
    .st-emotion-cache-1v0md8g, .st-emotion-cache-16qfowq {{
        background-color: {bg_color};
    }}


    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Apply the selected theme
apply_custom_theme()

#  Session States to store values 
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "resume" not in st.session_state:
    st.session_state.resume=""

if "job_desc" not in st.session_state:
    st.session_state.job_desc=""

# --- TITLE AND THEME TOGGLE (Replaced st.title) ---
col_title, col_toggle = st.columns([10, 1])

with col_title:
    # Use a custom class to apply the dynamic title color
    st.markdown('<h1 class="app-title">ResuMate 📝</h1>', unsafe_allow_html=True)

with col_toggle:
    # Theme Toggle Button using cleaner Lucide icons
    # The icon syntax is :icon_name:
    theme_icon = ":material/wb_sunny:" if st.session_state.theme == 'white' else ":material/dark_mode:"
    st.button(theme_icon, key="theme_toggle_button", on_click=toggle_theme)

st.markdown("---") # Separator


# <------- Defining Functions (Keep existing) ------->

# Function to extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        extracted_text = extract_text(uploaded_file)
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return "Could not extract text from the PDF file."


# Function to calculate similarity 
def calculate_similarity_bert(text1, text2):
    ats_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')       # Use BERT or SBERT or any model you want
    # Encode the texts directly to embeddings
    embeddings1 = ats_model.encode([text1])
    embeddings2 = ats_model.encode([text2])
    
    # Calculate cosine similarity without adding an extra list layer
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity


def get_report(resume,job_desc):
    # This check ensures the app doesn't crash if the key isn't set, although it will fail the API call later.
    if not api_key:
        st.error("OpenAI API Key not found. Please ensure you have set the OPENAI_API_KEY environment variable.")
        return "API Key Missing."
        
    client = OpenAI(api_key=api_key)

    # Change the prompt to get the results in your style
    prompt=f"""
    # Context:
    - You are an AI Resume Analyzer, you will be given Candidate's resume and Job Description of the role he is applying for.

    # Instruction:
    - Analyze candidate's resume based on the possible points that can be extracted from job description,and give your evaluation on each point with the criteria below:   
    - Consider all points like required skills, experience,etc that are needed for the job role.
    - Calculate the score to be given (out of 5) for every point based on evaluation at the beginning of each point with a detailed explanation.   
    - If the resume aligns with the job description point, mark it with ✅ and provide a detailed explanation.   
    - If the resume doesn't align with the job description point, mark it with ❌ and provide a reason for it.   
    - If a clear conclusion cannot be made, use a ⚠️ sign with a reason.   
    - The Final Heading should be "Suggestions to improve your resume:" and give where and what the candidate can improve to be selected for that job role.

    # Inputs:
    Candidate Resume: {resume}
    ---
    Job Description: {job_desc}

    # Output:
    - Each any every point should be given a score (example: 3/5 ). 
    - Mention the scores and  relevant emoji at the beginning of each point and then explain the reason.
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with OpenAI API: {e}")
        return "Error: Could not generate analysis report due to an API issue."


def extract_scores(text):
    # Regular expression pattern to find scores in the format x/5, where x can be an integer or a float
    pattern = r'(\d+(?:\.\d+)?)/5'
    # Find all matches in the text
    matches = re.findall(pattern, text)
    # Convert matches to floats
    scores = [float(match) for match in matches]
    return scores




# <--------- Starting the Work Flow ---------> 

# Displays Form only if the form is not submitted
if not st.session_state.form_submitted:
    with st.form("my_form"):

        # Taking input a Resume (PDF) file 
        resume_file = st.file_uploader(label="Upload your Resume/CV in PDF format", type="pdf")

        # Taking input Job Description
        st.session_state.job_desc = st.text_area("Enter the Job Description of the role you are applying for:",placeholder="Job Description...")

        # Form Submission Button
        submitted = st.form_submit_button("Analyze")
        if submitted:

            # Allow only if Both Resume and Job Description are Submitted
            if st.session_state.job_desc and resume_file:
                st.info("Extracting Information")

                st.session_state.resume = extract_pdf_text(resume_file)        # Calling the function to extract text from Resume

                st.session_state.form_submitted = True
                # st.rerun()  <--- REMOVED: This is redundant inside st.form_submit_button and caused the warning.

            # Donot allow if not uploaded
            else:
                st.warning("Please Upload both Resume and Job Description to analyze")


if st.session_state.form_submitted:
    score_place = st.info("Generating Scores...")

    # Call the function to get ATS Score
    # Add a check to prevent crashing if extraction failed
    if st.session_state.resume and st.session_state.job_desc and st.session_state.resume != "Could not extract text from the PDF file.":
        ats_score = calculate_similarity_bert(st.session_state.resume,st.session_state.job_desc)
    else:
        ats_score = 0.0 # Default if there's no valid input

    col1,col2 = st.columns(2,border=True)
    with col1:
        st.write("Few ATS uses this score to shortlist candidates, Similarity Score:")
        st.subheader(f"{ats_score:.2f}")

    # Call the function to get the Analysis Report from LLM (Groq)
    report = get_report(st.session_state.resume,st.session_state.job_desc)

    # Calculate the Average Score from the LLM Report
    report_scores = extract_scores(report)                 # Example : [3/5, 4/5, 5/5,...]
    # Calculate average score (sum of scores / total possible score (5 * number of points))
    if report_scores:
        avg_score = sum(report_scores) / (5 * len(report_scores))
    else:
        avg_score = 0.0


    with col2:
        st.write("Total Average score according to our AI report:")
        st.subheader(f"{avg_score:.2f}")
    score_place.success("Scores generated successfully!")


    st.subheader("AI Generated Analysis Report:")

    # Displaying Report with dynamic theme colors via custom class
    st.markdown(f"""
                <div class='report-container'>
                    {report}
                </div>
                """, unsafe_allow_html=True)
    
    # Download Button
    st.download_button(
        label="Download Report",
        data=report,
        file_name="report.txt",
        icon=":material/download:",
        )
    

# <-------------- End of the Work Flow --------------->
