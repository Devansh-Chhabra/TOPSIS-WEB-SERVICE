import streamlit as st
import pandas as pd
import numpy as np
import smtplib
import os
from email.message import EmailMessage
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
import time

# --- Load Environment Variables ---
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Topsis Master",
    page_icon="⚡",
    layout="centered"
)

# --- 🎨 Custom CSS & Animations ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* General Body Styles */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        font-family: 'Poppins', sans-serif;
        color: white;
    }
    
    /* Text Color Override */
    h1, h2, h3, h4, h5, p, div {
        color: #ffffff !important;
    }

    /* Input Fields Styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 10px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00d2ff;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.3);
    }

    /* File Uploader Styling */
    section[data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px dashed rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 30px;
        font-size: 18px;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 5px 15px rgba(0, 210, 255, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 210, 255, 0.6);
        background: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%);
    }

    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }

    /* Card-like container logic (optional visual tweak) */
    div.block-container {
        padding-top: 2rem;
    }

    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .animate-enter {
        animation: fadeIn 0.8s ease-out forwards;
    }

</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_secret(key):
    value = os.getenv(key)
    if value is None:
        try:
            value = st.secrets[key]
        except (FileNotFoundError, KeyError):
            return None
    return value

def send_email(user_email, result_csv_string):
    sender_email = get_secret("MAIL_USERNAME")
    sender_password = get_secret("MAIL_PASSWORD")

    if not sender_email or not sender_password:
        raise ValueError("Email credentials not found. Check your .env file or Streamlit Secrets.")
    
    msg = EmailMessage()
    msg['Subject'] = 'Your TOPSIS Analysis Result'
    msg['From'] = sender_email
    msg['To'] = user_email
    msg.set_content('Hello,\n\nSuccess! Attached is the ranked result file for your TOPSIS calculation.\n\nBest Regards,\nTOPSIS Service')

    msg.add_attachment(result_csv_string.encode('utf-8'), 
                       maintype='text', 
                       subtype='csv', 
                       filename='topsis_result.csv')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

def run_topsis(df, weights_str, impacts_str):
    if df.shape[1] < 3:
        raise ValueError("Input file must contain 3 or more columns.")

    try:
        weights = [float(w) for w in weights_str.split(',')]
    except ValueError:
        raise ValueError("Incorrect Weight Format. Correct format: '1,2,3'")

    impacts = impacts_str.split(',')
    if not all(i in ['+', '-'] for i in impacts):
        raise ValueError("Incorrect Impact Format. Correct format: '+,-,+'")

    df_processed = df.copy()
    
    # Preprocessing
    for col in df_processed.columns[1:]:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            try:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
            except Exception:
                raise ValueError(f"Column '{col}' contains non-numeric values that could not be encoded.")

    data = df_processed.iloc[:, 1:].values.astype(float)
    num_cols = data.shape[1]
    
    if len(weights) != num_cols:
        raise ValueError(f"Number of weights ({len(weights)}) does not match number of criteria columns ({num_cols}).")
    if len(impacts) != num_cols:
        raise ValueError(f"Number of impacts ({len(impacts)}) does not match number of criteria columns ({num_cols}).")

    # TOPSIS Core Logic
    rss = np.sqrt(np.sum(data**2, axis=0))
    if (rss == 0).any():
        raise ValueError("One of the columns contains only 0's, Normalization cannot be performed.")
    
    norm_matrix = data / rss
    weighted_mat = norm_matrix * weights

    ideal_best = []
    ideal_worst = []
    for i in range(len(impacts)):
        if impacts[i] == '+':
            ideal_best.append(np.max(weighted_mat[:, i]))
            ideal_worst.append(np.min(weighted_mat[:, i]))
        else:
            ideal_best.append(np.min(weighted_mat[:, i]))
            ideal_worst.append(np.max(weighted_mat[:, i]))
            
    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    dist_best = np.sqrt(np.sum((weighted_mat - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_mat - ideal_worst)**2, axis=1))

    total_dist = dist_best + dist_worst
    score = np.divide(dist_worst, total_dist, out=np.zeros_like(dist_worst), where=total_dist!=0)
    score = np.round(score, 5)

    df['Topsis Score'] = score
    df['Rank'] = df['Topsis Score'].rank(ascending=False).astype(int)
    
    return df

# --- 🚀 UI Layout ---

# Animated Header
st.markdown("""
<div class="animate-enter" style="text-align: center; margin-bottom: 30px;">
    <h1 style="font-size: 50px; font-weight: 700; background: -webkit-linear-gradient(#00d2ff, #3a7bd5); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TOPSIS Master</h1>
    <p style="font-size: 18px; color: #b0c4de;">The Ultimate Decision Support System</p>
</div>
""", unsafe_allow_html=True)

# Container for the form
with st.container():
    # Step 1
    st.markdown('<h3 style="color: #00d2ff;">📂 1. Upload Data</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="csv", help="Ensure the first column is the object name")
    
    st.write("") # Spacer

    # Step 2
    st.markdown('<h3 style="color: #00d2ff;">⚙️ 2. Configure Parameters</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        weights = st.text_input("Weights (comma separated)", placeholder="e.g., 1, 1, 1, 2")
    with col2:
        impacts = st.text_input("Impacts (+ or -)", placeholder="e.g., +, +, -, +")

    st.write("") # Spacer

    # Step 3
    st.markdown('<h3 style="color: #00d2ff;">📧 3. Recipient</h3>', unsafe_allow_html=True)
    email = st.text_input("Email Address", placeholder="name@example.com")

    st.divider()

    # Submit Button
    if st.button("🚀 Run Analysis & Email"):
        if uploaded_file and weights and impacts and email:
            try:
                # Progress Bar Animation
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                
                my_bar.empty()

                df = pd.read_csv(uploaded_file)
                
                # Calculation
                result_df = run_topsis(df, weights, impacts)
                
                # Email
                csv_string = result_df.to_csv(index=False)
                send_email(email, csv_string)
                
                # Success Message
                st.balloons()
                st.markdown(f"""
                <div style="background-color: rgba(0, 255, 127, 0.2); padding: 20px; border-radius: 10px; border: 1px solid #00ff7f; text-align: center;">
                    <h3 style="color: #00ff7f; margin:0;">✅ Results Sent Successfully!</h3>
                    <p style="margin-top: 10px;">Check your inbox at <strong>{email}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Preview
                st.write("### 📊 Result Preview")
                st.dataframe(result_df.head(), use_container_width=True)

            except ValueError as ve:
                st.error(f"❌ Input Error: {ve}")
            except Exception as e:
                st.error(f"❌ System Error: {e}")
        else:
            st.warning("⚠️ Please fill in all fields to proceed.")