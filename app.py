import streamlit as st
import pandas as pd
import numpy as np
import smtplib
import os
from email.message import EmailMessage
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

# --- Load Environment Variables (for local .env file) ---
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Topsis Web Service",
    page_icon="📊",
    layout="centered"
)

# --- Helper Functions (Same as before) ---
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
    msg['Subject'] = 'Your TOPSIS Result is Ready'
    msg['From'] = sender_email
    msg['To'] = user_email
    msg.set_content('Hello,\n\nPlease find attached the result file for your TOPSIS calculation.\n\nBest,\nDevansh')

    msg.add_attachment(result_csv_string.encode('utf-8'), 
                       maintype='text', 
                       subtype='csv', 
                       filename='result.csv')

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

    # TOPSIS Algorithm
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

# --- UI Layout (IMPROVED) ---

st.title("📊 TOPSIS Web Service")
st.markdown("""
This service calculates the **TOPSIS Score** and **Rank** for your dataset.
Upload your CSV file, define the criteria, and receive the results via email.
""")
st.divider()

# Step 1: Upload
st.subheader("1. Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Ensure the first column is the object name (e.g., M1, M2).")

# Step 2: Parameters (Side-by-Side for cleaner look)
st.subheader("2. Set Parameters")
col1, col2 = st.columns(2)

with col1:
    weights = st.text_input("Weights", placeholder="e.g., 1,1,1,2", help="Separate values with commas")
with col2:
    impacts = st.text_input("Impacts", placeholder="e.g., +,+,-,+", help="Use '+' for beneficial, '-' for non-beneficial")

# Step 3: Destination
st.subheader("3. Send Results")
email = st.text_input("Email ID", placeholder="example@thapar.edu", help="The results will be sent to this address")

# Submit Button (Centered logic visually)
st.write("") # Add a little space
if st.button("🚀 Calculate & Email Result", type="primary", use_container_width=True):
    if uploaded_file and weights and impacts and email:
        try:
            df = pd.read_csv(uploaded_file)
            
            with st.spinner("Running TOPSIS Algorithm..."):
                result_df = run_topsis(df, weights, impacts)
            
            with st.spinner("Sending Email..."):
                csv_string = result_df.to_csv(index=False)
                send_email(email, csv_string)
            
            st.success(f"✅ Success! Results have been sent to **{email}**")
            
            # Show Preview
            st.write("### Result Preview")
            st.dataframe(result_df.head(), use_container_width=True)

        except ValueError as ve:
            st.error(f"❌ Validation Error: {ve}")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
    else:
        st.warning("⚠️ Please fill in all fields before calculating.")