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
st.set_page_config(page_title="Topsis Calculator", page_icon="📊")

# --- Helper Functions ---
def get_secret(key):
    """
    Tries to get a secret from os.environ (local .env), 
    and falls back to st.secrets (Streamlit Cloud).
    """
    value = os.getenv(key)
    if value is None:
        try:
            value = st.secrets[key]
        except (FileNotFoundError, KeyError):
            return None
    return value

def send_email(user_email, result_csv_string):
    # 1. Get Credentials safely
    sender_email = get_secret("MAIL_USERNAME")
    sender_password = get_secret("MAIL_PASSWORD")

    if not sender_email or not sender_password:
        raise ValueError("Email credentials not found. Check your .env file or Streamlit Secrets.")
    
    # 2. Construct Email
    msg = EmailMessage()
    msg['Subject'] = 'Your TOPSIS Result is Ready'
    msg['From'] = sender_email
    msg['To'] = user_email
    msg.set_content('Hello,\n\nPlease find attached the result file for your TOPSIS calculation.\n\nBest,\nDevansh')

    # 3. Attach CSV
    msg.add_attachment(result_csv_string.encode('utf-8'), 
                       maintype='text', 
                       subtype='csv', 
                       filename='result.csv')

    # 4. Send
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

def run_topsis(df, weights_str, impacts_str):
    # --- Step 1: Validation & Preprocessing (From your topsis.py) ---
    
    # Check 1: Number of Columns must be >= 3
    if df.shape[1] < 3:
        raise ValueError("Input file must contain 3 or more columns.")

    # Check 2: Weights Format
    try:
        weights = [float(w) for w in weights_str.split(',')]
    except ValueError:
        raise ValueError("Incorrect Weight Format. Correct format: '1,2,3'")

    # Check 3: Impacts Format
    impacts = impacts_str.split(',')
    if not all(i in ['+', '-'] for i in impacts):
        raise ValueError("Incorrect Impact Format. Correct format: '+,-,+'")

    # Check 4: Non-Numeric Values (Your Encoding Logic)
    # We work on a copy to avoid modifying the uploaded file in place before we are ready
    df_processed = df.copy()
    
    # Iterate over criteria columns (skipping the first one)
    for col in df_processed.columns[1:]:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            try:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                # We can log this to console or UI if needed
                # st.info(f"Encoded non-numeric column: {col}") 
            except Exception:
                raise ValueError(f"Column '{col}' contains non-numeric values that could not be encoded.")

    # Prepare data matrix
    data = df_processed.iloc[:, 1:].values.astype(float)

    # Check 5: Dimension Matching
    num_cols = data.shape[1]
    if len(weights) != num_cols:
        raise ValueError(f"Number of weights ({len(weights)}) does not match number of criteria columns ({num_cols}).")
    if len(impacts) != num_cols:
        raise ValueError(f"Number of impacts ({len(impacts)}) does not match number of criteria columns ({num_cols}).")

    # --- Step 2: Calculation (From your topsis.py) ---

    # 1. Normalization
    rss = np.sqrt(np.sum(data**2, axis=0))
    if (rss == 0).any():
        raise ValueError("One of the columns contains only 0's, Normalization cannot be performed.")
    
    norm_matrix = data / rss

    # 2. Weighted Normalized Matrix
    weighted_mat = norm_matrix * weights

    # 3. Ideal Best and Ideal Worst
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

    # 4. Euclidean Distance
    dist_best = np.sqrt(np.sum((weighted_mat - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_mat - ideal_worst)**2, axis=1))

    # 5. Topsis Score
    total_dist = dist_best + dist_worst
    # Handle division by zero safely
    score = np.divide(dist_worst, total_dist, out=np.zeros_like(dist_worst), where=total_dist!=0)
    score = np.round(score, 5)

    # --- Step 3: Result Formatting ---
    df['Topsis Score'] = score
    df['Rank'] = df['Topsis Score'].rank(ascending=False).astype(int)
    
    return df

# --- UI Layout ---
st.title("📊 TOPSIS Web Service")
st.markdown("Upload your data, set parameters, and get the results emailed to you.")

# Form Input
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
with col2:
    email = st.text_input("Email ID", placeholder="example@thapar.edu")

weights = st.text_input("Weights (comma separated)", placeholder="1,1,1,1")
impacts = st.text_input("Impacts (comma separated)", placeholder="+,+,-,+")

# Submit Button
if st.button("Calculate & Email"):
    if uploaded_file and weights and impacts and email:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Calculate
            result_df = run_topsis(df, weights, impacts)
            
            # Convert to CSV string
            csv_string = result_df.to_csv(index=False)
            
            # Send Email
            with st.spinner("Sending Email..."):
                send_email(email, csv_string)
            
            st.success(f"Success! Result sent to {email}")
            st.dataframe(result_df)  # Show preview

        except ValueError as ve:
            st.error(f"Validation Error: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please fill in all fields.")