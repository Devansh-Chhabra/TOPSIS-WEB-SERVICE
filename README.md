# 🌐 TOPSIS Web Service

![Streamlit Demo](https://img.shields.io/badge/Streamlit-Deployed-blueviolet?style=flat-square) 
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square) 
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**TOPSIS Web Service** is a user-friendly web application that calculates the **Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)** score for a given dataset.

Users can upload a CSV file, specify weights and impacts, and receive the detailed results directly via **email**.

🔗 *Live Demo:* [Click here to view](https://topsis-web-service.streamlit.app/)

🔗 *My PyPi Package:* [Click here to view](https://github.com/Devansh-Chhabra/TOPSIS-PyPi-Package)

---

## ✨ Features

- **📂 CSV File Upload:** Easy drag-and-drop interface for data input.
- **📩 Email Integration:** Automatically sends the result CSV file to the user's provided email address.
- **🧠 Intelligent Encoding:** Automatically detects categorical values (e.g., "Low", "Medium") and converts them to numeric values using Label Encoding.
- **✅ Robust Validation:** Checks for file format, column mismatch, and data integrity before processing.
- **🔒 Secure:** Uses environment variables to protect sensitive email credentials.

---

## 🛠️ Local Installation

Follow these steps to run the application on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Devansh-Chhabra/TOPSIS-WEB-SERVICE
cd TOPSIS-WEB-SERVICE
```

### 2. Install Dependencies

Make sure you have Python installed. Then run:

```bash
pip install -r requirements.txt
```
### 3. Setup Environment Variables

To enable the email feature, you must create a `.env` file in the root directory.

Create a file named `.env` and add the following:

```ini
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_16_char_app_password
```

**Note:** For Gmail, you must use an App Password, not your regular login password. [Learn how to create one here](https://support.google.com/accounts/answer/185833).

### 4. Run the App

Launch the Streamlit server:

```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 🚀 Usage Guide

1. **Enter Email:** Provide the email address where you want to receive the result.
2. **Upload Data:** Upload a `.csv` file containing your dataset.
   - **Constraint:** The file must have at least 3 columns.
3. **Set Weights:** Enter numerical weights separated by commas (e.g., `0.25,0.25,0.25,0.25`).
4. **Set Impacts:** Enter signs (`+` or `-`) separated by commas (e.g., `+,+,-,+`).
5. **Submit:** Click the "Submit" button.

The application will process the data, calculate the TOPSIS score and Rank, and email the output file to you immediately.

---

## 📦 Dependencies

- `streamlit`: For the web interface.
- `pandas` & `numpy`: For data manipulation and calculation.
- `scikit-learn`: For handling categorical data (LabelEncoder).
- `python-dotenv`: For managing secure credentials.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Devansh Chhabra**  
📧 Email: [devanshchhabr@gmail.com](mailto:devanshchhabr@gmail.com)  

---
