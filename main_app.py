import streamlit as st
import pickle
import pandas as pd
import json

st.sidebar.title('Transaction Information')

# Header for main page
html_temp = """
<div style="background-color:Blue;padding:10px">
<h2 style="color:white;text-align:center;">KENAM</h2>
<h4 style="color:black;text-align:center;">Fraud Prevention for a Safer Fintech Future</h4>
</div><br>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: Black;'>Select Your Model</h1>", unsafe_allow_html=True)

# Model selection
selection = st.selectbox("", ["XGBClassifier"])

# Load the selected model
if selection == "XGBClassifier":
    st.write("You selected", selection, "model")
    model = pickle.load(open('xgb_model_2.pkl', 'rb'))

# File uploader for JSON-formatted .txt file
uploaded_file = st.sidebar.file_uploader("Drag and drop a .txt file with transaction details in JSON format", type="txt")

# Initialize a variable to store the DataFrame
user_inputs_df = None

if uploaded_file is not None:
    # Read and parse the uploaded JSON file
    try:
        file_content = uploaded_file.read().decode("utf-8")
        user_inputs = json.loads(file_content)  # Convert JSON string to dictionary

        # Ensure the user_inputs is a dictionary and convert it to a DataFrame
        if isinstance(user_inputs, dict):
            user_inputs_df = pd.DataFrame([user_inputs])

            # Define the expected columns
            expected_columns = [
                "account_id", "receiver_account_id", "transaction_amount", "account_age_days", 
                "daily_transaction_amount", "total_daily_transactions", "transaction_frequency",
                "transaction_frequency_same_account", 
                "account_type_personal", "payment_type_debit", 
                "transaction_type_bank_transfer", "transaction_type_Deposit", "transaction_type_sporty"
            ]

            # Check if all expected columns are present
            missing_cols = set(expected_columns) - set(user_inputs_df.columns)
            if missing_cols:
                st.error(f"Missing columns in input data: {', '.join(missing_cols)}")
            else:
                # Convert Boolean-like fields to numeric
                bool_columns = ['account_type_personal', 'payment_type_debit', 
                                'transaction_type_bank_transfer', 'transaction_type_sporty', 'transaction_type_Deposit']
                
                for col in bool_columns:
                    if col in user_inputs_df:
                        user_inputs_df[col] = user_inputs_df[col].map({'True': 1, 'False': 0}).astype(int)

                # Convert other object columns to numeric where needed
                for column in user_inputs_df.select_dtypes(include=['object']).columns:
                    user_inputs_df[column] = pd.to_numeric(user_inputs_df[column], errors='coerce')

                # Display the DataFrame for user verification
                st.write("Transaction Information:", user_inputs_df)
        else:
            st.sidebar.error("Uploaded file format is incorrect. Please provide a JSON dictionary.")
    except json.JSONDecodeError:
        st.sidebar.error("Failed to parse the file. Ensure it is valid JSON format.")

# Prediction section
if user_inputs_df is not None and user_inputs_df.shape[1] == len(expected_columns):
    st.markdown("<h1 style='text-align: center; color: Black;'>Transaction Information</h1>", unsafe_allow_html=True)
    st.table(user_inputs_df)

    # Prediction button
    st.subheader('Click PREDICT if configuration is OK')
    if st.button('PREDICT'):
        prediction = model.predict(user_inputs_df)

        # Display the prediction result
        if prediction[0] == 0:
            st.success('Transaction is SAFE :)')
        else:
            st.warning('ALARM! Transaction is SUSPICIOUS :(')
else:
    st.info("Awaiting valid transaction data in the correct format.")
