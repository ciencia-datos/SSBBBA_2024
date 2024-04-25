import datetime
import os
import pickle
import random


import pandas as pd
import requests
import streamlit as st

# Get current date and time
now_ = datetime.datetime.now()

# Custom functions.
my_classifier_url = r'https://github.com/ciencia-datos/SSBBBA_2024/blob/main/pii_phi_classifier_tidy.pkl?raw=true'
my_text_url = r'https://github.com/ciencia-datos/SSBBBA_2024/blob/main/pii_phi_textvectorizer_tidy.pkl'

my_cwd = os.getcwd()

xgb_model_file = os.path.join(my_cwd,'pii_phi_classifier_tidy.pkl')
text_vect_file = os.path.join(my_cwd,'pii_phi_textvectorizer_tidy.pkl')

my_model_resp = requests.get(my_classifier_url)
my_text_resp = requests.get(my_text_url)

with open('sms_email_tfidf_vect_xgb.pkl', 'wb') as fopen:
        fopen.write(my_text_resp.content)

with open('sms_email_classifier_xgb.pkl', 'wb') as fopen:
        fopen.write(my_model_resp.content)

with open(xgb_model_file, 'rb') as file:
	XGB_model_classifier = pickle.load(file)
 
with open(text_vect_file, 'rb') as file:
	text_tfidf_vectorizer = pickle.load(file) 

def model_predictor(text):
	txt_vec = text_tfidf_vectorizer.transform(text)
	txt_class = int(XGB_model_classifier.predict(txt_vec)[0])
	return txt_class

st.title("SixSigma-PII/PHI Classification!")

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

tab1, tab2 = st.tabs(["Bulk", "Single"])

with tab1:
    uploaded_file = st.file_uploader("Upload a file with header text to be classified.Input field name should be 'FIELD' ")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.header("Raw PII/PHI Data Elements.")
        st.write(df)
        st.header("Model Classified PII/PHI Data Elements.")
        if 'FIELD' in df.columns:
            df['FIELD'] = df['FIELD'].astype('U')
            df['predicted_class']=df['FIELD'].apply(lambda x:model_predictor([x]))
            df['predicted_class']= df['predicted_class'].replace({1:'PII/PHI',0:'NO-PII/PHI'})
            st.table(df)
            csv_exp = convert_df(df)
            st.download_button(
            "Click to Download",
            csv_exp,
            f"classified_pii_phi_{now_}.csv",
            "text/csv",
            key="download-csv",
            )
        else:
            st.error('Please Make sure to give a column name as (FIELD) to input column.')
        
with tab2:
    with st.form("Message classification"):
        txt_msg = st.text_input(
            "Enter a Header text to check it is a PII/PHI or NO PII/PHI?"
        )
        submitted = st.form_submit_button("Submit")
        if submitted and len(txt_msg) > 1:
            msg_cls = model_predictor([txt_msg])
            if msg_cls == 1:
                st.markdown(
                    f'<h2 style="color:#31a354;font-size:18px;">{"PII/PHI !!!"}</h2>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<h2 style="color:#de2d26;font-size:18px;">{"NO-PII/PHI !!!"}</h2>',
                    unsafe_allow_html=True,
                )
        elif submitted and len(txt_msg) == 1:
            st.error("Please enter a word should have atleast 2 letters.")
        elif submitted and len(txt_msg) == 0:
            st.error("Nothing is enter. and please do enter one.")
        else:
            pass
