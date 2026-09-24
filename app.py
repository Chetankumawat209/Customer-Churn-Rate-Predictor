import pickle as pkl

import pandas as pd
import streamlit as st

# with st.sidebar as ss:
#     st.title("predict")

st.title ("Customer Churn Rate Model")

with st.form("User Input"):
    st.write("Fill all Details")
    col,col1,col2=st.columns(3)

    with col:
        gender = st.radio("Gender",["Male","Female"])
        senior_citizen = st.radio("SeniorCitizen",["Yes","No"])####
        partner = st.radio("Partner",["Yes","No"])
        dependents = st.radio("Dependents",["Yes","No"])
        phone_service = st.radio("Phone Service",["Yes","No"])
        multiple_lines = st.radio("MultipleLines",["Yes","No"])
       




    with col1:
        # stayed time in our company
        internet_service = st.radio("InternetService",["Fiber optic","DSL","No"])
        online_security = st.radio("OnlineSecurity",["Yes","No"])
        online_backup = st.radio("OnlineBackup",["Yes","No"])
        device_protection = st.radio("DeviceProtection",["Yes","No"])
        tech_support = st.radio("TechSupport",["Yes","No"])
        streaming_tv = st.radio("StreamingTV",["Yes","No"])
       
    with col2:
        streaming_movies = st.radio("StreamingMovies",["Yes","No"])
        paperless_billing = st.radio("PaperlessBilling",["Yes","No"])
        tenure = st.number_input("Enter tenure")
        contract = st.selectbox("Contract",["Month-to-month","One year",'Two year'])  # was "Month" - not a real category, model didn't recognize it
        payment_method = st.selectbox("PaymentMethod",['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
        'Credit card (automatic)'])
        monthly_charges = st.number_input("MonthlyCharges",placeholder=20)


    submitted = st.form_submit_button("submit",use_container_width=True)

# nothing below this point existed before - the form had no way to actually
# use what was typed in, so submit did nothing
if submitted:
    input_row = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,  # model was trained on 1/0, not Yes/No
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges
    }])

    saved = pkl.load(open("ChrunModel.pkl", "rb"))
    model = saved["Model"]

    proba = model.predict_proba(input_row)[0][1]
    prediction = model.predict(input_row)[0]

    st.write(f"### Prediction: {prediction}")
    st.write(f"Probability of churn: {proba:.2%}")
