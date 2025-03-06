import streamlit as st 
st.title("Churn Prediction System")
import pandas as pd
import pickle as pkl
with open("customer_churn_model.pkl","rb") as f:
    model = pkl.load(f)
with open("encoders.pkl","rb") as encoders:
    encoders= pkl.load(encoders)
    
col1,col2,col3=st.columns(3)
with col1:
    gender=st.selectbox("gender",['Male','Female'])
    Senior_Citizen=st.selectbox("Senior Citizen",[0,1])
    Partner=st.selectbox("Partner",["Yes","No"])
    Dependent=st.selectbox("Dependent",[0,1])
    Tenure=st.number_input("Tenure",min_value=0,max_value=100,value=1)
    
with col2:
    Phone_Service= st.selectbox("Phone Service",['Yes','No'])
    multiple_Lines=st.selectbox("multiple_Lines",[0,1])
    Internet_service=st.selectbox("Internet Service",["Yes","No"])
    Online_security=st.selectbox("Online Security",["Yes","No"])
    Online_backup=st.selectbox("Online Backup",["Yes","No"])
    
with col3:
    Device_Protection=st.selectbox("Device Protection",['Yes','No'])
    Tech_Support=st.selectbox("Tech Support",["Yes","No"])
    Streaming_TV=st.selectbox("Streaming TV",["Yes","No"])
    Streaming_Movies=st.selectbox("Streaming Movies",["Yes","No"])
    Contract=st.selectbox("Contract",['Month-to-Month','One Year','Two Year'])
    Paperless_billing=st.selectbox("Paperless Billing",["Yes","No"])
    payment_method=st.selectbox("Payment_method",["Bank Withdrawal","Credit card","Mail Check"])
    monthly_charges=st.number_input("monthly charges",min_value=0.0,value=29.85)
    total_charges=st.number_input("total charges")
    

with open ("customer_churn_model.pkl",'rb') as f:
    model_data=pkl.load(f)
    
    loaded_model = model_data['model']
    features_names= model_data["features_names"]
    

if st.button("predict"):
    input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 0, # For some reason putting NO is not labelencoding it to 0 
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No',
    'InternetService': 'Yes',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-Month', # it was Month-to-month it should be Montth-to-Month
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Bank Withdrawal', # Electronic shock does not exist its Bank Withdrawal
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85,
    'SeniorCtizen': 0,
    
}
# Convert input data to DataFrame
    input_data_df = pd.DataFrame([input_data])

# Encode categorical features using the saved encoders
    for column in input_data_df.columns:
        if column in encoders:
            input_data_df[column] = encoders[column].transform(input_data_df[column])

    model=model['model']

    input_data_df = input_data_df[features_names]  # Reorder to match training features

    # Make prediction
    prediction = model.predict(input_data_df)
    pred_prob = model.predict_proba(input_data_df)

    # Results
    st.write(f"### Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    print(f"Prediction Probability: {pred_prob}")
