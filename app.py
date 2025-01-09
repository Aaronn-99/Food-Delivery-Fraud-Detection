import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import altair as alt

# App Title
st.title("Enhanced Fraud Detection for Food Delivery Refunds")

# Section 1: Data Upload
st.header("Upload Refund Request Data")
uploaded_file = st.file_uploader("Upload a CSV file with refund data", type=["csv"])
data = None

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())
else:
    st.warning("Please upload a dataset to proceed.")

# Sample dataset
if st.button("Load Sample Data"):
    data = pd.DataFrame({
        'Customer_ID': np.random.randint(1000, 1100, size=50),
        'Order_Value': np.random.uniform(10, 100, size=50),
        'Refund_Reason': np.random.choice(['Item Missing', 'Wrong Item', 'Quality Issue'], 50),
        'Refund_Amount': np.random.uniform(5, 100, size=50),
        'Fraud_Flag': np.random.choice([0, 1], size=50, p=[0.8, 0.2])
    })
    st.write("Sample Data:", data)

# Section 2: Train a Fraud Detection Model
if data is not None:
    st.header("Train Fraud Detection Model")

    # Preprocess data
    data['Refund_Reason'] = data['Refund_Reason'].astype('category').cat.codes  # Encode categorical data
    X = data[['Order_Value', 'Refund_Amount', 'Refund_Reason']]
    y = data['Fraud_Flag']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    st.write("Training and Optimizing the Model...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    
    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    
    st.subheader("Model Evaluation")
    st.write("Best Model Parameters:", grid_search.best_params_)
    st.write("Classification Report:")
    st.json(report)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    try:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(cm, 
                             index=["Not Fraud", "Fraud"], 
                             columns=["Predicted Not Fraud", "Predicted Fraud"])
        st.write(cm_df)
    except ValueError as e:
        st.error(f"Error generating confusion matrix: {e}")
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_prob)
    st.write(f"ROC-AUC Score: {roc_auc:.2f}")

    # Feature Importances
    st.subheader("Feature Importances")
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(feature_importances.set_index('Feature'))

else:
    st.info("Awaiting data upload or sample data to proceed.")

# Section 3: Predict Fraud
st.header("Predict Fraud for a Customer")

# Input form for new customer data
with st.form("fraud_prediction_form"):
    order_value = st.number_input("Enter Order Value ($):", min_value=0.0, step=0.1)
    refund_amount = st.number_input("Enter Refund Amount ($):", min_value=0.0, step=0.1)
    refund_reason = st.selectbox("Select Refund Reason:", ['Item Missing', 'Wrong Item', 'Quality Issue'])
    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    # Prepare input data for prediction
    refund_reason_encoded = {'Item Missing': 0, 'Wrong Item': 1, 'Quality Issue': 2}[refund_reason]
    X_new = pd.DataFrame({
        'Order_Value': [order_value],
        'Refund_Amount': [refund_amount],
        'Refund_Reason': [refund_reason_encoded]
    })
    
    # Make prediction
    prediction = model.predict(X_new)[0]  # Output will be 0 or 1
    prediction_prob = model.predict_proba(X_new)[0][1]  # Probability of fraud
    
    # Display result
    if prediction == 1:
        st.error(f"The customer is predicted to be fraudulent. (Fraud Probability: {prediction_prob:.2f})")
    else:
        st.success(f"The customer is predicted to be non-fraudulent. (Fraud Probability: {prediction_prob:.2f})")
