from pickle import load
import numpy as np
import pandas as pd
import streamlit as st
import xgboost

df = pd.read_csv('Data\cleaned_data2.csv')

# Loading pretrained models from the pickle files
dicts = load(open('models\dicts_cols.pkl', 'rb'))
scaler = load(open('models\scaler.pkl', 'rb'))
xgbr = load(open(r'models\xgb_model.pkl', 'rb'))

# Define Streamlit app
def main():
    st.title('Laptop Price Predictor')
    
    # Create inputs for user to enter laptop specifications
    ram_size = st.slider('RAM Size (in GB)', 4, 32, step=4)
    ram_type = st.selectbox('RAM Type', list(dicts['RAM_Type'].keys()))
    display_size = st.slider('Display Size (in inches)', 10.00, 20.00, step=0.1, format="%.2f")
    processor = st.selectbox('Processor', list(dicts['Processor'].keys()))
    storage = st.selectbox('Storage', list(dicts['Storage'].keys()))
    os = st.selectbox('Operating System', list(dicts['OS'].keys()))
    brand = st.selectbox('Brand', list(dicts['Brand'].keys()))
    
    # Create input data as a DataFrame
    input_data = pd.DataFrame({
        'RAM_Size': [ram_size],
        'RAM_Type': [dicts['RAM_Type'][ram_type]],
        'Display': [display_size],
        'Processor': [dicts['Processor'][processor]],
        'Storage': [dicts['Storage'][storage]],
        'OS': [dicts['OS'][os]],
        'Brand': [dicts['Brand'][brand]]
    })                          
    
    # Define function to preprocess input data
    def preprocess_input_data(input_data):
        num_cols = ['RAM_Size', 'Display'] 
        # Rescale and Reshape the input data to have 2D shape
        input_data[num_cols] = scaler.transform(input_data[num_cols].values.reshape(1, -1))  
        return input_data

    
    # Preprocess the input data
    input_data = preprocess_input_data(input_data)
    
    # Make prediction
    predicted_price = xgbr.predict(input_data)[0]
    predicted_price1 = np.exp(predicted_price)

    # Display the predicted price
    st.write('Predicted Price: $', predicted_price1)

if __name__ == '__main__':
    main()

