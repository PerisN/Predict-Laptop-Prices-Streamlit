from pickle import load
import numpy as np
import pandas as pd
import streamlit as st

df = pd.read_csv('Data\cleaned_data2.csv')

# Loading pretrained models from the pickle files
dicts = load(open('models\dicts.pkl', 'rb'))
scaler = load(open('models\scaler.pkl', 'rb'))
encoder = load(open('models\encoder.pkl', 'rb'))
gbr = load(open('models\gbr_model.pkl', 'rb'))
xgbr = load(open('models\xgbr_model.pkl', 'rb'))

# Define function to preprocess input data
def preprocess_input_data(df):
    # Apply same preprocessing steps as in training data
    input_data = df.replace(dicts)
    input_data['RAM_Type'] = input_data['RAM_Type'].astype('category')
    input_data['Processor'] = input_data['Processor'].astype('category')
    input_data['OS'] = input_data['OS'].astype('category')
    input_data['Brand'] = input_data['Brand'].astype('category')
    encoder.fit(input_data['Storage'])
    input_data['Storage'] = encoder.transform(input_data['Storage']) 
    scaler.fit(df[['RAM_Size', 'Display']])
    input_data[['RAM_Size', 'Display']] = scaler.transform(df[['RAM_Size', 'Display']])
    return input_data

# Define Streamlit app
def main():
    st.title('Laptop Price Predictor')
    
    # Create inputs for user to enter laptop specifications
    ram_size = st.slider('RAM Size (in GB)', 4, 32, step=4)
    ram_type = st.selectbox('RAM Type', list(dict_ram_type.keys()))
    display_size = st.slider('Display Size (in inches)', 10.00, 20.00, step=0.5, format="%.2f")
    processor = st.selectbox('Processor', list(dict_processor.keys()))
    storage = st.selectbox('Storage', list(df.Storage.unique()))
    os = st.selectbox('Operating System', list(dict_os.keys()))
    brand = st.selectbox('Brand', list(dict_brand.keys()))
    
    # Create input data as a DataFrame
    input_data = pd.DataFrame({
        'RAM_Size': [ram_size],
        'RAM_Type': [dict_ram_type[ram_type]],
        'Display': [display_size],
        'Processor': [dict_processor[processor]],
        'Storage': [storage],
        'OS': [dict_os[os]],
        'Brand': [dict_brand[brand]]
    })
    
    # Preprocess the input data
    input_data = preprocess_input_data(input_data)
    
    # Make prediction
    predicted_price = gbr.predict(input_data)[0]
    predicted_price1 = np.exp(predicted_price)

    # Display the predicted price
    st.write('Predicted Price: $', predicted_price1)

if __name__ == '__main__':
    main()

