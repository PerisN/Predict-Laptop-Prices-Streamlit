import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import streamlit as st

df = pd.read_csv('Data\cleaned_data2.csv')

# Load the model 
model = joblib.load('model.pkl')

# Define the mapping dictionaries
dict_ram_type = {'DDR4':0, 'DDR5':1, 'LPDDR4':2, 'Unified Memory':3, 'LPDDR4X':4, 'LPDDR5':5, 'LPDDR3':6}
dict_processor = {'AMD Athlon Dual Core':0, 'AMD Ryzen 3':1, 'AMD Ryzen 3 Dual Core':2, 'AMD Ryzen 3 Quad Core':3, 'AMD Ryzen 3 Hexa Core':4, 'AMD Ryzen 5':5, 'AMD Ryzen 5 Dual Core':6, 'AMD Ryzen 5 Quad Core':7, 'AMD Ryzen 5 Hexa Core':8, 'AMD Ryzen 7 Octa Core':9, 'AMD Ryzen 7 Quad Core':10, 'AMD Ryzen 9 Octa Core':11, 'Intel Core i3':12, 'Intel Core i5':13, 'Intel Evo Core i5':14, 'Intel Core i7':15, 'Intel Core i9':16, 'Intel Celeron Dual Core':17, 'Intel Celeron Quad Core':18, 'Intel Pentium Silver':19, 'Intel Pentium Quad Core':20, 'M1':21, 'M1 Pro':22, 'M1 Max':23, 'M2':24, 'Qualcomm Snapdragon 7c Gen 2':25}
dict_os = {'Windows':0, 'Mac OS':1, 'DOS':3, 'Chrome':4}
dict_brand = {'ASUS':0, 'Lenovo':1, 'HP':3, 'DELL':4, 'acer':5, 'MSI':6, 'APPLE':7, 'Infinix':8, 'realme':9, 'RedmiBook':10, 'SAMSUNG':11, 'Ultimus':12, 'ALIENWARE':13, 'Nokia':14, 'GIGABYTE':15, 'Vaio':16}
dicts_cols = {'RAM_Type':dict_ram_type, 'Processor':dict_processor, 'OS':dict_os, 'Brand':dict_brand}

# Define function to preprocess input data
def preprocess_input_data(df):
    # Apply same preprocessing steps as in training data
    input_data = df.replace(dicts_cols)
    le = LabelEncoder()
    le.fit(df['Storage'])
    input_data['Storage'] = le.transform(df['Storage'])
    scaler = MinMaxScaler()
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
    predicted_price = model.predict(input_data)[0]
    predicted_price1 = np.exp(predicted_price)

    # Display the predicted price
    st.write('Predicted Price: $', predicted_price1)

if __name__ == '__main__':
    main()

