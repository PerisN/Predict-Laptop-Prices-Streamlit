import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load the model 
model = joblib.load('model.pkl')

# Define the mapping dictionaries
dict_ram_type = {'DDR4':0, 'DDR5':1, 'LPDDR4':2, 'Unified Memory':3, 'LPDDR4X':4, 'LPDDR5':5, 'LPDDR3':6}
dict_processor = {'AMD Athlon Dual Core':0, 'AMD Ryzen 3':1, 'AMD Ryzen 3 Dual Core':2, 'AMD Ryzen 3 Quad Core':3, 'AMD Ryzen 3 Hexa Core':4, 'AMD Ryzen 5':5, 'AMD Ryzen 5 Dual Core':6, 'AMD Ryzen 5 Quad Core':7, 'AMD Ryzen 5 Hexa Core':8, 'AMD Ryzen 7 Octa Core':9, 'AMD Ryzen 7 Quad Core':10, 'AMD Ryzen 9 Octa Core':11, 'Intel Core i3':12, 'Intel Core i5':13, 'Intel Evo Core i5':14, 'Intel Core i7':15, 'Intel Core i9':16, 'Intel Celeron Dual Core':17, 'Intel Celeron Quad Core':18, 'Intel Pentium Silver':19, 'Intel Pentium Quad Core':20, 'M1':21, 'M1 Pro':22, 'M1 Max':23, 'M2':24, 'Qualcomm Snapdragon 7c Gen 2':25}
dict_os = {'Windows':0, 'Mac OS':1, 'DOS':3, 'Chrome':4}
dict_brand = {'ASUS':0, 'Lenovo':1, 'HP':3, 'DELL':4, 'acer':5, 'MSI':6, 'APPLE':7, 'Infinix':8, 'realme':9, 'RedmiBook':10, 'SAMSUNG':11, 'Ultimus':12, 'ALIENWARE':13, 'Nokia':14, 'GIGABYTE':15, 'Vaio':16}

# Create the user interface
st.title('Laptop Price Predictor')

# Create the input widgets for the laptop features
ram_size = st.slider('Select RAM size (GB)', 2, 32, 8, 2)
ram_type = st.selectbox('Select RAM type', list(dict_ram_type.keys()))
processor = st.selectbox('Select Processor', list(dict_processor.keys()))
os = st.selectbox('Select Operating System', list(dict_os.keys()))
storage_size = st.slider('Select Storage size (GB)', 128, 2000, 512, 128)
display_size = st.slider('Select Display size (inches)', 10, 30, 15, 1)
brand = st.selectbox('Select Brand', list(dict_brand.keys()))
