import streamlit as st
import pickle
import pandas as pd
from PIL import Image
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder

image = Image.open("Autoscout_logo.png")
st.image(image, width=450)

st.sidebar.title('Car Price Prediction')

make_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Espace'))
km=st.sidebar.slider("What is the km of your car", 0,320000, step=500)
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
age=st.sidebar.selectbox("What is the age of your car:",(0,1,2,3))
hp=st.sidebar.slider("What is the hp_kw of your car?", 40, 300, step=5)
displacement=st.sidebar.slider("What is the displacement(cc) of your car?", 890, 2967, step=10)


final_model = pickle.load(open('final_RF_model', 'rb'))
final_model_transformer = pickle.load(open('transformer', 'rb'))



my_dict = {
    "age": age,
    "hp_kW": hp,
    "km": km,
    "Gearing_Type":gearing_type,
    "make_model": make_model,
    'Displacement_cc' :displacement 
}

df = pd.DataFrame.from_dict([my_dict])


st.header("Selected features:")
st.table(df)


df2 = final_model_transformer.transform(df)

image2 = Image.open("middle_image.png")
st.image(image2, width=300)

st.subheader("If your selections are complete, click the predict button.")

if st.button("Predict"):
    prediction = final_model.predict(df2)
    
        
try:
    st.success(f"{make_model} Prediction Price is {round(int(prediction[0]),2)}€")
    if make_model == "Audi A3":
        st.image(Image.open("Cars/Audi_A3.jpg"))
    elif make_model == "Audi A1":
        st.image(Image.open("Cars/Audi_A1.jpg"))
    elif make_model == "Opel Insignia":
        st.image(Image.open("Cars/Opel_Insignia.jpg"))
    elif make_model == "Opel Astra":
        st.image(Image.open("Cars/Opel_Astra.jpg"))
    elif make_model == "Opel Corsa":
        st.image(Image.open("Cars/Opel_Corsa.jpg"))
    elif make_model == "Renault Clio":
        st.image(Image.open("Cars/Renault_Clio.jpg"))
    elif make_model == "Renault Espace":
        st.image(Image.open("Cars/Renault_Espace.jpg"))
except NameError:
    st.write("Please **Predict** button to display the result!")
    

    
