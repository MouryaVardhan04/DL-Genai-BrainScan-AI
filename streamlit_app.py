import streamlit as st

st.title('Brain Tumor Prediction')
st.write('Upload an MRI image to predict brain tumor presence.')

uploaded_file = st.file_uploader('Choose an MRI image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded MRI.', use_column_width=True)
    st.write('Prediction functionality coming soon!') 