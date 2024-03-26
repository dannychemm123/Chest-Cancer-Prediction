import streamlit as st
from cnnClassifier.pipeline.prediction import PredictionPipeline
import os
# Load your trained model (ensure PredictionPipeline is properly implemented to load the model)
model_path = os.path.join("model", "model.h5")
classifier = PredictionPipeline(model_path)


# Set the title and introduction of the app
st.title('Chest Cancer Classification App')
st.write("This app classifies chest CT scan images into Normal and Adenocarcinoma cancer. Upload an image to see its classification.")

# Organize the layout: sidebar for additional options or information
with st.sidebar:
    st.header("About the App")
    st.write('''This is an image classification app using a CNN model. This project uses Transfer Learning techniques to interpret chest CT scan pictures,
        with a focus on deep learning using Machine Learning. The goal is to predict a person's risk of developing lung cancer.
        The project uses a dataset of chest CT scans from Kaggle to help detect whether an individual has lung disease, 
        with the VGG16 model serving as the basis for training. This method suggests integrating cutting-edge neural network designs 
        to improve medical imaging diagnosis accuracy.''')
    # You can add more information or options here

# Main section of the app
st.header("Image Upload and Classification")

# File uploader allows the user to upload images
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button('Predict'):
        with st.spinner('Classifying...'):
            # Perform prediction
            prediction = classifier.predict(uploaded_file)
        
        st.success('Done!')
        st.write(f'Prediction: {prediction}')

# Optionally, use an expander to hide detailed information that is not always necessary
with st.expander("See explanation"):
    st.write("""
             If the prediction is Normal then it means that you do not have cancer but when it predict Adenocarcianoma cancer you have cancer and medical attention 
             is required to treat it accordingly.
        """)
footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #888;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        <p>Copyright © 2024 DotPy AI. All rights reserved.</p>
    </div>
    """
st.markdown(footer, unsafe_allow_html=True)