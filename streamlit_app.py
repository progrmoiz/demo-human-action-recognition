import streamlit as st
import pickle
from skimage.transform import resize
from skimage.io import imread
import numpy as np

from PIL import Image

def run_model(model_path, uploaded_file):
    dtc_model = pickle.load(open(model_path, "rb"))
                
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        new_image = image.resize((80, 80))
        st.image(new_image)

        img = np.array(new_image)

        nx, ny, nrgb = img.shape
        img_reshape = img.reshape((1,nx*ny*nrgb))

        y_pred_dtc = dtc_model.predict(img_reshape)

    st.header("Model Output")
    st.text(y_pred_dtc[0])

def main():

    st.sidebar.title("Control Panel")

    display = ("Decision Tree Classifier", "Stochastic Gradient Descent", "K-Nearest Neighbour")

    options = list(range(len(display)))

    ml_algos_titles = lambda x: display[x]

    with st.sidebar:

        # Used for looking some model outputs
        with st.form('Form'):
            st.header("Run Model")
            
            uploaded_file = st.file_uploader(label="Image Uploader", type=["png","jpg","jpeg"])
            ml_algos_option = st.selectbox('Select an algorithm.', options, format_func=ml_algos_titles, key='algorithm_option')

            submitted = st.form_submit_button('Submit')

    st.title("Human Action Recognition Dashboard")

    if submitted:

        # models loaded in the streamlit app, which can be used to perform prediction on demand and showing the inference results
        if ml_algos_option == 0:
            
            run_model("dtc_classifier.pkl", uploaded_file)
          
        if ml_algos_option == 1:
            run_model("sgd_classifier.pkl", uploaded_file)

          
        if ml_algos_option == 2:
            run_model("knn_classifier.pkl", uploaded_file)

if __name__ == '__main__':
    main()