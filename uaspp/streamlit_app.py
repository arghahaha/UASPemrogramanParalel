import streamlit as st
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu

st.title("Colorize Your Picture, Colorize Your Memory")

st.subheader("Paralel Programming Final Exams")

selected3 = option_menu(None, ["Home", "Interest",  "Upload"], 
    icons=['house', 'list-task', "cloud-upload"], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
)

def display_home_tab():
    link = 'https://images.unsplash.com/photo-1558056524-97698af21ff8?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80'
    st.image(link, use_column_width=True)

    st.title("AMAZING! ~ Colorful Image Colorization")

    st.subheader("AMAZING! is a machine learning web application that has undergone training to reconstruct color images from their grayscale or black-and-white input equivalents, which users provide to the system.")

    st.write("The system operates based on Autoencoders, a concept pioneered by Richard Zhang in his Colorful Image Colorization paper. Autoencoders intelligently encode crucial details of a large image into a compact space and subsequently endeavor to reconstruct the image in color. The autoencoder is penalized when its performance is subpar, encouraging continuous refinement until it achieves satisfactory results.", unsafe_allow_html=True)

    st.write("Please be aware that the colorization process for this image might require some time to complete.")

def display_interest_tab():
    link2 = 'https://images.unsplash.com/photo-1558056524-97698af21ff8?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80'
    st.image(link2, use_column_width=True)

    st.title("AMAZING!: Transforming Monochrome Magic into Vivid Realities")

    st.write("The allure of transforming black and white images into vibrant, colorful masterpieces has captivated the imagination of artists, photographers, and enthusiasts alike. Our project, Chromatic Alchemy, delves into the realm of image conversion, leveraging the power of machine learning to breathe life and hues into grayscale compositions. Black and white images possess a timeless charm, evoking nostalgia and emphasizing the subtleties of light and shadow. However, the desire to witness these compositions in full color often sparks curiosity and creativity. Our project aims to satisfy this curiosity by employing advanced machine learning techniques, specifically Autoencoders, to unlock the hidden potential within monochromatic images.")
    
    st.write("The heart of our endeavor lies in the application of Autoencoders, a neural network architecture adept at encoding and decoding intricate details. Developed based on the groundbreaking work of Richard Zhang, our approach involves training the model to discern the nuances of grayscale images and subsequently generate vibrant, realistic colorizations. This transformative process is an intricate dance between preserving the essence of the original and introducing a rich spectrum of colors. By embarking on this journey, we strive not only to fulfill the aesthetic desire for colorization but also to uncover the latent information embedded within grayscale images. Autoencoders cleverly distill the essence of a large image into a compact space, encapsulating the vital features that define its character.") 
    
    st.write("The model is then refined through a process of iterative learning, penalizing discrepancies until it attains a nuanced understanding of color relationships. Our project acknowledges that this conversion is not merely a technical feat but an art form in itself. The interplay between algorithmic precision and creative expression results in a symphony of colors that transforms mundane black and white snapshots into captivating, dynamic visual experiences. Through Chromatic Alchemy, we invite users to explore the intersection of technology and art, witnessing the magic unfold as monochrome images undergo a metamorphosis into breathtaking, colorful realities.")

def display_upload_tab():
    st.subheader("Take a Wonderful Experience by Using Our Tools!")

    img_file_buffer = st.file_uploader("Upload Black & White Image", type=['png', 'jpg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)

    model_path = 'Model/recolor.h5'
    if img_file_buffer is not None:

        with st.spinner("Colorizing Image..."):
            model = load_model(model_path)

            image = Image.open(img_file_buffer)
            st.image(image, caption="Original Image", use_column_width=True)
            img_array = np.array(image)
            img = preProc([img_array])
            colorImg = recolor(model, img)
            st.image(colorImg, caption = "Colorized Image", use_column_width=True)

            # Assuming preProc and recolor functions are defined elsewhere
            # img = preProc([img_array])
            # colorImg = recolor(model, img)
            # st.image(colorImg, caption="Colorized Image", use_column_width=True)

        st.success("Successfully colorized the image!")

def load_model(path):
	model = keras.models.load_model(path)
	model.summary()
	return model

def preProc(A):
	A = np.array(A)
	normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
	B = normalization_layer(A)
	B = tf.image.resize(B, [256, 256])
	A = tf.image.rgb_to_hsv(B)
	return A[:,:,:,-1:]

def recolor(model, img_array):
	
	print(img_array.shape)
	predImg = model.predict(img_array, verbose=0)
	predImg = tf.image.hsv_to_rgb(predImg)
	return np.array(predImg)

def main():
	
	st.write("by M. Rakhmat Dramaga")

if selected3 == "Home":
        display_home_tab()
if selected3 == "Interest":
		display_interest_tab()
elif selected3 == "Upload":
        display_upload_tab()

if __name__ == "__main__":
	main()