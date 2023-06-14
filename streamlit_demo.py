import numpy as np
import matplotlib.pyplot as plt

import random
import torchvision.transforms as transforms
from constants import *

import streamlit as st
import time

from dcgan_model import *
from io import BytesIO
from torchvision.utils import save_image
from PIL import Image
from streamlit_cropper import st_cropper



# dcgan module
# import torch
# import torch.nn as nn

@st.cache_data
def load_data():
    '''
    TO-DO
    '''
    from data.CAN_dataset import ds, dataset, class_names, invTrans
    return ds, dataset, class_names, invTrans

# Will only run once if already cached
ds, dataset, class_names, invTrans = load_data()

def manage_progress():
    '''
    TO-DO
    '''
    if 'in_progress' not in st.session_state:
        st.session_state.in_progress = 1
    elif st.session_state.in_progress == 0:
        st.session_state.in_progress = 1
    else:
        st.session_state.in_progress = 0

def plot_images(viz_idx_list, style):
    '''
    TO-DO
    '''
    n = len(viz_idx_list)
    if n == 1:
        fig = plt.figure(figsize=(8, 6))
        rows, cols = 1, 1
    elif n == 2:
        fig = plt.figure(figsize=(10, 8))
        rows, cols = 1, 2
    elif n == 4:
        fig = plt.figure(figsize=(12, 8))
        rows, cols = 2, 2
    elif n == 8:
        fig = plt.figure(figsize=(16, 10))
        rows, cols = 2, 4
    else:
        fig = plt.figure(figsize=(18, 12))
        rows, cols = 4, 4

    plt.suptitle(f'{style.capitalize()} Image(s)', fontsize=15)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        viz_img = dataset[viz_idx_list[i]]['images']
        viz_img = invTrans(viz_img)  # reverts image normalizations for display purposes
        viz_img = np.transpose(viz_img.cpu(), (1, 2, 0))
        plt.imshow(viz_img)

    return fig

# Transformations to visualize
rand_crop = transforms.Compose([
    transforms.RandomCrop(IMG_SIZE*0.75, padding=2)])
rand_hrz_flip = transforms.Compose([
    transforms.RandomHorizontalFlip()])
normalize = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def plot_transformations(viz_idx):
    '''
    TO-DO
    '''
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle(ds.labels.data()['text'][viz_idx][0].capitalize())
    viz_img = dataset[viz_idx]['images']
    viz_img = invTrans(viz_img)
    ax[0][0].imshow(viz_img.T)
    print(viz_img.T.shape, type(viz_img.T))
    ax[0][1].imshow(rand_crop(viz_img).T)
    print(rand_crop(viz_img).T.shape, type(rand_crop(viz_img).T))
    ax[1][0].imshow(rand_hrz_flip(viz_img).T)
    ax[1][1].imshow(normalize(viz_img).T)
    ax[0][0].set_title('Original Image')
    ax[0][1].set_title('Random Crop (75%)')
    ax[1][0].set_title('Random Horizontal Flip (p=0.5)')
    ax[1][1].set_title('Normalize')

    return fig
    

# TO-DO:

# We are not allowed to control the width of the figure in streamlit. One work around:
# Step 1: plot a fig while passing plt.subplot(figsize=(width,height)) argument.
# Step 2: Save the fig fig1.savefig("figure_name.png")
# Step 3: Show the image using st.image

# from PIL import Image
# image = Image.open('figure_name.png')
# st.image(image)

####################################
def main():

    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Generation", "Real Or Fake?"])
    dcgan_generator, dcgan_discriminator = load_dcgan_models()
    if page == "Homepage":
        st.header("Painting Data Exploration")
        st.write(
        "[![Star](https://img.shields.io/github/stars/cmhamakawa/painting_generation.svg)](https://gitHub.com/cmhamakawa/painting_generation)")
        st.markdown("### 🎨 The Application")
        st.write("For other demos, please select a page on the left.")

        with st.form('visualize'):
            batch_size = st.select_slider(
                'Select the number of images to show:',
                options=[1, 2, 4, 8, 16], value=8)

            styles = ['Cubism', 'Impressionism', 'Pointillism', 'Baroque', 'Romanticism']  # to-do: incorporate a few styles from the actual data
            style = st.radio('Select an art style to display:', styles, index=0)

            submit = st.form_submit_button("Submit")

        if submit:
            with st.spinner("We're hard at work generating your painting(s)..."):
                style_idx = np.where(class_names == style.lower())[0][0]
                styles_int = [x[0] for x in ds.labels.data()['value']]
                match_idx = np.where(np.array(styles_int) == style_idx)[0]
                idx_list = [int(i) for i in random.choices(match_idx, k=batch_size)]
                st.write(plot_images(idx_list, style))

        transform = st.button("Visualize Transformations")
        if transform:
            idx = random.choice([i for i in range(len(dataset))])
            st.write(plot_transformations(idx))

    elif page == "Generation":
        download = False
        st.title("Painting Generation")
        fig, image = dcgan_generate_images(dcgan_generator, dcgan_discriminator)
        save_image(image[0], 'testing.png')
        result = st.button("Magic Generator!", on_click=manage_progress)
        if result and st.session_state.in_progress == 1:

            st.write("We're hard at work generating your painting...")

            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(100):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Progress: {(i+1)}%')
                bar.progress(i+1)
                time.sleep(0.1)

            st.write("Generated image:")
            fig, image = dcgan_generate_images(dcgan_generator, dcgan_discriminator)
            save_image(image[0], 'testing.png')
            img = Image.open("testing.png")
            buf = BytesIO()
            img.save(buf, format="png")
            byte_im = buf.getvalue()
            st.write(fig)
            download = True
            if download:
                btn = st.download_button(
                    label="Click here to download image!",
                    data=byte_im,
                    file_name="testing.png",
                    mime="image/png"
                 )
                st.write(":smile:")

        elif result and st.session_state.in_progress == 0:
            st.write("You've clicked it already! Image generation stopped.\n\n"
                     "To re-generate your painting, please click again.")


    else:
        st.title("Real or Fake?")
        file = st.file_uploader(label='Upload an image:')
        if file is not None:
            image_data = file.getvalue()
            image = Image.open(file)
            width  = image.size[0]
            height = image.size[1]
            
            aspect = width / float(height)
            
            ideal_width = 300
            ideal_height = 300
            
            ideal_aspect = ideal_width / float(ideal_height)
            
            if aspect > ideal_aspect:
                # Then crop the left and right edges:
                new_width = int(ideal_aspect * height)
                offset = (width - new_width) / 2
                resize = (offset, 0, width - offset, height)
            else:
                # ... crop the top and bottom:
                new_height = int(width / ideal_aspect)
                offset = (height - new_height) / 2
                resize = (0, offset, width, height - offset)
            
            new_image = image.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)
            st.image(new_image)
            real_prob = image_classifier(dcgan_discriminator, file)
            reported_probability = "The scalar percentage that this is a real image is " + real_prob + "%."
            st.subheader(reported_probability)

if __name__ == "__main__":
    main()