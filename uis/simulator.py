import streamlit as st
import pickle
import base64
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import time
from stqdm import stqdm

from neuraldistributions.utility import imread
from neuraldistributions.sampler import generate_n_samples


def downloader(data):
    pickled_data = pickle.dumps(data)
    b64 = base64.b64encode(pickled_data).decode()
    data_filename = "data.pkl"
    href = f'<a href="data:file/pkl;base64,{b64}" download="{data_filename}">Download data (.pkl)</a>'
    st.markdown(href, unsafe_allow_html=True)


@st.cache(suppress_st_warning=True)
def generate_samples(uploaded_images, n_samples):
    samples = []
    model = torch.load("/project/models/ziffa_scan2")
    for uploaded_image in stqdm(uploaded_images):
        samples_per_image = generate_n_samples(
            model, uploaded_image, n_samples=n_samples, random_seed=None
        )
        samples.append(samples_per_image)

    return samples


def index_parser(neuron_idx):
    return [int(idx) for idx in neuron_idx.split(",")]


def visualize_responses(samples):
    col1, col2 = st.beta_columns(2)
    with col1:
        image_idx = st.text_input("Select an image", 0)
        image_idx = int(image_idx)
    with col2:
        neuron_idx = st.text_input('Select neurons (separate idx with ",")', 0)
        neuron_idx = index_parser(neuron_idx)

    np.random.seed(42)
    unique_colors = list(mcolors.TABLEAU_COLORS.values())
    colors = np.random.choice(unique_colors, size=len(neuron_idx))

    fig, ax = plt.subplots(figsize=(10.0, 2.0), dpi=150)

    for i, s in enumerate(samples[image_idx][:, neuron_idx].T):
        ax.plot(s, c=colors[i], lw=2, label=f"Neuron {neuron_idx[i]}")
    ax.set(yticks=[])
    ax.legend(frameon=False)
    sns.despine(trim=True, left=True)
    st.pyplot(fig)


def main():

    st.title("Simulate population responses")

    ### Intro ###############################################################
    st.write(
        "Here we provide a use-interface for a response simulator. \
        To simulate responses for your images follow these steps:"
    )

    ### Instructions ###############################################################
    col1, col2, col3 = st.beta_columns(3)

    col1.header("Step 1")
    with col1:
        st.info("Upload input images to generate responses..")

    col2.header("Step 2")
    with col2:
        st.warning("Simulate $n$ (default=1) responses to each image..")

    col3.header("Step 3")
    with col3:
        st.success("Visualize and download the simulated responses..")

    st.markdown("---")
    ### Upload images ###############################################################
    uploaded_images = st.file_uploader(
        "Upload image", type=["png", "jpg"], accept_multiple_files=True
    )

    ### Generate samples ###############################################################
    n_samples = st.text_input("How many samples per image?", 1)
    n_samples = int(n_samples)
    start_simulation = st.button("Simulate/Visualize")
    if start_simulation:
        samples = generate_samples(uploaded_images, n_samples)
        visualize_responses(samples)
        downloader(np.stack(samples))


if __name__ == "__main__":
    main()