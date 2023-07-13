import streamlit as st
import numpy as np
import torchvision.transforms as T
import torch
from PIL import Image
import io
import pandas as pd
def classify(image):

    # Load image that has been uploaded
    fn = io.BytesIO(image.getvalue())

    img = Image.open(fn)
    img.load()


    # Display the image
    ratio = img.size[0] / img.size[1]
    c = img.copy()
    c.thumbnail([ratio * 200, 200])
    st.image(c)

    # Transform to tensor
    timg = T.ToTensor()(img).unsqueeze_(0)

    # Calling the model
    softmax = learn_inf(timg).data.cpu().numpy().squeeze()
    
    # Get the indexes of the classes ordered by softmax
    # (larger first)
    idxs = np.argsort(softmax)[::-1]
    
    # Loop over the classes with the largest softmax
    labels_indeices=[]
    labels_names=[]
    labels_p=[]
    for i in range(5):
        # Get softmax value
        p = softmax[idxs[i]]
    
        # Get class name
        landmark_name = learn_inf.class_names[idxs[i]].split('.')
        
        labels_indeices.append(landmark_name[0])
        labels_names.append(landmark_name[1])
        labels_p.append(p)
    
    st.write(pd.DataFrame({
        'model_in':labels_indeices,
        'name':labels_names,
        'prob':labels_p}).T)

learn_inf = torch.jit.load('checkpoints/transfer_exported.pt')

uploaded_images = st.file_uploader("Upload your images here...",type=['png','jpg', 'jpeg'],accept_multiple_files=True)

for image in uploaded_images:
    classify(image)

