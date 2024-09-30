from model import create_model

import gradio as gr
import time
import torch
from torch import nn
from torchvision import models, transforms
import pandas as pd
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

model_ft, data_transforms, class_names = create_model()

def predict_prob(img_path):

    df = pd.read_csv('https://github.com/Rudy1209/food/blob/main/indian_food_df_needed (1).csv', index_col='name')
    was_training = model_ft.training
    model_ft.eval()
    t0 = time.time()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model_ft(img)
        pred_proba = torch.softmax(outputs, dim=1)
        pred_proba_labels = {class_names[i]: pred_proba[0][i].item() for i in range(len(class_names))}
        _, preds = torch.max(outputs, 1)

        model_ft.train(mode=was_training)
    return pred_proba_labels, df.loc[class_names[preds[0]]]['description'], time.time()-t0

title  = "Indian Food Classifier and Nutritional value Calculator"
description = "Identity the food in the picture and a get a description of their nutritional value"
article = "Works for Indian cuisine only and does not pick on fast foods"
example_list = [['https://github.com/Rudy1209/food/blob/main/biryani.jpg'],
 ['https://github.com/Rudy1209/food/blob/main/butter_chicken.jpg'],
 ['https://github.com/Rudy1209/food/blob/main/chana_masala.jpg'],
 ['https://github.com/Rudy1209/food/blob/main/gulab_jamoon.jpg']]


demo = gr.Interface(
    fn=predict_prob,
    inputs=gr.Image(type='filepath'),
    outputs=[
        gr.Label(num_top_classes=4, label="Predictions"),
        gr.Textbox(label='Description'),
        gr.Number(label='Time Taken (s)')
    ],
    title=title,
    description=description,
    article=article,
    examples=example_list
)

demo.launch(
    debug=False,
    share=True
)

