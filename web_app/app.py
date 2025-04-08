import scikit_learn
import skops.io as sio
import gradio as gr


import warnings
from sklearn.execeptions import InconsistentVersionWarning
trusted_types=[
    "sklearn.pipeline.Pipeline",
    "sklearn.preprocessing.StandardScaler",
    "sklearn.compose.ColumnTransformer",
    "sklearn.preprocessing.OrdinalEncoder",
    "sklearn.impute.SimpleImputer",
    "sklearn.tree.DecisonTreeClassifier",
    "sklearn.ensemble.RandomForestClassifier",
    "numpy.dtype"
]
pipeline=sio.load("./model/drug_pipeline.siops",trusted_types=trusted_types)

def predict_drug(age,sex,blood_pressure,cholestrol,na_to_k_ratio):
    """
    Predicts drugs based on patient features.

    Args:
    age(int): Age of patient
    sex(str):Sex of patient
    blood_pressure(str): Blood_pressure level
    cholestrol(str): Cholestrol level
    na_to_k_ratio(float):Ratio of sodium to potassium in blood
    Returns:
     str : Predicted drug label

    """
    features=[age,sex,blood_pressure,cholestrol,na_to_k_ratio]
    predicted_drug=pipeline.predict([features])[0]
    label= print(f"Predicted_drug: {predicted_drug}")
    return label

inputs=[
    gr.Slider(15,74,step=1,label="Age"),
    gr.Radio(["M","F"],label="Sex"),
    gr.Radio(["High","Low","Normal"],label="Blood_pressure"),
    gr.Radio(["High","Normal"],label="Cholestrol"),
    gr.Slider(6.2,38.2,step=0.1,label="Na_to_K")
]
outputs=[gr.label(num_top_classes=5)]
examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]
title="Drug Classification"
description="Enter the details to correctly identify drug type"
article= "This predicts drug based patient's features"
gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    article=article,
    description=description,
    theme=gr.themes.Soft(),

).launch()