import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_eff
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

# Labels des catégories
CATEGORY_LABELS = [
    "Baby Care",
    "Beauty and Personal Care",
    "Computers",
    "Home Decor & Festive Needs",
    "Home Furnishing",
    "Kitchen & Dining",
    "Watches"
]

# Charger les modèles
@st.cache_resource  # Pour éviter de recharger les modèles à chaque interaction
def load_models():
    model_vgg16 = load_model('best_model_vgg16.keras')
    model_efficientnet = load_model('best_model_efficientnet.keras')
    return model_vgg16, model_efficientnet

model_vgg16, model_efficientnet = load_models()

# Fonction pour prétraiter une image
def preprocess_image(img_path, model_type="vgg16", target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    if model_type == "vgg16":
        img_array = preprocess_vgg(img_array)
    elif model_type == "efficientnet":
        img_array = preprocess_eff(img_array)
    return img, img_array

# Analyse exploratoire des données
def exploratory_analysis(image_folder):
    images = [
        os.path.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.endswith(('.jpg', '.jpeg', '.png'))
    ]
    if not images:
        st.error("Aucune image trouvée dans le dossier sélectionné.")
        return

    st.subheader("Exemples d'images du dataset")
    cols = st.columns(3)
    for i, img_path in enumerate(images[:9]):  # Montrer jusqu'à 9 images
        img = image.load_img(img_path, target_size=(224, 224))
        # Afficher l'image avec la catégorie associée
        img_name = os.path.basename(img_path)
        img_category = img_name.split('_')[0]  # Extraction de la catégorie à partir du nom de fichier (ajuster selon votre convention)
        with cols[i % 3]:
            st.image(img, caption=f"{img_name} - {img_category}", use_column_width=True)

    # Sélectionner une image pour les transformations
    st.subheader("Transformation d'images")
    selected_img_path = st.selectbox("Choisir une image à transformer :", images)
    if selected_img_path:
        original_img, _ = preprocess_image(selected_img_path, model_type="vgg16")
        
        # Appliquer les transformations
        hist_eq_img = ImageOps.equalize(original_img)
        blurred_img = original_img.filter(ImageFilter.GaussianBlur(5))

        col1, col2, col3 = st.columns(3)
        col1.image(original_img, caption="Originale", use_column_width=True)
        col2.image(hist_eq_img, caption="Égalisation d'histogramme", use_column_width=True)
        col3.image(blurred_img, caption="Floutage", use_column_width=True)

# Prédictions
def make_predictions(image_path):
    _, img_array_vgg = preprocess_image(image_path, model_type="vgg16")
    _, img_array_eff = preprocess_image(image_path, model_type="efficientnet")

    prediction_vgg = model_vgg16.predict(img_array_vgg)
    prediction_eff = model_efficientnet.predict(img_array_eff)

    predicted_label_vgg = np.argmax(prediction_vgg, axis=1)[0]
    predicted_label_eff = np.argmax(prediction_eff, axis=1)[0]

    return {
        "vgg16": CATEGORY_LABELS[predicted_label_vgg],
        "efficientnet": CATEGORY_LABELS[predicted_label_eff],
    }

# Interface utilisateur
st.title("Application de Classification d'Images")
st.sidebar.header("Options")
option = st.sidebar.selectbox("Choisissez une action :", ["Exploration du Dataset", "Prédictions"])

if option == "Exploration du Dataset":
    st.header("Exploration du Dataset")
    image_folder = st.text_input("Chemin vers le dossier contenant les images :", "images_de_test")
    if os.path.exists(image_folder):
        exploratory_analysis(image_folder)
    else:
        st.error("Le dossier spécifié n'existe pas.")

elif option == "Prédictions":
    st.header("Prédictions des modèles")
    # Récupérer les images du dossier
    image_folder = st.text_input("Chemin vers le dossier contenant les images :", "images_de_test")
    images = [
        img for img in os.listdir(image_folder)
        if img.endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if images:
        selected_img = st.selectbox("Choisir une image à prédire :", images)
        
        if selected_img:
            image_path = os.path.join(image_folder, selected_img)
            st.image(image_path, caption="Image sélectionnée", use_column_width=True)

            # Obtenir les prédictions
            predictions = make_predictions(image_path)

            # Afficher les résultats (affichage uniquement de la catégorie)
            st.subheader("Résultats des prédictions")
            st.write(f"**VGG16** : {predictions['vgg16']}")
            st.write(f"**EfficientNet** : {predictions['efficientnet']}")
    else:
        st.error("Aucune image trouvée dans le dossier spécifié.")
