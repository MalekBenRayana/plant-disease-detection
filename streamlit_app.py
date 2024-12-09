import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os
import google.generativeai as genai
from googletrans import Translator  # Google Translate

# Configuration de Google Generative AI
genai.configure(api_key="AIzaSyBWlYpnB7qyHX74pJS4mnlmRyhu_cMYqis")
try:
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    model = None
    st.error(f"Erreur lors de la configuration du modèle génératif : {e}")

# Configuration de la traduction Google
translator = Translator()

def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(model, image, class_indices_rev):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices_rev.get(predicted_class_index, "Inconnu")
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence

def get_disease_info(disease_name):
    if not model:
        return "Le modèle génératif n'a pas été correctement configuré."
    
    prompt = (f"Fournir des informations détaillées sur la maladie '{disease_name}', y compris :\n"
              f"- Le nom réel de la plante affectée\n"
              f"- Description de la maladie\n"
              f"- Régions en Tunisie où elle se trouve principalement\n"
              f"- Conditions météorologiques nécessaires\n"
              f"- Solutions ou traitements pour la maladie.")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur lors de la génération du contenu : {e}"

def translate_text(text, target_language):
    try:
        if target_language == "fr":
            dest = "fr"
        elif target_language == "en":
            dest = "en"
        elif target_language == "ar":
            dest = "ar"
        else:
            dest = "en"  # Langue par défaut
        translation = translator.translate(text, dest=dest)
        return translation.text
    except Exception as e:
        return f"Erreur de traduction : {e}"

def main():
    st.set_page_config(
        page_title="Détection des Maladies des Plantes",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    page_bg_img = """
    <style>
    body {{
        background-image: url("pixel.jpg");
        background-size: cover;
        background-attachment: fixed;
    }}
    .main {{
         background: #FFFFFF;;
        border-radius: 15px;
        padding: 20px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("🌾 Application de Détection des Maladies des Plantes")
    st.subheader("Téléchargez une image de la feuille d'une culture, et l'application prédira la maladie.")

    with st.sidebar:
        st.image("C:/Users/malek/Desktop/polyy.png", use_column_width=True)
        st.markdown("")

        st.markdown("### 🌱 Fonctionnalités")
        st.markdown("- **Prédire les Maladies des Plantes**")
        st.markdown("- **Informations sur les Maladies**")
        st.markdown("- **Solutions Multilingues**")
        st.markdown("- **Scores de Confiance Visuels**")
        st.markdown("")
        st.markdown("")
        st.markdown("")

        st.markdown("**Enseignante :** Mahbouba Hattab")
        st.markdown("**Réalisé par :** Malek Ben Rayana et Yahia Chouk")

    model_path = 'plant_disease_prediction_model.h5'
    if not os.path.exists(model_path):
        st.error(f"Le fichier du modèle '{model_path}' n'a pas été trouvé. Veuillez vous assurer que le modèle est formé et enregistré.")
        return

    @st.cache_resource
    def load_trained_model():
        return load_model(model_path)

    model = load_trained_model()

    class_indices_path = 'class_indices.json'
    if not os.path.exists(class_indices_path):
        st.error(f"Le fichier des indices de classes '{class_indices_path}' n'a pas été trouvé. Veuillez vous assurer qu'il existe.")
        return

    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)

    class_indices_rev = {int(k): v for k, v in class_indices.items()}

    uploaded_file = st.file_uploader("Télécharger une image...", type=["jpg", "jpeg", "png"])

    target_language = st.selectbox("Sélectionnez une langue pour la solution :", ["fr", "en", "ar"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Image téléchargée', use_column_width=True)
            st.write("Classification en cours...")

            predicted_class, confidence = predict_image_class(model, image, class_indices_rev)
            st.success(f"### Prédiction : {predicted_class}")
            st.progress(int(confidence))
            st.write(f"**Confiance :** {confidence:.2f}%")

            disease_info = get_disease_info(predicted_class)
            st.write("### Informations sur la Maladie :")
            st.info(disease_info)

            translated_info = translate_text(disease_info, target_language)
            st.write("### Solution Traduites :")
            st.markdown(f"**{translated_info}**")

        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {e}")

if __name__ == "__main__":
    main()
