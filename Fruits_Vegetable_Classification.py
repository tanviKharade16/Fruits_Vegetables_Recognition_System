import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import os
import requests

# ----------------- Load Model -----------------
model = load_model('FV.h5')

# ----------------- Labels -----------------
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage',
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn',
    10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes',
    15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango',
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas',
    25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
    29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato',
    33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# ----------------- Improved Calorie Fetching -----------------
def fetch_calories_from_off(prediction):
    try:
        # Map certain names to improve API search results
        name_map = {
            "Sweetpotato": "sweet potato",
            "Bell Pepper": "bell pepper",
            "Chilli Pepper": "chili pepper",
            "Jalepeno": "jalapeno",
            "Capsicum": "bell pepper",
            "Raddish": "radish",
            "Soy Beans": "soybeans",
            "Sweetcorn": "sweet corn",
            "Paprika": "paprika"
        }

        query = name_map.get(prediction, prediction).lower().strip().replace(" ", "%20")
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&action=process&json=1&page_size=5"
        response = requests.get(url)
        products = response.json().get("products", [])

        for product in products:
            nutrients = product.get("nutriments", {})
            calories = nutrients.get("energy-kcal_100g")
            if calories:
                return f"{int(calories)} kcal"

        return "Calories info not found (per 100g)"
    except Exception as e:
        print("API Error:", e)
        return "Calories info not found (per 100g)"

# ----------------- Image Prediction -----------------
def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(" ".join(str(x) for x in y_class))
    res = labels[y]
    return res.capitalize()

# ----------------- Streamlit UI -----------------
def run():
    st.title("🍍 Fruit & Vegetable Classifier with Calorie Info")
    img_file = st.file_uploader("📷 Upload an image", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, caption="Uploaded Image", use_column_width=False)

        # Save image locally
        upload_dir = './upload_images'
        os.makedirs(upload_dir, exist_ok=True)
        save_image_path = os.path.join(upload_dir, img_file.name)
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        result = processed_img(save_image_path)

        # Show result
        st.success(f"**Prediction: {result}**")

        if result in vegetables:
            st.info("**Category: Vegetable 🥬**")
        else:
            st.info("**Category: Fruit 🍎**")
        


        # Show calories
        calories = fetch_calories_from_off(result)
        st.warning(f"**Calories (per 100g): {calories}**")

# ----------------- Run App -----------------
run()
