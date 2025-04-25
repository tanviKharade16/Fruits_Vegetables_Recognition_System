import os
import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# CSS for background image
page_bg_img = """
<style>
body {
    background-image: url("image.png");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white; /* Ensures text is visible against the background */
}
</style>
"""
# Apply CSS for background
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load your trained model
model = load_model('FV.h5')

# Label mapping
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

# Fruit and Vegetable category lists
fruits = [
    'Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno',
    'Kiwi', 'Lemon', 'Mango', 'Orange', 'Paprika', 'Pear', 'Pineapple',
    'Pomegranate', 'Watermelon'
]

vegetables = [
    'Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn',
    'Cucumber', 'Eggplant', 'Ginger', 'Lettuce', 'Onion', 'Peas', 'Potato',
    'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato', 'Tomato',
    'Turnip'
]

# Fetch calories info from Google
def fetch_calories(prediction):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        query = f"calories in 100 grams of {prediction}"
        url = f"https://www.google.com/search?q={query}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        for span in soup.find_all("span"):
            if "calories" in span.text.lower() and "100" in span.text:
                return span.text

        result = soup.find("div", class_="BNeawe iBp4i AP7Wnd")
        if result:
            return result.text

        return "Calories info not found"
    except Exception as e:
        st.error("Could not fetch calories.")
        print(e)
        return None

# Predict image
def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    prediction = model.predict(img)
    pred_class = prediction.argmax(axis=-1)
    pred_label = labels[int(pred_class)]
    return pred_label.capitalize()

# Streamlit Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Image Recognition"])

# Home Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    st.image("home_img.jpg", use_column_width=True)

    st.markdown("""
    Welcome to the Fruits and Vegetables Recognition System! 🍎🥦🍇

    Our goal is to help you quickly and accurately identify different fruits and vegetables. 
    Simply upload an image, and our system will recognize it, making your experience enjoyable and informative. 
    Let’s celebrate the diversity of produce together and make identification fun and easy!
    """)

    st.markdown("---")

    st.markdown("""
    ### How It Works
    1. *Upload Image:* Head over to the *Image Recognition page* and upload an image of a fruit or vegetable.
    2. *Analysis:* Our system will analyze the image using cutting-edge algorithms to identify the type of produce.
    3. *Results:* See the results instantly, along with helpful information about the fruit or vegetable you uploaded.
    """)

    st.markdown("---")

    st.markdown("""
    ### Why Choose Us?
    - *Accuracy:* Utilizing advanced machine learning, our system provides accurate recognition.
    - *User-Friendly:* Designed for a smooth, enjoyable experience for all users.
    - *Fast and Reliable:* Get results in seconds, allowing you to focus on what matters.
    """)

    st.markdown("---")

    st.markdown("""
    ### Get Started
    Click on the *Image Recognition page* in the sidebar to upload an image and discover the power of our Fruits and Vegetables Recognition System!
    """)

    st.markdown("---")

# About Page
elif app_mode == "About":
    st.header("About")

    st.markdown("---")

    st.markdown("""
    This app uses a deep learning model to classify common fruits and vegetables.

    - Dataset includes 35+ different categories.
    - Model is trained using Keras and TensorFlow.
    """)

    st.markdown("---")

    st.markdown("""
    ### Dataset
    - A diverse collection of images representing various fruits and vegetables, designed for image recognition tasks.
    - Includes commonly found produce from kitchens and markets, aiding food-recognition applications.
    - Covered categories:
      - *Fruits*: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango
      - *Vegetables*: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant
    """)

    st.markdown("---")

    st.markdown("""
    ### Content
    - The dataset is organized into three main folders:
      1. *Train*: Contains 100 images per category, used to train the model.
      2. *Test*: Contains 10 images per category for evaluation.
      3. *Validation*: Contains 10 images per category for fine-tuning.
    - Each folder has subdirectories for each produce type, ensuring organized model training.
    """)

    st.markdown("---")

# Image Recognition Page
elif app_mode == "Image Recognition":
    st.header("Image Recognition")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    if test_image and st.button("Predict"):
        upload_dir = './upload_images'
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

        result = prepare_image(save_path)

        if result in vegetables:
            st.info("*Category : Vegetable 🥦*")
        elif result in fruits:
            st.info("*Category : Fruit 🍓*")
        else:
            st.info("*Category : Unknown*")

        st.success("*Predicted : " + result + "*")

        calories = fetch_calories(result)
        if calories:
            st.warning(f"*Calories: {calories} (per 100 grams)*")