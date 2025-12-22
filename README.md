# Astro Image Classification App

A **deep learning web app** for classifying astronomical images using an **ensemble of CNN models** (VGG19 + DenseNet201) with optional attention mechanisms. Built with **TensorFlow/Keras** and **Streamlit**.

---

## 🚀 Features

- Upload any astronomical image and get a **predicted class**.
- Ensemble learning for **improved accuracy**.
- Uses **pretrained CNN models** with fine-tuning on your dataset.
- Displays **prediction probabilities** for all classes.
- Compatible with **CBAM attention mechanisms** for more robust feature learning.
- Easy to extend for new models or datasets.

---

## 📦 Requirements

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Key dependencies:

- tensorflow
- streamlit
- numpy
- pillow
- matplotlib
- seaborn
- scikit-learn


## 🛠️ Setup

- Clone the repository:
```bash
git clone <your-repo-url>
cd astro_app
```
- Place your trained model checkpoints in:
```bash
astro_app/models/
```
Ensemble model: ensemble_model.keras
(Optional) Individual models: best_model_vgg.keras, best_densenet.keras
- Ensure your class names are listed in labels.py or in the same order as your training dataset.

## 💻 Run the Streamlit App
```bash
streamlit run app.py
```
- Open the local URL provided by Streamlit.
- Upload an astronomical image.
- View predicted class and probability scores.
- Images will automatically scale to container width (use_container_width=True).

## 🧩 Project Structure

```bash
astro_app/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
│
├── models/
│   └── ensemble_model.keras
│
└── utils/
    ├── preprocessing.py
    └── labels.py
```
## 📷 Image Upload Example

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

## ⚡ Notes

- Preprocessing is applied before predictions:

vgg_img = vgg_preprocess_input(img_array)
densenet_img = densenet_preprocess_input(img_array)
preds = ensemble_model.predict([vgg_img, densenet_img])

- Ensure you register any custom preprocessing functions or layers with:

from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def vgg_preprocess(x):
    return vgg19_preprocess_input(x)


## 📈 Model Performance

- Ensemble of VGG19 + DenseNet201 improves robustness.
- Optional attention mechanisms (CBAM) enhance focus on key image regions.
- Metrics include: Accuracy, F1-score, ROC-AUC, Confusion Matrix.

##  📃 License
This project is for academic or personal use.
Please check individual OCR engine licenses for their specific terms.