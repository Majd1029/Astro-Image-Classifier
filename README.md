# 🌌 Astro Image Classification App

An end-to-end **deep learning project** for classifying astronomical images using **transfer learning, attention mechanisms, and ensemble learning**.  
The project follows a clear pipeline from **experimentation and training (Jupyter Notebook)** to **deployment (Streamlit + Docker)**.

Built with **TensorFlow/Keras** and **Streamlit**.

---

## 🚀 Features

- Upload an astronomical image and receive a **predicted class**.
- **Ensemble model** (VGG19 + DenseNet201) for improved robustness.
- Transfer learning with **pretrained CNNs**.
- Advanced architectures including **CBAM attention mechanisms**.
- Displays **prediction probabilities** for all classes.
- Modular and extensible project structure.
- Dockerized for easy deployment.

---

## 🧠 Project Pipeline

This project follows a complete and structured machine learning workflow:

### 1️⃣ Experimentation & Training (Notebook)

All experiments, training, and evaluations are performed in the notebook:

The notebook includes:
- Dataset loading and preprocessing
- Data augmentation
- Transfer learning (DenseNet, VGG, ResNet)
- Fine-tuning strategies
- Attention mechanisms (Spatial Attention, CBAM)
- Ensemble learning
- Model evaluation (accuracy, F1-score, ROC-AUC, confusion matrix)
- Saving trained models (`.keras`)

---

### 2️⃣ Model Selection

The best-performing model (ensemble of VGG19 + DenseNet201) is selected for deployment.

⚠️ **Trained model files are NOT included in this repository** due to GitHub’s file size limitations.

---

### 3️⃣ Inference & Deployment

The trained model is used in a **Streamlit web application** (`app.py`) to:
- Load the trained model
- Preprocess user-uploaded images
- Perform inference
- Display predictions and probabilities

The application can be run locally or deployed using **Docker**.

---

## 📦 Requirements

Python **3.10+** recommended.

Install dependencies with:

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

- 1️⃣ Clone the repository:
```bash
git clone <your-repo-url>
cd astro_app
```
- 2️⃣ Download the trained model

Download the trained ensemble model separately and place it in:
```bash
astro_app/models/ensemble_model.keras
```
The models/ directory is ignored by Git and should be created locally.

- 3️⃣ Class labels

Ensure the class names in utils/labels.py are in the same order as used during training.
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
├── app.py                     # Streamlit inference app
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
│
├── notebooks/
│   └── Astro_CNN_project.ipynb # Training & experimentation notebook
│
├── models/                    # (ignored by Git)
│   └── ensemble_model.keras
│
├── utils/
│   ├── preprocessing.py       # Image preprocessing utilities
│   └── labels.py              # Class names
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

- Ensemble learning improves generalization and robustness.
- Attention mechanisms help the model focus on relevant regions.
- Evaluation metrics include:
    - Accuracy
    - Precision / Recall / F1-score
    - ROC-AUC (multiclass)
    - Confusion Matrix

## 🐳 Docker Support

- The application can be containerized and deployed using Docker.
```bash
docker build -t astro-classifier .
docker run -p 8501:8501 astro-classifier
```

##  📃 License
This project is intended for academic and personal use.

Please check individual dataset and pretrained model licenses (ImageNet, VGG, DenseNet, ResNet) for their respective terms.