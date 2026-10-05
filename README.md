---
title: Astro Image Classifier
emoji: 🌌
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.29.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: VGG19 + DenseNet201 ensemble over 11 astronomical classes
---

# 🌌 Astro Image Classifier

An end-to-end **deep learning project** that classifies astronomical images into
11 classes (black hole, earth, galaxy, jupiter, mars, mercury, neptune, pluto,
saturn, uranus, venus) using **transfer learning, attention mechanisms and
ensemble learning**. It goes from experimentation and training in a Jupyter
notebook to a live web demo.

Built with **TensorFlow/Keras** and **Gradio**; the demo page is a static site on **Vercel**.

---

## 🚀 Features

- Upload an astronomical image (or pick an example) and get the **predicted class**.
- **Ensemble model** (VGG19 + DenseNet201) for improved robustness.
- Transfer learning with **pretrained CNNs**.
- Experiments with **CBAM and spatial attention**.
- Shows **probabilities** for all 11 classes.

---

## 📈 Results

Ensemble (VGG19 + DenseNet201, averaged) on the held-out test split
(446 images, `validation_split=0.3`, `seed=42`):

| Model | Test accuracy |
|---|---|
| DenseNet201 branch | 0.9888 |
| VGG19 branch | 0.9888 |
| **Ensemble (deployed)** | **0.9910** |

Macro F1 0.990. The four errors are three venus/mars confusions and one
black_hole predicted as galaxy.

The model is closed-world: every input is forced into one of the 11 classes, so
an image of anything else still gets a confident, meaningless label.

---

## 🧠 Project pipeline

### 1️⃣ Experimentation & training — `notebooks/Astro_CNN_project.ipynb`

- Dataset loading and preprocessing, data augmentation
- Transfer learning (DenseNet, VGG, ResNet) and fine-tuning
- Attention mechanisms (spatial attention, CBAM)
- Ensemble learning
- Evaluation: accuracy, F1-score, ROC-AUC, confusion matrix
- Saving trained models (`.keras`)

### 2️⃣ Model selection

The best model, the VGG19 + DenseNet201 ensemble, is published on the Hugging Face
Hub as [`MA29/astro-image-classifier`](https://huggingface.co/MA29/astro-image-classifier)
(267 MB). It is not committed to this repository.

### 3️⃣ Deployment

```
Vercel (web/index.html)  ──@gradio/client──▶  Hugging Face Space (app.py)
   static page, free                            TensorFlow on a free ZeroGPU Space
```

- `app.py` is a Gradio app. It downloads the weights from the Hub at startup and
  exposes a `/classify` endpoint (image in, class probabilities out).
- `web/` is the demo page. It sends the image to the Space with
  [`@gradio/client`](https://www.npmjs.com/package/@gradio/client) and draws the result.

Vercel alone can't run the model: TensorFlow plus the 267 MB weights are far
over its serverless size limit. Free Gradio Spaces only come with ZeroGPU
hardware, which requires one `@spaces.GPU` function to exist; the classifier
itself runs on the Space's CPU in about a second per image, so it uses none of
the daily GPU quota. Free Spaces sleep after a period without visitors; the page then
shows "Waking up the server…" and waits while it restarts.

---

## ☁️ Deploying

### 1. Hugging Face Space

1. On huggingface.co, create a **new Space** → SDK **Gradio** → *Blank* →
   hardware **ZeroGPU** (the free option), visibility **Public**. Name it
   `astro-image-classifier`.
2. Create an access token with **write** permission (Settings → Access Tokens).
3. In this GitHub repo, go to Settings → Secrets and variables → Actions and add:
   - secret `HF_TOKEN` = the token
   - variable `HF_SPACE` = `MA29/astro-image-classifier`
4. Run the **Sync to Hugging Face Space** workflow (Actions tab), or push to
   `main`. It copies `app.py`, `utils/`, `requirements.txt` and this README to
   the Space, which installs the requirements and starts the app.

### 2. Vercel page

1. On vercel.com, **Add New → Project** and import this repository.
2. Set **Root Directory** to `web`, Framework Preset **Other**, no build command. Deploy.
3. If the Space isn't `MA29/astro-image-classifier`, change `window.SPACE_ID`
   at the top of `web/index.html`.

---

## 💻 Running locally

Python **3.11** (TensorFlow 2.18 has no wheels for 3.14).

```bash
pip install -r requirements.txt
python app.py
```

Open the local URL Gradio prints. The weights download from the Hub on first start.

With Docker:

```bash
docker build -t astro-classifier .
docker run -p 7860:7860 astro-classifier
```

---

## 🔍 Input format

The saved model preprocesses internally: the densenet branch via a
`Lambda(preprocess_input)`, the vgg branch via a channel-swap and
mean-subtraction chain. **Feed raw 0-255 RGB.** Applying `preprocess_input`
beforehand applies it twice and drops accuracy to 71.8% on a 110-image check
(`verify_model.py` reproduces this).

`custom_objects={"preprocess_input": ...}` is required at load time, because the
densenet Lambda is serialised under that name.

---

## 🧩 Project structure

```
├── app.py                      # Gradio app (Hugging Face Space)
├── requirements.txt
├── Dockerfile
├── verify_model.py             # checks the input format on labelled images
├── notebooks/
│   └── Astro_CNN_project.ipynb # training & experimentation
├── utils/
│   ├── preprocessing.py        # image preparation
│   └── labels.py               # class names, in training order
├── web/                        # demo page deployed on Vercel
│   ├── index.html
│   └── examples/               # example images
└── .github/workflows/
    └── sync-to-hf-space.yml    # pushes the app to the Space
```

---

## 📃 License

This project is intended for academic and personal use.

Please check individual dataset and pretrained model licenses (ImageNet, VGG,
DenseNet, ResNet) for their respective terms.
