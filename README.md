# GTSRB Traffic Sign Classifier — Backend

A FastAPI backend serving 7 Keras models trained on the German Traffic Sign Recognition Benchmark (GTSRB) — the result of a systematic ablation study comparing optimizers, learning rates, batch sizes, DNN vs. CNN architectures, data augmentation, and transfer learning, across 56 trained model variants.

Pairs with the [gtsrb-frontend](https://github.com/ubaidmuazzam23-lgtm/gtsrb-frontend) repo, which visualizes the comparison results and provides an image-upload UI.

## Table of Contents

- [Experiment Overview](#experiment-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Status](#status)
- [Author](#author)

## Experiment Overview

Each "section" is a separate ablation experiment on GTSRB (43 traffic sign classes), with the best model from each kept and served:

| Section | Experiment | Best Accuracy | Best Config |
|---|---|---|---|
| 1 | Optimizer & learning rate | 75.9% | Adam, lr=0.001 |
| 2 | Optimizer × LR × batch size | 74.5% | SGD, lr=0.01, bs=32 |
| 3A | DNN vs. CNN (no augmentation) | 86.1% | CNN, RMSprop, lr=0.001 |
| 3B | DNN vs. CNN (with augmentation) | 93.4% | CNN, RMSprop, lr=0.001 |
| 4A | Deep DNN vs. deep CNN (no aug.) | 92.0% | Deep CNN, Adam, lr=0.001 |
| 4B | Deep DNN vs. deep CNN (with aug.) | 93.0% | Deep CNN, Adam, lr=0.001, aug |
| 5 | Transfer learning | 89.2% | VGG16 (unfrozen) |

Best result overall: **93.36%** (Section 3B, CNN + RMSprop + augmentation). Trained models are hosted on the Hugging Face Hub at [`Ubaidkundlik/gtsrb-models`](https://huggingface.co/Ubaidkundlik/gtsrb-models) and downloaded automatically on server startup.

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn |
| ML | TensorFlow / Keras |
| Image processing | OpenCV, Pillow |
| Model hosting | Hugging Face Hub |
| Deployment | Render |

## Project Structure

```
app/
  main.py            # FastAPI app: model loading, preprocessing, prediction endpoints
models/               # downloaded .keras model files (auto-fetched from HF Hub at startup)
results/
  frontend_data.json  # generated comparison summary (also auto-fetched from HF Hub)
requirements.txt
render.yaml           # Render deployment config
Procfile
```

## Prerequisites

- Python 3.9+
- Internet access on first run (to download models from Hugging Face Hub)

## Setup

```bash
git clone https://github.com/ubaidmuazzam23-lgtm/road-sign-prediction.git gtsrb-backend
cd gtsrb-backend

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

On first run, the server downloads all 7 `.keras` model files and the results summary from the `Ubaidkundlik/gtsrb-models` Hugging Face repo into `models/` and `results/` — no manual setup needed. Subsequent runs reuse the local copies.

Runs at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `MODELS_DIR` | Optional | Override where model files are stored/read (default: `../models` relative to `app/`) |

No API keys are required — the Hugging Face model repo is public.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `GET` | `/sections` | List all sections and their loaded status |
| `GET` | `/classes` | All 43 GTSRB class names |
| `GET` | `/results/summary` | Full experiment summary |
| `GET` | `/results/comparison` | Section-by-section accuracy comparison |
| `POST` | `/predict/{section_key}` | Predict using one section's best model |
| `POST` | `/predict/all` | Run all 7 models and return a majority vote |

Section keys: `section1`, `section2`, `section3a`, `section3b`, `section4a`, `section4b`, `section5`.

### Example

```bash
curl -X POST "http://localhost:8000/predict/section5" -F "file=@sign.jpg"
```

```json
{
  "section": "section5",
  "section_name": "Section 5: Transfer Learning",
  "prediction": { "class_id": 14, "class_name": "Stop", "confidence": 98.76 },
  "top5": [
    {"class_id": 14, "class_name": "Stop", "confidence": 98.76},
    {"class_id": 17, "class_name": "No entry", "confidence": 0.89}
  ],
  "image_size": 64
}
```

## Deployment

Configured for [Render](https://render.com) via `render.yaml` / `Procfile`:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

Models are fetched from Hugging Face Hub at boot, so no persistent disk or Git LFS is required.

## Status

Academic project exploring systematic ablation over CNN architectures, training hyperparameters, and transfer learning on GTSRB. Not a production classifier.

## Author

[Ubaid Muazzam](https://github.com/ubaidmuazzam23-lgtm)
