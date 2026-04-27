# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import cv2, os, json
# from collections import Counter
# import tensorflow as tf

# app = FastAPI(title="GTSRB Classifier API", version="1.0.0")
# app.add_middleware(CORSMiddleware, allow_origins=["*"],
#                    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# SECTION_INFO = {
#     "section1":  {"name": "Optimizer & LR",           "img_size": 32},
#     "section2":  {"name": "Opt x LR x Batch Size",    "img_size": 32},
#     "section3a": {"name": "DNN vs CNN No Aug",         "img_size": 32},
#     "section3b": {"name": "DNN vs CNN With Aug",       "img_size": 32},
#     "section4a": {"name": "Deep DNN vs CNN No Aug",    "img_size": 32},
#     "section4b": {"name": "Deep DNN vs CNN With Aug",  "img_size": 32},
#     "section5":  {"name": "Transfer Learning",         "img_size": 64},
# }

# MODELS = {}
# MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
# SUMMARY    = {}

# def load_all():
#     global SUMMARY
#     # Load models
#     for key in SECTION_INFO:
#         path = os.path.join(MODELS_DIR, f"best_{key}.h5")
#         if os.path.exists(path):
#             try:
#                 MODELS[key] = tf.keras.models.load_model(path)
#                 print(f"  Loaded: {key} | {MODELS[key].input_shape}")
#             except Exception as e:
#                 print(f"  Failed {key}: {e}")
#         else:
#             print(f"  Not found: {path}")
#     # Load summary JSON
#     json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "frontend_data.json")
#     if os.path.exists(json_path):
#         with open(json_path) as f:
#             SUMMARY = json.load(f)
#         print(f"  Loaded summary | {len(SUMMARY.get('sections',{}))} sections")
#     print(f"Models: {len(MODELS)}/{len(SECTION_INFO)}")

# load_all()

# def preprocess(image_bytes, img_size):
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("Could not decode image")
#     img = cv2.resize(img, (img_size, img_size))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32) / 255.0
#     return np.expand_dims(img, axis=0)

# @app.get("/")
# def root():
#     return {
#         "message":       "GTSRB Traffic Sign Classifier API",
#         "models_loaded": len(MODELS),
#         "sections":      len(SECTION_INFO),
#         "classes":       SUMMARY.get("num_classes", 43),
#     }

# @app.get("/health")
# def health():
#     return {"status": "ok", "models_loaded": len(MODELS)}

# @app.get("/sections")
# def get_sections():
#     sections = []
#     for k, v in SECTION_INFO.items():
#         sec_data = SUMMARY.get("sections", {}).get(k, {})
#         sections.append({
#             "key":         k,
#             "name":        sec_data.get("name", v["name"]),
#             "loaded":      k in MODELS,
#             "best_key":    sec_data.get("best_key", ""),
#             "accuracy":    sec_data.get("accuracy", None),
#             "input_shape": str(MODELS[k].input_shape) if k in MODELS else "not loaded"
#         })
#     return {"sections": sections}

# @app.get("/classes")
# def get_classes():
#     names = SUMMARY.get("class_names", [])
#     return {"classes": names, "total": len(names)}

# @app.get("/results/summary")
# def get_summary():
#     if SUMMARY:
#         return SUMMARY
#     return {"message": "frontend_data.json not found. Run the notebook first."}

# @app.get("/results/comparison")
# def get_comparison():
#     """Return all section accuracies for comparison charts"""
#     if not SUMMARY:
#         raise HTTPException(404, "No summary data available")
#     sections = SUMMARY.get("sections", {})
#     overall  = SUMMARY.get("overall_best", {})

#     chart_data = []
#     for key, val in sections.items():
#         chart_data.append({
#             "section_key":  key,
#             "section_name": val.get("name", key),
#             "best_key":     val.get("best_key", ""),
#             "accuracy":     val.get("accuracy", 0),
#             "is_best":      key == overall.get("section", "").lower().replace(" ", "").replace(":", "").replace("/", ""),
#         })

#     # Sort by accuracy descending
#     chart_data.sort(key=lambda x: x["accuracy"], reverse=True)

#     return {
#         "sections":     chart_data,
#         "overall_best": overall,
#         "dataset":      SUMMARY.get("dataset", "GTSRB"),
#         "total_models": SUMMARY.get("total_models", 0),
#         "num_classes":  SUMMARY.get("num_classes", 43),
#     }

# @app.post("/predict/all")
# async def predict_all(file: UploadFile = File(...)):
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(400, "File must be an image")
#     image_bytes = await file.read()
#     class_names = SUMMARY.get("class_names", [f"Class {i}" for i in range(43)])
#     results = {}
#     for key, model in MODELS.items():
#         try:
#             img_size = SECTION_INFO[key]["img_size"]
#             img      = preprocess(image_bytes, img_size)
#             preds    = model.predict(img, verbose=0)[0]
#             top1     = int(np.argmax(preds))
#             top5_idx = np.argsort(preds)[::-1][:5]
#             sec_data = SUMMARY.get("sections", {}).get(key, {})
#             results[key] = {
#                 "section_name":    SECTION_INFO[key]["name"],
#                 "best_training_key": sec_data.get("best_key", ""),
#                 "training_accuracy": sec_data.get("accuracy", None),
#                 "predicted_class": class_names[top1] if top1 < len(class_names) else str(top1),
#                 "class_id":        top1,
#                 "confidence":      float(round(float(preds[top1]) * 100, 2)),
#                 "top5": [
#                     {
#                         "class_id":   int(i),
#                         "class_name": class_names[i] if i < len(class_names) else str(i),
#                         "confidence": float(round(float(preds[i]) * 100, 2))
#                     }
#                     for i in top5_idx
#                 ]
#             }
#         except Exception as e:
#             results[key] = {"error": str(e)}

#     preds_list = [v["predicted_class"] for v in results.values() if "predicted_class" in v]
#     majority   = Counter(preds_list).most_common(1)[0][0] if preds_list else "Unknown"

#     # Best section model prediction
#     overall_best_section = SUMMARY.get("overall_best", {}).get("section", "")
#     best_section_key = None
#     for k in SECTION_INFO:
#         if k in overall_best_section.lower() or overall_best_section.lower() in k:
#             best_section_key = k
#             break
#     if not best_section_key and results:
#         best_section_key = max(
#             [k for k in results if "confidence" in results[k]],
#             key=lambda k: results[k].get("confidence", 0),
#             default=None
#         )

#     return {
#         "results":            results,
#         "majority_vote":      majority,
#         "models_used":        len(results),
#         "best_section_key":   best_section_key,
#         "best_prediction":    results.get(best_section_key, {}) if best_section_key else {},
#     }

# @app.post("/predict/{section_key}")
# async def predict(section_key: str, file: UploadFile = File(...)):
#     if section_key not in SECTION_INFO:
#         raise HTTPException(404, f"Section not found. Available: {list(SECTION_INFO.keys())}")
#     if section_key not in MODELS:
#         raise HTTPException(503, f"Model for {section_key} not loaded")
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(400, "File must be an image")

#     image_bytes = await file.read()
#     info        = SECTION_INFO[section_key]
#     model       = MODELS[section_key]
#     class_names = SUMMARY.get("class_names", [f"Class {i}" for i in range(43)])

#     try:
#         img = preprocess(image_bytes, info["img_size"])
#     except Exception as e:
#         raise HTTPException(400, f"Image processing failed: {e}")

#     preds    = model.predict(img, verbose=0)[0]
#     top5_idx = np.argsort(preds)[::-1][:5]
#     top5     = [
#         {
#             "class_id":   int(i),
#             "class_name": class_names[i] if i < len(class_names) else str(i),
#             "confidence": float(round(float(preds[i]) * 100, 2))
#         }
#         for i in top5_idx
#     ]
#     sec_data = SUMMARY.get("sections", {}).get(section_key, {})

#     return {
#         "section":              section_key,
#         "section_name":         info["name"],
#         "best_training_key":    sec_data.get("best_key", ""),
#         "training_accuracy":    sec_data.get("accuracy", None),
#         "prediction":           top5[0],
#         "top5":                 top5,
#         "image_size":           info["img_size"],
#         "input_shape":          str(model.input_shape),
#         "all_confidences":      [float(round(float(p)*100, 4)) for p in preds],
#     }
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2, os, json
from collections import Counter
import tensorflow as tf

app = FastAPI(title="GTSRB Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

SECTION_INFO = {
    "section1":  {"name": "Optimizer & LR",              "img_size": 32},
    "section2":  {"name": "Opt x LR x Batch Size",       "img_size": 32},
    "section3a": {"name": "DNN vs CNN (No Aug)",          "img_size": 32},
    "section3b": {"name": "DNN vs CNN (With Aug)",        "img_size": 32},
    "section4a": {"name": "Deep DNN vs CNN (No Aug)",     "img_size": 32},
    "section4b": {"name": "Deep DNN vs CNN (With Aug)",   "img_size": 32},
    "section5":  {"name": "Transfer Learning",            "img_size": 64},
}

HF_REPO    = "Ubaidkundlik/gtsrb-models"
MODELS_DIR = os.environ.get("MODELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def download_models_from_hf():
    """Download models from HuggingFace if not present locally"""
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files_needed = [f"best_{k}.h5" for k in SECTION_INFO] + ["frontend_data.json"]
    missing = []
    for fname in files_needed:
        dest = os.path.join(MODELS_DIR if fname.endswith(".h5") else RESULTS_DIR, fname)
        if not os.path.exists(dest):
            missing.append(fname)

    if not missing:
        print("All files already present locally")
        return

    print(f"Downloading {len(missing)} files from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        for fname in missing:
            dest_dir = MODELS_DIR if fname.endswith(".h5") else RESULTS_DIR
            dest     = os.path.join(dest_dir, fname)
            print(f"  Downloading {fname}...")
            path = hf_hub_download(
                repo_id   = HF_REPO,
                filename  = fname,
                repo_type = "model",
                local_dir = dest_dir,
            )
            print(f"  Done: {path}")
    except Exception as e:
        print(f"HF download failed: {e}")
        print("Continuing without downloaded models...")

download_models_from_hf()

MODELS  = {}
SUMMARY = {}

def load_all():
    global SUMMARY

    for key in SECTION_INFO:
        path = os.path.join(MODELS_DIR, f"best_{key}.h5")
        if os.path.exists(path):
            try:
                # Try normal load first
                MODELS[key] = tf.keras.models.load_model(path, compile=False)
                print(f"  Loaded: {key} | {MODELS[key].input_shape}")
            except Exception as e1:
                try:
                    # Fallback — patch InputLayer to handle batch_shape
                    import keras.layers as kl
                    orig_init = kl.InputLayer.__init__
                    def patched_init(self, *args, **kwargs):
                        kwargs.pop('batch_shape', None)
                        kwargs.pop('sparse', None)
                        kwargs.pop('ragged', None)
                        orig_init(self, *args, **kwargs)
                    kl.InputLayer.__init__ = patched_init
                    MODELS[key] = tf.keras.models.load_model(path, compile=False)
                    kl.InputLayer.__init__ = orig_init
                    print(f"  Loaded (patched): {key} | {MODELS[key].input_shape}")
                except Exception as e2:
                    print(f"  Failed {key}: {e2}")
        else:
            print(f"  Not found: {path}")

    json_path = os.path.join(RESULTS_DIR, "frontend_data.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            SUMMARY = json.load(f)
        print(f"  Loaded summary | {len(SUMMARY.get('sections',{}))} sections")

    print(f"Models: {len(MODELS)}/{len(SECTION_INFO)}")

load_all()

def preprocess(image_bytes, img_size):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

@app.get("/")
def root():
    return {
        "message":       "GTSRB Traffic Sign Classifier API",
        "models_loaded": len(MODELS),
        "sections":      len(SECTION_INFO),
        "classes":       SUMMARY.get("num_classes", 43),
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(MODELS)}

@app.get("/sections")
def get_sections():
    return {
        "sections": [
            {
                "key":         k,
                "name":        SUMMARY.get("sections",{}).get(k,{}).get("name", v["name"]),
                "loaded":      k in MODELS,
                "best_key":    SUMMARY.get("sections",{}).get(k,{}).get("best_key",""),
                "accuracy":    SUMMARY.get("sections",{}).get(k,{}).get("accuracy", None),
                "input_shape": str(MODELS[k].input_shape) if k in MODELS else "not loaded"
            }
            for k, v in SECTION_INFO.items()
        ]
    }

@app.get("/classes")
def get_classes():
    names = SUMMARY.get("class_names", [])
    return {"classes": names, "total": len(names)}

@app.get("/results/summary")
def get_summary():
    if SUMMARY:
        return SUMMARY
    return {"message": "Summary not available"}

@app.get("/results/comparison")
def get_comparison():
    if not SUMMARY:
        raise HTTPException(404, "No summary data available")
    sections = SUMMARY.get("sections", {})
    overall  = SUMMARY.get("overall_best", {})
    chart_data = []
    for key, val in sections.items():
        chart_data.append({
            "section_key":  key,
            "section_name": val.get("name", key),
            "best_key":     val.get("best_key", ""),
            "accuracy":     val.get("accuracy", 0),
        })
    chart_data.sort(key=lambda x: x["accuracy"], reverse=True)
    return {
        "sections":     chart_data,
        "overall_best": overall,
        "dataset":      SUMMARY.get("dataset", "GTSRB"),
        "total_models": SUMMARY.get("total_models", 0),
        "num_classes":  SUMMARY.get("num_classes", 43),
    }

@app.post("/predict/all")
async def predict_all(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()
    class_names = SUMMARY.get("class_names", [f"Class {i}" for i in range(43)])
    results = {}
    for key, model in MODELS.items():
        try:
            img   = preprocess(image_bytes, SECTION_INFO[key]["img_size"])
            preds = model.predict(img, verbose=0)[0]
            top1  = int(np.argmax(preds))
            results[key] = {
                "section_name":    SECTION_INFO[key]["name"],
                "predicted_class": class_names[top1] if top1 < len(class_names) else str(top1),
                "class_id":        top1,
                "confidence":      float(round(float(preds[top1])*100, 2)),
                "top3": [
                    {"class_id": int(i), "class_name": class_names[i] if i < len(class_names) else str(i),
                     "confidence": float(round(float(preds[i])*100, 2))}
                    for i in np.argsort(preds)[::-1][:3]
                ]
            }
        except Exception as e:
            results[key] = {"error": str(e)}
    preds_list = [v["predicted_class"] for v in results.values() if "predicted_class" in v]
    majority   = Counter(preds_list).most_common(1)[0][0] if preds_list else "Unknown"
    best_key   = max([k for k in results if "confidence" in results[k]],
                     key=lambda k: results[k].get("confidence",0), default=None)
    return {
        "results":          results,
        "majority_vote":    majority,
        "models_used":      len(results),
        "best_section_key": best_key,
        "best_prediction":  results.get(best_key, {}) if best_key else {},
    }

@app.post("/predict/{section_key}")
async def predict(section_key: str, file: UploadFile = File(...)):
    if section_key not in SECTION_INFO:
        raise HTTPException(404, f"Section not found. Available: {list(SECTION_INFO.keys())}")
    if section_key not in MODELS:
        raise HTTPException(503, f"Model for {section_key} not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()
    info        = SECTION_INFO[section_key]
    model       = MODELS[section_key]
    class_names = SUMMARY.get("class_names", [f"Class {i}" for i in range(43)])
    try:
        img = preprocess(image_bytes, info["img_size"])
    except Exception as e:
        raise HTTPException(400, f"Image processing failed: {e}")
    preds    = model.predict(img, verbose=0)[0]
    top5_idx = np.argsort(preds)[::-1][:5]
    top5     = [
        {"class_id": int(i),
         "class_name": class_names[i] if i < len(class_names) else str(i),
         "confidence": float(round(float(preds[i])*100, 2))}
        for i in top5_idx
    ]
    sec_data = SUMMARY.get("sections", {}).get(section_key, {})
    return {
        "section":           section_key,
        "section_name":      info["name"],
        "best_training_key": sec_data.get("best_key", ""),
        "training_accuracy": sec_data.get("accuracy", None),
        "prediction":        top5[0],
        "top5":              top5,
        "image_size":        info["img_size"],
        "input_shape":       str(model.input_shape),
        "all_confidences":   [float(round(float(p)*100, 4)) for p in preds],
    }