# config.py
# This file just keeps all the common settings in one place

# ---------- Folders ----------
DATASET_DIR = "dataset"      # where all gesture CSV files are saved
MODELS_DIR = "models"        # folder to store trained model + label encoder

# Paths for model files
MODEL_PATH = f"{MODELS_DIR}/gesture_model.pkl"
ENCODER_PATH = f"{MODELS_DIR}/label_encoder.pkl"

# ---------- Data / Camera ----------
SAMPLES_PER_GESTURE = 300    # how many samples I record for each gesture
CAMERA_INDEX = 0             # laptop webcam is usually index 0

# Number of stable frames needed before accepting a word
STABLE_FRAMES_THRESHOLD = 12  # higher = more strict, lower = faster

# ---------- Languages for translation ----------
TARGET_LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "telugu": "te",
    "tamil": "ta",
    "kannada": "kn",
    "gujarati": "gu",
}
