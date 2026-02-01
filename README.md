# 🖐️ Real-Time Indian Sign Language Recognition System

A real-time **Indian Sign Language (ISL) Recognition System** designed to bridge the communication gap between signers and non-signers using **computer vision** and **machine learning**.
The system captures hand gestures via a webcam, recognizes ISL words, forms grammatically correct sentences, estimates basic facial emotion, and provides both **text and speech output** for effective communication.

---

## ✨ Features

* Real-time hand gesture detection using webcam
* Indian Sign Language (ISL) word recognition
* Sentence formation with grammar correction
* Facial emotion estimation (Happy / Neutral)
* Text-to-speech output for non-signers
* Multilingual translation support
* Interactive chat-style interface

---

## 🛠️ Technologies Used

* Python 3.11
* OpenCV
* MediaPipe
* scikit-learn (MLPClassifier)
* NumPy
* Pandas
* Git & GitHub

---

## 📁 Project Structure

```text
real-time-isl-recognition/
│
├── app.py          # Basic ISL sentence recognition
├── chat.py         # Full real-time chat interface
├── collect.py      # Gesture data collection
├── train.py        # Model training script
├── config.py       # Common configuration settings
├── grammar.py      # Grammar correction logic
├── emotion.py      # Facial emotion estimation
├── speech.py       # Text-to-speech handling
├── translate.py    # Multilingual translation
│
├── dataset/        # Gesture CSV datasets
├── models/         # Trained model & label encoder
├── requirements.txt# Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.11
* Webcam
* macOS (for text-to-speech support)

### Setup Instructions

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train the Model (Run Once)

```bash
python train.py
```

### Run the Application

```bash
python chat.py
```

---

## ⌨️ Keyboard Controls

| Key | Action                  |
| --- | ----------------------- |
| `e` | End sentence            |
| `b` | Remove last word        |
| `c` | Clear current sentence  |
| `t` | Type reply (non-signer) |
| `q` | Quit application        |

---

## 🧠 How the System Works

1. **MediaPipe** detects hand landmarks in real time.
2. Landmark coordinates are extracted and normalized.
3. A trained machine learning model predicts ISL words.
4. Predictions are stabilized across multiple frames.
5. Grammar rules convert words into meaningful sentences.
6. Facial emotion is estimated using face landmarks.
7. Output is spoken aloud and translated into multiple languages.

---

## ⚠️ Limitations

* Emotion detection is basic and rule-based
* Gesture accuracy depends on lighting and camera quality
* Limited ISL vocabulary (expandable)

---

## 🚀 Future Enhancements

* Deep learning-based gesture recognition
* Improved emotion detection
* Expanded ISL vocabulary
* Cross-platform text-to-speech support
* Web or mobile application version

---

## 🔐 Privacy & Security

* No video or images are stored
* No personal or sensitive data is collected
* Camera input is processed in real time only
* Only sentence text is used for translation
* No API keys or credentials are stored

---

## 🎓 Academic Note

This project was developed for **academic purposes** as a *minor project* in the domain of **Computer Vision and Machine Learning**, with applications in **accessibility and assistive communication systems**.

---

## 👩‍💻 Developer

**Sravya**

Cyber Security & Digital Forensics Student
National Forensic Sciences University (NFSU), Gujarat

---

## 📄 License

This project is **open-source** and intended for **educational use**.


