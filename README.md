# Edge_vox
An ultra-lightweight edge-based Keyword Spotting system using ESP32 and INMP441. The custom wake word “ESP” is detected locally using a lightweight ML model, triggering audio streaming to a remote ASR server only when needed. Designed for low latency, low CPU/RAM usage, and near-zero false activations.
📌 Problem Statement

Traditional voice-controlled IoT systems continuously send audio to cloud servers, resulting in:

Higher latency
Increased network usage
Higher computational cost
Unnecessary audio transmission
Privacy concerns

This project addresses these problems by performing the initial wake-word detection directly on the edge device.
💡 Proposed Solution

The system uses a hybrid Edge + Cloud architecture.

The ESP32 continuously listens to the microphone and runs a lightweight Keyword Spotting model locally. When the custom keyword "ESP" is detected, the device triggers audio streaming to a remote Automated Speech Recognition (ASR) server.

This avoids continuously sending microphone data to the cloud.

🏗️ System Architecture
        ┌───────────────┐
        │   INMP441     │
        │ I2S Microphone│
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │     ESP32     │
        │               │
        │ Audio Capture │
        │      ↓        │
        │ Preprocessing │
        │      ↓        │
        │     MFCC      │
        │      ↓        │
        │   KWS Model   │
        └───────┬───────┘
                │
         "ESP" Detected
                │
                ▼
        ┌───────────────┐
        │ Audio Stream  │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Remote ASR    │
        │    Server     │
        └───────────────┘
🎯 Custom Keyword

Wake Word: ESP

The model is trained specifically for the custom keyword rather than using a pre-trained global assistant keyword such as "Alexa" or "Hey Google".

🎙️ Hardware
Component	Purpose
ESP32	Edge processing and KWS inference
INMP441	I2S digital microphone
USB Cable	Programming and power
INMP441 Connections
INMP441 → ESP32

VDD  → 3.3V
GND  → GND
SCK  → GPIO 26
WS   → GPIO 25
SD   → GPIO 33
L/R  → GND
🧠 Machine Learning Pipeline

The audio captured from the microphone is processed through the following pipeline:

Raw Audio
    ↓
16 kHz PCM Audio
    ↓
Preprocessing
    ↓
MFCC Feature Extraction
    ↓
Lightweight KWS Model
    ↓
Keyword Classification
    ↓
ESP / Unknown / Silence
📊 Dataset

The initial dataset contains three classes:

dataset/
├── keyword/
├── unknown/
└── silence/

Current dataset:

ESP: 100 samples
Unknown: 200 samples
Silence: 100 samples

The unknown class contains speech other than the target keyword, while the silence class contains background/environmental audio.

⚡ False Activation Handling

A major challenge in Keyword Spotting is preventing normal speech from being incorrectly detected as the wake word.

The system therefore evaluates:

Prediction confidence
Detection threshold
Consecutive prediction windows
Unknown speech
Background noise
Unseen words

The objective is to achieve a high true-positive rate with near-zero false activations.

☁️ Edge + Cloud Architecture

The system does not continuously stream audio to the cloud.

Continuous Listening
        ↓
Local KWS on ESP32
        ↓
Is "ESP" detected?
     ↙       ↘
   NO        YES
   ↓          ↓
Continue    Start
Listening   Streaming
              ↓
           Remote ASR

This architecture reduces unnecessary network communication and allows the computationally intensive ASR process to run remotely.

📏 Target Evaluation Metrics

The project is designed around the following evaluation parameters:

Efficiency
RAM usage
Flash/model size
Idle CPU utilization
Accuracy
True Positive Rate
False Positive Rate
False Activations
Latency

Time between:

Keyword Ending
      ↓
ASR Receiving Audio
Target Constraints
RAM: < 256 KB
Idle CPU: < 10%
Lightweight model suitable for microcontroller deployment
🛠️ Technologies
ESP32
INMP441
Python
MFCC
Machine Learning
TensorFlow Lite / TensorFlow Lite Micro
Embedded C/C++
Remote ASR
