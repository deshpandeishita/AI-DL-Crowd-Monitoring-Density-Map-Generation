import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import urllib.request
import tempfile
import platform
from torchvision import transforms

# =========================
# CONFIG
# =========================
SWITCH_THRESHOLD = 75
ALERT_THRESHOLD = 120

MODEL_DIR = "models"
MODEL_A_PATH = f"{MODEL_DIR}/csrnet_final.pth"
MODEL_B_PATH = f"{MODEL_DIR}/csrnet_finalB.pth"

MODEL_A_URL = "https://huggingface.co/deshpandeishita/csrnet-crowd-counting/resolve/main/csrnet_final.pth"
MODEL_B_URL = "https://huggingface.co/deshpandeishita/csrnet-crowd-counting/resolve/main/csrnet_finalB.pth"

# =========================
# PREPARE MODEL FILES
# =========================
os.makedirs(MODEL_DIR, exist_ok=True)

def download_model(url, path):
    if not os.path.exists(path):
        with st.spinner(f"Downloading {os.path.basename(path)}..."):
            urllib.request.urlretrieve(url, path)

download_model(MODEL_A_URL, MODEL_A_PATH)
download_model(MODEL_B_URL, MODEL_B_PATH)

# =========================
# CSRNet MODEL
# =========================
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
        )

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, dilation=2, padding=2), nn.ReLU()
        )

        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return self.output(self.backend(self.frontend(x)))

# =========================
# LOAD MODELS (SAFE)
# =========================
@st.cache_resource
def load_models():
    device = torch.device("cpu")

    modelA = CSRNet().to(device)
    modelB = CSRNet().to(device)

    modelA.load_state_dict(torch.load(MODEL_A_PATH, map_location=device))
    modelB.load_state_dict(torch.load(MODEL_B_PATH, map_location=device))

    modelA.eval()
    modelB.eval()

    return modelA, modelB, device

modelA, modelB, device = load_models()

# =========================
# PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return transform(frame).unsqueeze(0).to(device)

# =========================
# INFERENCE
# =========================
def run_inference(tensor):
    with torch.no_grad():
        dA = modelA(tensor)
        countA = dA.sum().item()

        if countA > SWITCH_THRESHOLD:
            return dA, countA, "Part A"
        else:
            dB = modelB(tensor)
            return dB, dB.sum().item(), "Part B"

# =========================
# UI
# =========================
st.set_page_config(layout="wide")
st.title("Crowd Density Monitoring System")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_box = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tensor = preprocess(frame)
        density, count, model_used = run_inference(tensor)

        density = density.squeeze().cpu().numpy()
        density = cv2.resize(density, (frame.shape[1], frame.shape[0]))
        density = (density / (density.max() + 1e-6) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)
        output = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        st.write(f"Count: {int(count)} | Model: {model_used}")
        frame_box.image(output, channels="BGR")

    cap.release()
