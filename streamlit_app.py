import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import yagmail
import requests
from torchvision import transforms
from pathlib import Path

# =========================
# GLOBAL SETTINGS
# =========================
torch.set_grad_enabled(False)

SWITCH_THRESHOLD = 75
ALERT_THRESHOLD = 120
MAX_FRAMES = 300  # Limit processing for cloud safety

MODEL_A_URL = "https://huggingface.co/deshpandeishita/csrnet-crowd-counting/resolve/main/csrnet_final.pth"
MODEL_B_URL = "https://huggingface.co/deshpandeishita/csrnet-crowd-counting/resolve/main/csrnet_finalB.pth"

MODEL_A_PATH = "csrnet_final.pth"
MODEL_B_PATH = "csrnet_finalB.pth"

# =========================
# CSRNet MODEL
# =========================
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        return self.output(self.backend(self.frontend(x)))

# =========================
# MODEL DOWNLOAD
# =========================
def download_model(url, path):
    if not Path(path).exists():
        with st.spinner(f"Downloading {path}..."):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# =========================
# LOAD MODELS (CPU ONLY)
# =========================
@st.cache_resource
def load_models():
    device = torch.device("cpu")

    download_model(MODEL_A_URL, MODEL_A_PATH)
    download_model(MODEL_B_URL, MODEL_B_PATH)

    model_A = CSRNet().to(device)
    model_B = CSRNet().to(device)

    model_A.load_state_dict(torch.load(MODEL_A_PATH, map_location=device))
    model_B.load_state_dict(torch.load(MODEL_B_PATH, map_location=device))

    model_A.eval()
    model_B.eval()

    return model_A, model_B, device

# =========================
# PREPROCESS
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(frame, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return transform(frame).unsqueeze(0).to(device)

# =========================
# INFERENCE
# =========================
def infer(tensor, model_A, model_B):
    dA = model_A(tensor)
    countA = dA.sum().item()
    if countA > SWITCH_THRESHOLD:
        return dA, countA, "CSRNet Part A"
    dB = model_B(tensor)
    return dB, dB.sum().item(), "CSRNet Part B"

# =========================
# HEATMAP
# =========================
def generate_heatmap(density, frame):
    density = density.squeeze().cpu().numpy()
    density = cv2.resize(density, (frame.shape[1], frame.shape[0]))
    density = density / (density.max() + 1e-6)
    heat = cv2.applyColorMap(np.uint8(255 * density), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heat, 0.4, 0)

# =========================
# SMTP EMAIL (CLOUD SAFE)
# =========================
def send_alert(count):
    yag = yagmail.SMTP(
        user=st.secrets["email"]["sender"],
        password=st.secrets["email"]["app_password"]
    )

    yag.send(
        to=st.secrets["email"]["receiver"],
        subject="Crowd Alert Triggered",
        contents=f"Crowd count exceeded threshold.\nCurrent Count: {int(count)}"
    )

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="CSRNet Crowd Monitoring (Cloud)", layout="wide")
st.title("CSRNet Crowd Monitoring System")

st.warning("Webcam is not supported on Streamlit Cloud. Upload a video instead.")

model_A, model_B, device = load_models()

uploaded = st.file_uploader("Upload Video", type=["mp4"])

if uploaded:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded.read())
    temp.flush()
    temp.close()

    cap = cv2.VideoCapture(temp.name)

    frame_box = st.empty()
    count_box = st.empty()
    model_box = st.empty()

    if "alert_sent" not in st.session_state:
        st.session_state.alert_sent = False
        st.session_state.last_alert_count = 0

    stop = st.button("Stop Processing")

    frame_count = 0

    while cap.isOpened() and frame_count < MAX_FRAMES and not stop:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = preprocess(frame, device)
        density, count, model_used = infer(tensor, model_A, model_B)
        output = generate_heatmap(density, frame)

        count_box.metric("Crowd Count", int(count))
        model_box.write(f"Model Used: {model_used}")

        if count > ALERT_THRESHOLD and not st.session_state.alert_sent:
            if abs(count - st.session_state.last_alert_count) > 10:
                send_alert(count)
                st.session_state.alert_sent = True
                st.session_state.last_alert_count = count
                st.error("🚨 Alert Email Sent")

        frame_box.image(output, channels="BGR")
        frame_count += 1

    cap.release()
