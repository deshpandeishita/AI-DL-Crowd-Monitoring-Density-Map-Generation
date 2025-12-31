import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import urllib.request
import tempfile
import platform
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from torchvision import transforms

# ======================================================
# CONFIGURATION
# ======================================================
SWITCH_THRESHOLD = 75
ALERT_THRESHOLD = 120

# --- SMTP (use Gmail App Password) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"          # CHANGE
SENDER_PASSWORD = "your_app_password"          # CHANGE

ALERT_EMAILS = [
    "admin1@example.com",
    "admin2@example.com"
]

# ======================================================
# MODEL DOWNLOAD (HUGGING FACE)
# ======================================================
os.makedirs("models", exist_ok=True)

MODEL_A_PATH = "models/csrnet_final.pth"
MODEL_B_PATH = "models/csrnet_finalB.pth"

MODEL_A_URL = "https://huggingface.co/deshpandeishita/csrnet-crowd-counting/resolve/main/csrnet_final.pth"
MODEL_B_URL = "https://huggingface.co/deshpandeishita/csrnet-crowd-counting/resolve/main/csrnet_finalB.pth"

if not os.path.exists(MODEL_A_PATH):
    urllib.request.urlretrieve(MODEL_A_URL, MODEL_A_PATH)

if not os.path.exists(MODEL_B_PATH):
    urllib.request.urlretrieve(MODEL_B_URL, MODEL_B_PATH)

# ======================================================
# CSRNet MODEL DEFINITION
# ======================================================
class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

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
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output(x)
        return x

# ======================================================
# LOAD MODELS (CACHED)
# ======================================================
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

# ======================================================
# PREPROCESSING
# ======================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame).unsqueeze(0)
    return tensor.to(device)

# ======================================================
# MODEL SWITCHING
# ======================================================
def switch_inference(tensor):
    with torch.no_grad():
        dA = modelA(tensor)
        countA = dA.sum().item()

        if countA > SWITCH_THRESHOLD:
            return dA, countA, "CSRNet Part A"
        else:
            dB = modelB(tensor)
            return dB, dB.sum().item(), "CSRNet Part B"

# ======================================================
# HEATMAP
# ======================================================
def generate_heatmap(density, frame):
    density = density.squeeze().cpu().numpy()
    density = cv2.resize(density, (frame.shape[1], frame.shape[0]))
    density = density / (density.max() + 1e-6)
    density = (density * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

# ======================================================
# SMTP ALERT
# ======================================================
def send_alert(count):
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["Subject"] = "Crowd Alert"
    msg.attach(MIMEText(f"Crowd count exceeded: {int(count)}", "plain"))

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)

    for email in ALERT_EMAILS:
        server.sendmail(SENDER_EMAIL, email, msg.as_string())

    server.quit()

# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(layout="wide")
st.title("Crowd Monitoring & Density Map System")

IS_CLOUD = platform.system() == "Linux"

tab1, tab2 = st.tabs(["📷 Webcam", "🎥 Video Upload"])

# ======================================================
# TAB 1 – WEBCAM (LOCAL ONLY)
# ======================================================
with tab1:
    if IS_CLOUD:
        st.warning("Webcam is not supported on Streamlit Cloud.")
    else:
        start = st.button("Start Webcam")
        if start:
            cap = cv2.VideoCapture(0)
            frame_slot = st.image([])
            alert_sent = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                tensor = preprocess(frame)
                density, count, model_used = switch_inference(tensor)
                output = generate_heatmap(density, frame)

                st.metric("Crowd Count", int(count))
                st.write(f"Model Used: {model_used}")

                if count > ALERT_THRESHOLD and not alert_sent:
                    send_alert(count)
                    st.error("ALERT SENT")
                    alert_sent = True

                frame_slot.image(output, channels="BGR")

            cap.release()

# ======================================================
# TAB 2 – VIDEO UPLOAD (CLOUD SAFE)
# ======================================================
with tab2:
    uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_slot = st.image([])
        alert_sent = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            tensor = preprocess(frame)
            density, count, model_used = switch_inference(tensor)
            output = generate_heatmap(density, frame)

            st.metric("Crowd Count", int(count))
            st.write(f"Model Used: {model_used}")

            if count > ALERT_THRESHOLD and not alert_sent:
                send_alert(count)
                st.error("ALERT SENT")
                alert_sent = True

            frame_slot.image(output, channels="BGR")

        cap.release()
