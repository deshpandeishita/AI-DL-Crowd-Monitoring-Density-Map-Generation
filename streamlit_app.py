import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from torchvision import transforms

# =========================
# CONFIGURATION
# =========================
SWITCH_THRESHOLD = 75
ALERT_THRESHOLD = 120

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"

ALERT_EMAILS = [
    "admin1@example.com",
    "admin2@example.com"
]

MODEL_A_PATH = "models/csrnet_partA.pth"
MODEL_B_PATH = "models/csrnet_partB.pth"

# =========================
# CSRNet MODEL
# =========================
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
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, dilation=2, padding=2), nn.ReLU(inplace=True)
        )

        self.output = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output(x)
        return x

# =========================
# LOAD MODELS ONCE
# =========================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_A = CSRNet().to(device)
    model_B = CSRNet().to(device)

    model_A.load_state_dict(torch.load(MODEL_A_PATH, map_location=device))
    model_B.load_state_dict(torch.load(MODEL_B_PATH, map_location=device))

    model_A.eval()
    model_B.eval()

    return model_A, model_B, device

# =========================
# PREPROCESSING
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess(frame, device):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame).unsqueeze(0)
    return tensor.to(device)

# =========================
# SWITCHING LOGIC
# =========================
def switch_inference(frame_tensor, model_A, model_B):
    with torch.no_grad():
        dA = model_A(frame_tensor)
        dB = model_B(frame_tensor)

    count_A = dA.sum().item()

    if count_A > SWITCH_THRESHOLD:
        return dA, count_A, "CSRNet Part A"
    else:
        return dB, dB.sum().item(), "CSRNet Part B"

# =========================
# HEATMAP
# =========================
def generate_heatmap(density, frame):
    density = density.squeeze().cpu().numpy()
    density = cv2.resize(density, (frame.shape[1], frame.shape[0]))
    density = density / (density.max() + 1e-5)
    density = (density * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(density, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

# =========================
# SMTP ALERT
# =========================
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

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("Real-Time Crowd Monitoring System")

model_A, model_B, device = load_models()

tab1, tab2 = st.tabs(["📷 Webcam Monitoring", "🎥 Video Upload Monitoring"])

# =========================
# TAB 1 – WEBCAM
# =========================
with tab1:
    st.header("Live Webcam Crowd Detection")
    start = st.button("Start Webcam")

    if start:
        cap = cv2.VideoCapture(0)
        alert_sent = False
        frame_window = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            tensor = preprocess(frame, device)
            density, count, model_used = switch_inference(tensor, model_A, model_B)
            output = generate_heatmap(density, frame)

            st.metric("Crowd Count", int(count))
            st.write(f"Model Used: {model_used}")

            if count > ALERT_THRESHOLD and not alert_sent:
                send_alert(count)
                st.error("ALERT SENT: Crowd Limit Exceeded")
                alert_sent = True

            frame_window.image(output, channels="BGR")

        cap.release()

# =========================
# TAB 2 – VIDEO UPLOAD
# =========================
with tab2:
    st.header("Video Upload Crowd Detection")
    uploaded = st.file_uploader("Upload video", type=["mp4"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)

        alert_sent = False
        frame_window = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            tensor = preprocess(frame, device)
            density, count, model_used = switch_inference(tensor, model_A, model_B)
            output = generate_heatmap(density, frame)

            st.metric("Crowd Count", int(count))
            st.write(f"Model Used: {model_used}")

            if count > ALERT_THRESHOLD and not alert_sent:
                send_alert(count)
                st.error("ALERT SENT: Crowd Limit Exceeded")
                alert_sent = True

            frame_window.image(output, channels="BGR")

        cap.release()
