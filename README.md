# 🛡️ SafeGuard AI – Smart Surveillance System

> **AI-powered women & child safety surveillance using Computer Vision, IoT & Real-time Alerts**

---

## 📋 Features

| Feature | Description |
|---|---|
| 🎥 Live Video Monitoring | Real-time MJPEG stream with AI overlay |
| ⚔️ Violence Detection | Optical flow + motion analysis to detect aggressive behaviour |
| 🧍 Pose Safety Estimation | MediaPipe pose landmarks estimate fighting, kidnapping, harassment, chasing, falls, weapon-carry posture & crowd anomalies |
| ✋ SOS Gesture Recognition | MediaPipe hand landmarks detect "Signal for Help" gesture |
| 🕵️ Suspicious Loitering | Detects persons staying in one area too long |
| 🏃 Panic Detection | Identifies sustained rapid movement / running |
| 🕵️ Thief Face Recognition | YuNet + SFace ML matches enrolled suspect faces on camera |
| 😷 Face Cover Detection | Detects masks / covered faces (MediaPipe + vision heuristics) |
| 🔪 Sharp Object Detection | YOLOv8 detects knives and similar dangerous objects |
| 📁 Thief Database | Enroll suspect photos, case notes, and aliases from the dashboard |
| 📧 Gmail Email Alerts | Sends HTML emails with snapshot to guardian's inbox |
| 📱 SMS / Voice Call | Twilio integration for SMS and automated voice calls |
| 🔌 ESP32 Integration | Triggers physical alarm, buzzer & RGB LED via HTTP |
| 📊 Web Dashboard | Beautiful dark-mode real-time monitoring dashboard |
| ⚙️ Settings UI | Configure Gmail, phone, detection thresholds from browser |

---

## 🗂️ Project Structure

```
AI-Powered Smart Surveillance/
├── backend/
│   ├── app.py               ← Flask server (main entry point)
│   ├── config.py            ← Configuration loader/saver
│   ├── detector.py          ← Violence & activity AI detector
│   ├── gesture_detector.py  ← Hand gesture (SOS) detector
│   ├── alert_system.py      ← Gmail + Twilio alerts
│   ├── esp32_comm.py        ← ESP32 HTTP controller
│   ├── face_thief_detector.py ← Face + thief match + face-cover ML
│   ├── object_detector.py   ← YOLOv8 sharp object detection
│   ├── thief_registry.py    ← Thief profile storage
│   └── requirements.txt     ← Python dependencies
├── config/
│   └── settings.json        ← Persistent settings file
├── esp32/
│   └── surveillance_esp32/
│       └── surveillance_esp32.ino  ← Arduino firmware
└── web_dashboard/
    ├── index.html
    ├── css/style.css
    └── js/app.js
```

---

## 🚀 Quick Start

### 1. Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Backend Server
```bash
cd backend
python app.py
```

### 3. Open Dashboard
Navigate to: **http://127.0.0.1:5000**

### 4. Configure Alerts (Settings Page)
- Enter your **Gmail** address + **App Password**
- Enter **Twilio** credentials for SMS/calls
- Configure ESP32 IP address
- Click **💾 Save All Settings**

### 5. Start Monitoring
Click **▶ Start System** in the dashboard

---

## 🔧 Gmail Setup (App Password)

1. Enable **2-Step Verification** on your Google account
2. Go to: https://myaccount.google.com/apppasswords
3. Create an App Password for "Mail" on "Windows Computer"
4. Use the 16-character password in the Settings page

---

## 📱 Twilio Setup (SMS & Calls)

1. Create account at https://www.twilio.com
2. Get a phone number (free trial available)
3. Copy **Account SID**, **Auth Token**, and your **Twilio number**
4. Enter these in the Settings → Phone Alerts section

---

## 🔌 ESP32 Setup

### Hardware Connections
| Component | ESP32 GPIO |
|---|---|
| Buzzer (+) | GPIO 25 |
| LED Red | GPIO 26 |
| LED Green | GPIO 27 |
| LED Blue | GPIO 14 |

### Flash Firmware
1. Open `esp32/surveillance_esp32/surveillance_esp32.ino` in Arduino IDE
2. Install ESP32 board support and **ArduinoJson** library
3. Edit `WIFI_SSID` and `WIFI_PASSWORD` in the sketch
4. Flash to your ESP32
5. Note the IP address from Serial Monitor
6. Enter the IP in Dashboard → ESP32 Control → Settings

---

## 🤝 Hand Gesture – Signal for Help

The system detects the **international domestic violence signal for help**:

1. **Open palm** facing camera (initial distress signal)
2. **Tuck thumb** inside fist
3. **Close fingers** over thumb = 🆘 SOS Fist

Both the open palm and closed SOS fist trigger emergency alerts.

---

## 🛡️ Detection Thresholds (adjustable in Settings)

| Setting | Default | Description |
|---|---|---|
| Violence Threshold | 75% | Motion score to classify as violence |
| Gesture Confidence | 80% | Min confidence for SOS gesture |
| Pose Safety Threshold | 68% | Min confidence for pose-based safety scenarios |
| Sensitivity | Medium | Loitering/panic detection sensitivity |
| Alert Cooldown | 30s | Minimum seconds between repeat alerts |

---

## ⚡ Emergency Numbers (India)
- **Police:** 100
- **Women Helpline:** 1091
- **Emergency:** 112
- **Child Helpline:** 1098
