"""
AI Smart Surveillance System - Configuration
Edit settings.json or use the Web Dashboard to configure.
"""

import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.json')

DEFAULT_CONFIG = {
    "gmail": {
        "sender_email": "sainathpavanv@gmail.com",
        "sender_password": "",   # Use Gmail App Password (16-char)
        "recipient_email": "sainathpavanv@gmail.com"
    },
    "phone": {
        "twilio_account_sid": "",
        "twilio_auth_token": "",
        "twilio_phone_number": "",
        "recipient_phone_number": ""
    },
    "esp32": {
        "host": "192.168.1.100",
        "port": 80,
        "enabled": True
    },
    "detection": {
        "violence_threshold": 0.75,
        "gesture_confidence": 0.80,
        "pose_safety_threshold": 0.68,
        "suspicious_sensitivity": "medium",
        "alert_cooldown_seconds": 30,
        "face_match_threshold": 0.45,
        "face_cover_threshold": 0.55,
        "object_confidence": 0.45
    },
    "camera": {
        "source": 0,
        "width": 640,
        "height": 480,
        "fps": 30
    }
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            # Merge with defaults for missing keys
            merged = DEFAULT_CONFIG.copy()
            for key in merged:
                if key in cfg:
                    merged[key].update(cfg[key])
            return merged
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    return True
