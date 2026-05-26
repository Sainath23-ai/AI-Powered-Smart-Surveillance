"""
Alert System - Gmail Email & Twilio SMS/Call Alerts
Sends emergency notifications when threats are detected.
Supports: Email with snapshot, SMS, Phone Call
"""

import smtplib
import ssl
import os
import time
import base64
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
import threading

logger = logging.getLogger("AlertSystem")


class AlertSystem:
    def __init__(self, config: dict):
        self.config = config
        self.last_alert_times = {}   # threat_type -> timestamp
        self.alert_log = []
        self._lock = threading.Lock()

    def _get_cooldown(self):
        return self.config.get("detection", {}).get("alert_cooldown_seconds", 30)

    def _can_alert(self, threat_type: str):
        """Enforce cooldown between alerts of same type."""
        now = time.time()
        cooldown = self._get_cooldown()
        last = self.last_alert_times.get(threat_type, 0)
        return (now - last) >= cooldown

    def _record_alert(self, threat_type: str, method: str, success: bool, message: str):
        now = time.time()
        self.last_alert_times[threat_type] = now
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "threat_type": threat_type,
            "method": method,
            "success": success,
            "message": message
        }
        with self._lock:
            self.alert_log.insert(0, entry)
            if len(self.alert_log) > 100:
                self.alert_log = self.alert_log[:100]
        logger.info(f"Alert [{method}] {threat_type}: {message}")
        return entry

    # ─── Gmail Email Alert ────────────────────────────────────────────────────

    def send_email_alert(self, threat_type: str, confidence: float,
                         snapshot_bytes: bytes = None, location: str = "Unknown"):
        """Send Gmail email alert with optional snapshot image."""
        gmail_cfg = self.config.get("gmail", {})
        sender = gmail_cfg.get("sender_email", "").strip()
        password = gmail_cfg.get("sender_password", "").strip()
        recipient = gmail_cfg.get("recipient_email", "").strip()

        if not all([sender, password, recipient]):
            return self._record_alert(threat_type, "email", False, "Gmail not configured")

        try:
            msg = MIMEMultipart("related")
            msg["Subject"] = f"🚨 ALERT: {threat_type.upper()} Detected - Women Safety System"
            msg["From"] = f"SafeGuard AI <{sender}>"
            msg["To"] = recipient

            timestamp = datetime.now().strftime("%d %B %Y, %I:%M:%S %p")
            html_body = f"""
            <html><body style="font-family:Arial,sans-serif;background:#f5f5f5;padding:20px">
            <div style="max-width:600px;margin:auto;background:white;border-radius:12px;
                        box-shadow:0 4px 20px rgba(0,0,0,0.1);overflow:hidden">
              <div style="background:linear-gradient(135deg,#c0392b,#e74c3c);padding:30px;text-align:center">
                <h1 style="color:white;margin:0;font-size:28px">🚨 EMERGENCY ALERT</h1>
                <p style="color:rgba(255,255,255,0.9);margin:8px 0 0">SafeGuard AI Surveillance System</p>
              </div>
              <div style="padding:30px">
                <div style="background:#fff3cd;border-left:5px solid #e74c3c;padding:15px;border-radius:6px;margin-bottom:20px">
                  <h2 style="color:#c0392b;margin:0 0 10px">{threat_type}</h2>
                  <p style="margin:5px 0;color:#555"><strong>Confidence:</strong> {confidence:.1%}</p>
                  <p style="margin:5px 0;color:#555"><strong>Time:</strong> {timestamp}</p>
                  <p style="margin:5px 0;color:#555"><strong>Location:</strong> {location}</p>
                </div>
                {"<h3 style='color:#333'>📸 Captured Snapshot</h3><img src='cid:snapshot' style='width:100%;border-radius:8px;border:3px solid #e74c3c'/>" if snapshot_bytes else ""}
                <div style="background:#f8f9fa;padding:15px;border-radius:8px;margin-top:20px">
                  <p style="color:#666;margin:0;font-size:13px">
                    ⚡ This alert was generated automatically by the AI Surveillance System.<br>
                    Please take immediate action and contact authorities if necessary.<br>
                    Emergency: <strong>112 (India)</strong>
                  </p>
                </div>
              </div>
            </div></body></html>
            """
            msg.attach(MIMEText(html_body, "html"))

            if snapshot_bytes:
                img = MIMEImage(snapshot_bytes)
                img.add_header("Content-ID", "<snapshot>")
                img.add_header("Content-Disposition", "inline", filename="alert_snapshot.jpg")
                msg.attach(img)

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())

            return self._record_alert(threat_type, "email", True,
                                       f"Email sent to {recipient}")
        except Exception as e:
            return self._record_alert(threat_type, "email", False, str(e))

    # ─── Twilio SMS Alert ─────────────────────────────────────────────────────

    def send_sms_alert(self, threat_type: str, confidence: float, location: str = "Unknown"):
        """Send SMS alert via Twilio."""
        try:
            from twilio.rest import Client
        except ImportError:
            return self._record_alert(threat_type, "sms", False,
                                       "Twilio not installed. Run: pip install twilio")

        phone_cfg = self.config.get("phone", {})
        sid = phone_cfg.get("twilio_account_sid", "").strip()
        token = phone_cfg.get("twilio_auth_token", "").strip()
        from_num = phone_cfg.get("twilio_phone_number", "").strip()
        to_num = phone_cfg.get("recipient_phone_number", "").strip()

        if not all([sid, token, from_num, to_num]):
            return self._record_alert(threat_type, "sms", False, "Twilio not configured")

        try:
            client = Client(sid, token)
            timestamp = datetime.now().strftime("%d/%m/%Y %I:%M %p")
            body = (f"🚨 SafeGuard AI ALERT!\n"
                    f"Threat: {threat_type}\n"
                    f"Confidence: {confidence:.0%}\n"
                    f"Time: {timestamp}\n"
                    f"Location: {location}\n"
                    f"Please check immediately! Emergency: 112")
            message = client.messages.create(body=body, from_=from_num, to=to_num)
            return self._record_alert(threat_type, "sms", True,
                                       f"SMS sent (SID: {message.sid})")
        except Exception as e:
            return self._record_alert(threat_type, "sms", False, str(e))

    # ─── Twilio Voice Call Alert ──────────────────────────────────────────────

    def make_phone_call(self, threat_type: str, confidence: float):
        """Trigger an automated phone call via Twilio."""
        try:
            from twilio.rest import Client
        except ImportError:
            return self._record_alert(threat_type, "call", False,
                                       "Twilio not installed")

        phone_cfg = self.config.get("phone", {})
        sid = phone_cfg.get("twilio_account_sid", "").strip()
        token = phone_cfg.get("twilio_auth_token", "").strip()
        from_num = phone_cfg.get("twilio_phone_number", "").strip()
        to_num = phone_cfg.get("recipient_phone_number", "").strip()

        if not all([sid, token, from_num, to_num]):
            return self._record_alert(threat_type, "call", False, "Twilio not configured")

        try:
            client = Client(sid, token)
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
            <Response>
              <Say voice="alice" language="en-IN">
                Emergency Alert! Emergency Alert!
                The SafeGuard A I system has detected {threat_type}.
                Confidence level is {int(confidence * 100)} percent.
                Please take immediate action. Contact authorities if needed.
                This is an automated message from the SafeGuard A I surveillance system.
              </Say>
              <Pause length="1"/>
              <Say voice="alice">Emergency Alert repeated.</Say>
            </Response>"""

            call = client.calls.create(
                twiml=twiml, from_=from_num, to=to_num
            )
            return self._record_alert(threat_type, "call", True,
                                       f"Call initiated (SID: {call.sid})")
        except Exception as e:
            return self._record_alert(threat_type, "call", False, str(e))

    # ─── Telegram Bot Alert ───────────────────────────────────────────────────

    def send_telegram_alert(self, threat_type: str, confidence: float, snapshot_bytes: bytes = None, location: str = "Unknown"):
        """Send push notification to Telegram with optional image."""
        telegram_cfg = self.config.get("telegram", {})
        bot_token = telegram_cfg.get("bot_token", "").strip()
        chat_id = telegram_cfg.get("chat_id", "").strip()

        if not all([bot_token, chat_id]):
            return self._record_alert(threat_type, "telegram", False, "Telegram not configured")

        try:
            import requests
            timestamp = datetime.now().strftime("%d/%m/%Y %I:%M %p")
            caption = (f"🚨 *SafeGuard AI ALERT!*\n\n"
                       f"⚠️ *Threat:* {threat_type}\n"
                       f"📊 *Confidence:* {confidence:.0%}\n"
                       f"🕒 *Time:* {timestamp}\n"
                       f"📍 *Location:* {location}")

            if snapshot_bytes:
                url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                files = {"photo": ("snapshot.jpg", snapshot_bytes, "image/jpeg")}
                data = {"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"}
                response = requests.post(url, data=data, files=files, timeout=5)
            else:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                data = {"chat_id": chat_id, "text": caption, "parse_mode": "Markdown"}
                response = requests.post(url, data=data, timeout=5)

            if response.status_code == 200:
                return self._record_alert(threat_type, "telegram", True, "Telegram push notification sent")
            else:
                return self._record_alert(threat_type, "telegram", False, f"Telegram API Error: {response.text}")
        except Exception as e:
            return self._record_alert(threat_type, "telegram", False, str(e))

    # ─── Discord Webhook Alert ────────────────────────────────────────────────

    def send_discord_alert(self, threat_type: str, confidence: float, snapshot_bytes: bytes = None, location: str = "Unknown"):
        """Send plain-text alert to Discord Webhook."""
        discord_cfg = self.config.get("discord", {})
        webhook_url = discord_cfg.get("webhook_url", "").strip()

        if not webhook_url:
            return self._record_alert(threat_type, "discord", False, "Discord not configured")

        try:
            import requests
            timestamp = datetime.now().strftime("%d/%m/%Y %I:%M %p")
            content = (f"🚨 **SafeGuard AI ALERT!**\n"
                       f"⚠️ Threat: {threat_type}\n"
                       f"📊 Confidence: {confidence:.0%}\n"
                       f"🕒 Time: {timestamp}\n"
                       f"📍 Location: {location}")

            if snapshot_bytes:
                files = {"file": ("snapshot.jpg", snapshot_bytes, "image/jpeg")}
                data = {"content": content}
                response = requests.post(webhook_url, data=data, files=files, timeout=5)
            else:
                data = {"content": content}
                response = requests.post(webhook_url, json=data, timeout=5)

            if response.status_code in [200, 204]:
                return self._record_alert(threat_type, "discord", True, "Discord webhook sent")
            else:
                return self._record_alert(threat_type, "discord", False, f"Discord API Error: {response.text}")
        except Exception as e:
            return self._record_alert(threat_type, "discord", False, str(e))

    # ─── Unified Alert Dispatcher ─────────────────────────────────────────────

    def trigger_alert(self, threat_type: str, confidence: float,
                      snapshot_bytes: bytes = None, location: str = "Camera Feed"):
        """
        Main entry point. Checks cooldown then fires all configured alert methods
        in background threads to avoid blocking the video stream.
        """
        if not self._can_alert(threat_type):
            return []

        results = []

        def _send_all():
            # Email
            r = self.send_email_alert(threat_type, confidence, snapshot_bytes, location)
            results.append(r)
            # SMS
            r = self.send_sms_alert(threat_type, confidence, location)
            results.append(r)
            # Telegram
            r = self.send_telegram_alert(threat_type, confidence, snapshot_bytes, location)
            results.append(r)
            # Discord
            r = self.send_discord_alert(threat_type, confidence, snapshot_bytes, location)
            results.append(r)
            # Call only for critical threats
            if confidence >= 0.85:
                r = self.make_phone_call(threat_type, confidence)
                results.append(r)

        t = threading.Thread(target=_send_all, daemon=True)
        t.start()
        return results

    def get_alert_log(self, limit: int = 50):
        with self._lock:
            return self.alert_log[:limit]

    def update_config(self, new_config: dict):
        self.config = new_config
