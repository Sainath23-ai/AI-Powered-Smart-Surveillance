"""
ESP32 Communication Module
Sends HTTP commands to the ESP32 microcontroller to:
  - Trigger buzzer/alarm
  - Flash LED indicators
  - Display threat type on OLED
  - Reset/silence alarms
"""

import requests
import logging
import time
import threading
from datetime import datetime

logger = logging.getLogger("ESP32Comm")


class ESP32Controller:
    def __init__(self, config: dict):
        self.config = config
        self.esp_cfg = config.get("esp32", {})
        self.host = self.esp_cfg.get("host", "192.168.1.100")
        self.port = self.esp_cfg.get("port", 80)
        self.enabled = self.esp_cfg.get("enabled", True)
        self.base_url = f"http://{self.host}:{self.port}"
        self.connected = False
        self.last_ping = 0
        self.command_log = []
        self._lock = threading.Lock()

    def _send(self, endpoint: str, params: dict = None, timeout: int = 3):
        """Send HTTP GET request to ESP32."""
        if not self.enabled:
            return False, "ESP32 disabled"
        try:
            url = f"{self.base_url}/{endpoint}"
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                self.connected = True
                return True, resp.text
            return False, f"HTTP {resp.status_code}"
        except requests.exceptions.ConnectTimeout:
            self.connected = False
            return False, "Connection timeout"
        except requests.exceptions.ConnectionError:
            self.connected = False
            return False, "Connection refused - Check ESP32 IP/WiFi"
        except Exception as e:
            self.connected = False
            return False, str(e)

    def _log_command(self, command: str, success: bool, response: str):
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "command": command,
            "success": success,
            "response": response
        }
        with self._lock:
            self.command_log.insert(0, entry)
            if len(self.command_log) > 50:
                self.command_log = self.command_log[:50]
        return entry

    def ping(self):
        """Check if ESP32 is reachable."""
        now = time.time()
        if now - self.last_ping < 5:
            return self.connected
        self.last_ping = now
        ok, _ = self._send("ping", timeout=2)
        self.connected = ok
        return ok

    def trigger_alarm(self, threat_level: str = "warning", threat_type: str = "Unknown"):
        """
        Activate buzzer + LEDs on ESP32.
        threat_level: 'warning' | 'critical'
        """
        params = {
            "level": threat_level,
            "type": threat_type[:20]  # ESP32 display limit
        }
        ok, resp = self._send("alarm", params)
        self._log_command(f"ALARM [{threat_level}]", ok, resp)
        if not ok:
            logger.warning(f"ESP32 alarm failed: {resp}")
        return ok

    def silence_alarm(self):
        """Stop the alarm on ESP32."""
        ok, resp = self._send("silence")
        self._log_command("SILENCE", ok, resp)
        return ok

    def set_led_status(self, status: str = "normal"):
        """
        Set LED color on ESP32.
        status: 'normal' | 'warning' | 'critical' | 'off'
        """
        ok, resp = self._send("led", {"status": status})
        self._log_command(f"LED [{status}]", ok, resp)
        return ok

    def send_location_alert(self, location_data: dict):
        """Send GPS location info to ESP32 for display."""
        params = {
            "lat": location_data.get("lat", "0.0"),
            "lon": location_data.get("lon", "0.0"),
            "addr": location_data.get("address", "Unknown")[:20]
        }
        ok, resp = self._send("location", params)
        self._log_command("LOCATION", ok, resp)
        return ok

    def display_message(self, message: str):
        """Show a short message on ESP32 OLED."""
        ok, resp = self._send("display", {"msg": message[:32]})
        self._log_command(f"DISPLAY: {message[:20]}", ok, resp)
        return ok

    def handle_threat(self, threat_type: str, threat_level: str, confidence: float):
        """
        Full threat response sequence sent to ESP32 in background thread.
        """
        def _sequence():
            self.trigger_alarm(threat_level, threat_type)
            self.display_message(f"ALERT: {threat_type[:20]}")
            if threat_level == "critical":
                self.set_led_status("critical")
                time.sleep(0.5)
                self.trigger_alarm("critical", threat_type)
            else:
                self.set_led_status("warning")

        t = threading.Thread(target=_sequence, daemon=True)
        t.start()

    def get_status(self):
        return {
            "connected": self.connected,
            "host": self.host,
            "port": self.port,
            "enabled": self.enabled,
            "base_url": self.base_url,
            "last_commands": self.command_log[:5]
        }

    def update_config(self, new_config: dict):
        self.config = new_config
        esp = new_config.get("esp32", {})
        self.host = esp.get("host", self.host)
        self.port = esp.get("port", self.port)
        self.enabled = esp.get("enabled", self.enabled)
        self.base_url = f"http://{self.host}:{self.port}"
        self.connected = False  # Reset connection on config change
