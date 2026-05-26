/*
 * SafeGuard AI - ESP32 Surveillance Node Firmware
 * ─────────────────────────────────────────────────
 * Receives HTTP commands from the Python backend to:
 *  - Trigger buzzer alarm (warning/critical patterns)
 *  - Control RGB LED indicators
 *  - Display threat info on Serial (or OLED if connected)
 *  - Respond to ping checks
 *
 * Wiring:
 *  - Buzzer     → GPIO 25
 *  - LED Red    → GPIO 26
 *  - LED Green  → GPIO 27
 *  - LED Blue   → GPIO 14
 *  (Optional) SSD1306 OLED → SDA:21, SCL:22
 *
 * Libraries needed (install via Arduino Library Manager):
 *  - WiFi (built-in)
 *  - WebServer (built-in ESP32)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// ── WiFi Credentials ─────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ── GPIO Pin Definitions ──────────────────────────────────────
#define BUZZER_PIN   25
#define LED_RED      26
#define LED_GREEN    27
#define LED_BLUE     14

// ── HTTP Server ───────────────────────────────────────────────
WebServer server(80);

// ── State ─────────────────────────────────────────────────────
bool alarmActive   = false;
String currentMode = "normal";
unsigned long alarmStartTime = 0;
unsigned long alarmDuration  = 0;  // 0 = indefinite

// ── Helpers ───────────────────────────────────────────────────
void setLED(bool r, bool g, bool b) {
  digitalWrite(LED_RED,   r ? HIGH : LOW);
  digitalWrite(LED_GREEN, g ? HIGH : LOW);
  digitalWrite(LED_BLUE,  b ? HIGH : LOW);
}

void buzzPattern(const String& level) {
  if (level == "critical") {
    // Fast urgent beeping
    for (int i = 0; i < 5; i++) {
      digitalWrite(BUZZER_PIN, HIGH); delay(100);
      digitalWrite(BUZZER_PIN, LOW);  delay(80);
    }
  } else {
    // Slow warning beep
    for (int i = 0; i < 3; i++) {
      digitalWrite(BUZZER_PIN, HIGH); delay(200);
      digitalWrite(BUZZER_PIN, LOW);  delay(150);
    }
  }
}

void addCORSHeaders() {
  server.sendHeader("Access-Control-Allow-Origin",  "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void sendJSON(int code, const String& json) {
  addCORSHeaders();
  server.send(code, "application/json", json);
}

// ── Route Handlers ────────────────────────────────────────────
void handlePing() {
  String ip = WiFi.localIP().toString();
  sendJSON(200, "{\"status\":\"ok\",\"device\":\"SafeGuard-ESP32\",\"ip\":\"" + ip + "\",\"mode\":\"" + currentMode + "\"}");
}

void handleAlarm() {
  String level = server.hasArg("level") ? server.arg("level") : "warning";
  String type  = server.hasArg("type")  ? server.arg("type")  : "Unknown";

  alarmActive    = true;
  alarmStartTime = millis();
  alarmDuration  = (level == "critical") ? 30000 : 15000;
  currentMode    = level;

  Serial.printf("[ALARM] Level=%s Type=%s\n", level.c_str(), type.c_str());

  if (level == "critical") {
    setLED(true, false, false);  // Red
  } else {
    setLED(true, true, false);   // Yellow (Red+Green)
  }

  buzzPattern(level);
  sendJSON(200, "{\"status\":\"alarm_triggered\",\"level\":\"" + level + "\"}");
}

void handleSilence() {
  alarmActive = false;
  currentMode = "normal";
  setLED(false, true, false);  // Green = safe
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("[SILENCE] Alarm silenced");
  sendJSON(200, "{\"status\":\"silenced\"}");
}

void handleLED() {
  String status = server.hasArg("status") ? server.arg("status") : "normal";
  if      (status == "critical") setLED(true,  false, false);
  else if (status == "warning")  setLED(true,  true,  false);
  else if (status == "off")      setLED(false, false, false);
  else if (status == "blue")     setLED(false, false, true);
  else                           setLED(false, true,  false);  // normal = green
  currentMode = status;
  Serial.printf("[LED] Status=%s\n", status.c_str());
  sendJSON(200, "{\"status\":\"led_set\",\"mode\":\"" + status + "\"}");
}

void handleDisplay() {
  String msg = server.hasArg("msg") ? server.arg("msg") : "";
  Serial.printf("[DISPLAY] %s\n", msg.c_str());
  // If you have an OLED, add display.print(msg) here
  sendJSON(200, "{\"status\":\"displayed\"}");
}

void handleLocation() {
  String lat  = server.hasArg("lat")  ? server.arg("lat")  : "0";
  String lon  = server.hasArg("lon")  ? server.arg("lon")  : "0";
  String addr = server.hasArg("addr") ? server.arg("addr") : "";
  Serial.printf("[LOCATION] lat=%s lon=%s addr=%s\n", lat.c_str(), lon.c_str(), addr.c_str());
  sendJSON(200, "{\"status\":\"location_received\"}");
}

void handleStatus() {
  String json = "{\"status\":\"ok\",\"alarm\":" + String(alarmActive ? "true" : "false") +
                ",\"mode\":\"" + currentMode + "\",\"ip\":\"" + WiFi.localIP().toString() + "\"}";
  sendJSON(200, json);
}

void handleNotFound() {
  sendJSON(404, "{\"error\":\"Not found\"}");
}

// ── Setup ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("\n=== SafeGuard AI ESP32 Node ===");

  // GPIO setup
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_RED,    OUTPUT);
  pinMode(LED_GREEN,  OUTPUT);
  pinMode(LED_BLUE,   OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  setLED(false, false, true);  // Blue = booting

  // WiFi
  Serial.printf("Connecting to WiFi: %s\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500); Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
    setLED(false, true, false);  // Green = ready

    // Startup chime
    for (int i = 0; i < 2; i++) {
      digitalWrite(BUZZER_PIN, HIGH); delay(80);
      digitalWrite(BUZZER_PIN, LOW);  delay(80);
    }
  } else {
    Serial.println("\nWiFi FAILED – running in offline mode");
    setLED(true, false, false);  // Red = error
  }

  // Register HTTP routes
  server.on("/ping",     HTTP_GET, handlePing);
  server.on("/alarm",    HTTP_GET, handleAlarm);
  server.on("/silence",  HTTP_GET, handleSilence);
  server.on("/led",      HTTP_GET, handleLED);
  server.on("/display",  HTTP_GET, handleDisplay);
  server.on("/location", HTTP_GET, handleLocation);
  server.on("/status",   HTTP_GET, handleStatus);
  server.onNotFound(handleNotFound);
  server.begin();

  Serial.println("HTTP Server started on port 80");
  Serial.println("Configure Python backend with this IP address.");
}

// ── Loop ──────────────────────────────────────────────────────
void loop() {
  server.handleClient();

  // Auto-silence alarm after duration
  if (alarmActive && alarmDuration > 0) {
    if (millis() - alarmStartTime >= alarmDuration) {
      alarmActive = false;
      currentMode = "normal";
      setLED(false, true, false);
      digitalWrite(BUZZER_PIN, LOW);
      Serial.println("[AUTO] Alarm auto-silenced after timeout");
    }
  }

  // Heartbeat: slow blue blink when idle
  if (!alarmActive && currentMode == "normal") {
    static unsigned long lastBlink = 0;
    static bool blinkState = false;
    if (millis() - lastBlink > 2000) {
      blinkState = !blinkState;
      digitalWrite(LED_BLUE, blinkState ? HIGH : LOW);
      lastBlink = millis();
    }
  }
}
