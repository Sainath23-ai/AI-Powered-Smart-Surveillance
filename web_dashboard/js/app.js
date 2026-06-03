/* ───────────────────────────────────────────────────────── */
/* SafeGuard AI – Dashboard JavaScript                       */
/* ───────────────────────────────────────────────────────── */

const BACKEND_PORT = '5000';
const API = (() => {
  const loc = window.location;
  if (loc.protocol === 'file:') {
    return `http://127.0.0.1:${BACKEND_PORT}/api`;
  }
  const host = loc.hostname || '127.0.0.1';
  if (loc.protocol === 'http:' || loc.protocol === 'https:') {
    if (loc.port === BACKEND_PORT || loc.host.endsWith(`:${BACKEND_PORT}`)) {
      return `${loc.protocol}//${loc.host}/api`;
    }
  }
  return `http://${host}:${BACKEND_PORT}/api`;
})();
const API_ORIGIN = API.replace(/\/api\/?$/, '');
const captureUrl = path => (path.startsWith('http') ? path : `${API_ORIGIN}${path}`);

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text == null ? '' : String(text);
  return d.innerHTML;
}
let systemRunning = false;
let startTime = null;
let uptimeTimer = null;
let sseSource = null;
let currentFilter = 'all';
let selectModeActive = false;
let selectedIncidentIds = new Set();
let renderedIncidentsList = [];

// ── DOM References ─────────────────────────────────────────
const $ = id => document.getElementById(id);
const el = (sel, ctx = document) => ctx.querySelector(sel);

// ── Clock ──────────────────────────────────────────────────
function updateClock() {
  $('clock').textContent = new Date().toLocaleTimeString('en-IN', { hour12: false });
}
setInterval(updateClock, 1000);
updateClock();

// ── Navigation ─────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    const page = item.dataset.page;
    navigateTo(page);
  });
});

function navigateTo(page) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const navEl = document.querySelector(`[data-page="${page}"]`);
  const pageEl = $(`page-${page}`);
  if (navEl) navEl.classList.add('active');
  if (pageEl) pageEl.classList.add('active');
  const titles = {
    dashboard: 'Dashboard',
    live: 'Live Monitor',
    history: 'Suspicious History',
    thieves: 'Thief Database',
    alerts: 'Alert History',
    settings: 'Settings',
    esp32: 'ESP32 Control'
  };
  $('page-title').textContent = titles[page] || page;
  if (page === 'alerts') loadAlerts();
  if (page === 'history') loadIncidents();
  if (page === 'thieves') loadThieves();
  if (page === 'esp32') loadESP32Status();
  if (page === 'settings') loadSettings();
}

// ── System Start / Stop ────────────────────────────────────
$('start-btn').addEventListener('click', startSystem);
$('stop-btn').addEventListener('click', stopSystem);
$('start-btn-live')?.addEventListener('click', startSystem);

async function startSystem() {
  try {
    const res = await fetch(`${API}/start`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `Backend returned ${res.status}`);
    }
    if (data.status === 'started' || data.status === 'already_running') {
      disconnectSSE();
      systemRunning = true;
      startTime = Date.now();
      setSystemRunning(true);
      connectSSE();
      startStatusPoll();
      fetchStatus();
      showToast('Monitoring started');
    } else {
      throw new Error(data.error || 'Backend did not start monitoring');
    }
  } catch (e) {
    showToast(`Backend connection failed: ${e.message}`, 'error');
  }
}

async function stopSystem() {
  try {
    await fetch(`${API}/stop`, { method: 'POST' });
  } catch { /* backend may be offline */ }

  systemRunning = false;
  disconnectSSE();
  clearInterval(uptimeTimer);
  uptimeTimer = null;
  startTime = null;

  setSystemRunning(false);
  resetDashboardIdle();

  const uptime = $('stat-uptime');
  if (uptime) uptime.textContent = '--:--:--';
  const sysDot = $('sys-dot');
  if (sysDot) sysDot.className = 'status-dot offline';
  const sysLabel = $('sys-label');
  if (sysLabel) sysLabel.textContent = 'System Offline';

  const liveDot = $('live-indicator');
  if (liveDot) liveDot.className = 'live-dot';
  const liveText = $('live-status-text');
  if (liveText) liveText.textContent = 'Camera Offline';

  const feed = $('video-feed');
  if (feed) {
    feed.removeAttribute('src');
    feed.classList.add('hidden');
  }
  const placeholder = $('video-placeholder');
  if (placeholder) placeholder.style.display = 'flex';

  const threatBar = $('threat-bar');
  if (threatBar) threatBar.style.display = 'none';
  $('threat-overlay')?.classList.add('hidden');

  startStatusPoll();
  loadIncidents();
  showToast('Monitoring stopped');
}

function disconnectSSE() {
  if (!sseSource) return;
  sseSource.onerror = null;
  sseSource.onmessage = null;
  sseSource.close();
  sseSource = null;
}

function setSystemRunning(running) {
  const startBtn = $('start-btn');
  const stopBtn = $('stop-btn');
  if (startBtn) startBtn.classList.toggle('hidden', running);
  if (stopBtn) stopBtn.classList.toggle('hidden', !running);

  if (running) {
    const sysDot = $('sys-dot');
    if (sysDot) sysDot.className = 'status-dot online';
    const sysLabel = $('sys-label');
    if (sysLabel) sysLabel.textContent = 'System Active';
    startUptimeTimer();

    const feed = $('video-feed');
    if (feed) {
      feed.src = `${API}/stream?ts=${Date.now()}`;
      feed.classList.remove('hidden');
    }
    const placeholder = $('video-placeholder');
    if (placeholder) placeholder.style.display = 'none';
    const liveDot = $('live-indicator');
    if (liveDot) liveDot.className = 'live-dot active';
    const liveText = $('live-status-text');
    if (liveText) liveText.textContent = 'Live – AI Monitoring';
  }
}

function startUptimeTimer() {
  clearInterval(uptimeTimer);
  uptimeTimer = setInterval(() => {
    if (!startTime) return;
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    $('stat-uptime').textContent = `${h}:${m}:${s}`;
  }, 1000);
}

// ── Status Polling ─────────────────────────────────────────
let pollInterval = null;

function startStatusPoll() {
  clearInterval(pollInterval);
  const interval = systemRunning ? 1500 : 4000;
  pollInterval = setInterval(() => {
    fetchStatus();
    if (!systemRunning) loadIncidents();
  }, interval);
  fetchStatus();
  loadIncidents();
}

async function fetchStatus() {
  try {
    const res = await fetch(`${API}/status`, { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();
    if (!data.running && systemRunning) {
      systemRunning = false;
      setSystemRunning(false);
    }
    updateDashboard(data);
  } catch { /* backend offline */ }
}

const DETECTION_BAR_IDS = [
  'violence', 'gesture', 'loiter', 'panic', 'thief', 'facecover', 'sharp',
  'pose-fighting', 'pose-kidnapping', 'pose-harassment', 'pose-chasing',
  'pose-child-fall', 'pose-weapon-carry', 'pose-crowd-anomaly'
];

function resetDashboardIdle() {
  DETECTION_BAR_IDS.forEach(id => updateBar(id, 0, false));
  updateThreatPill('none');
  const thiefName = $('thief-match-name');
  if (thiefName) thiefName.textContent = '';
  const sharpLabel = $('sharp-object-label');
  if (sharpLabel) sharpLabel.textContent = '';
  const fps = $('stat-fps');
  if (fps) fps.textContent = '0';
  const persons = $('stat-persons');
  if (persons) persons.textContent = '0';
  const livePersons = $('live-persons');
  if (livePersons) livePersons.textContent = '0';
  const container = $('last-alert-content');
  if (container) {
    container.className = 'no-alert';
    container.textContent = 'System stopped – press Start to monitor';
  }
}

function updateDashboard(data) {
  if (!data) return;

  if ($('history-badge') && data.incidents_count != null) {
    $('history-badge').textContent = data.incidents_count;
  }
  if ($('alert-badge')) $('alert-badge').textContent = data.total_alerts || 0;

  if (!data.running) {
    if ($('stat-fps')) $('stat-fps').textContent = '0';
    return;
  }

  // Stats
  $('stat-fps').textContent = data.fps || 0;
  $('stat-alerts').textContent = data.alerts_today || 0;
  $('video-fps').textContent = `${data.fps || 0} FPS`;
  $('alert-badge').textContent = data.total_alerts || 0;
  if ($('history-badge')) $('history-badge').textContent = data.incidents_count ?? $('history-badge').textContent;

  const det = data.current_detections || {};
  const act = det.activity || {};
  const pose = det.pose || {};
  const gest = det.gesture || {};
  const face = det.face || {};
  const obj = det.object || {};

  // Violence
  const violenceScore = act.violence?.score || 0;
  updateBar('violence', violenceScore, act.violence?.detected);
  $('live-persons').textContent = act.persons_count || 0;
  $('stat-persons').textContent = act.persons_count || 0;
  $('live-motion').textContent = ((act.motion?.ratio || 0) * 100).toFixed(1) + '%';
  $('live-flow').textContent = (act.motion?.flow_magnitude || 0).toFixed(1);

  // Gesture
  const gestScore = gest.confidence || 0;
  updateBar('gesture', gestScore, gest.detected);
  $('live-hands').textContent = gest.hands_detected || 0;

  // Loitering
  const loitScore = act.loitering?.confidence || 0;
  updateBar('loiter', loitScore, act.loitering?.detected);

  // Panic
  const panicScore = act.running_panic?.confidence || 0;
  updateBar('panic', panicScore, act.running_panic?.detected);

  // Thief face match
  const thief = face.thief_match || {};
  updateBar('thief', thief.confidence || 0, thief.detected);
  const thiefNameEl = $('thief-match-name');
  if (thiefNameEl) {
    thiefNameEl.textContent = thief.detected ? `Match: ${thief.name || 'Unknown'}` : '';
  }

  // Face covered
  const cover = face.face_cover || {};
  updateBar('facecover', cover.confidence || 0, cover.detected);

  // Sharp object
  const sharp = obj.sharp_object || {};
  updateBar('sharp', sharp.confidence || 0, sharp.detected);
  const sharpLabel = $('sharp-object-label');
  if (sharpLabel) {
    sharpLabel.textContent = sharp.detected && sharp.label ? `Detected: ${sharp.label}` : '';
  }

  // Pose safety scenarios
  const scenarios = pose.scenarios || {};
  const poseMap = {
    fighting: 'pose-fighting',
    kidnapping: 'pose-kidnapping',
    harassment: 'pose-harassment',
    chasing: 'pose-chasing',
    child_fall: 'pose-child-fall',
    weapon_carry: 'pose-weapon-carry',
    crowd_anomaly: 'pose-crowd-anomaly'
  };
  Object.entries(poseMap).forEach(([key, id]) => {
    const scenario = scenarios[key] || {};
    updateBar(id, scenario.confidence || 0, scenario.detected);
  });

  // Threat level
  const level = data.threat_level || 'none';
  updateThreatPill(level);

  // Last alert
  if (data.last_alert) {
    updateLastAlert(data.last_alert);
  } else if (data.threat_level === 'none') {
    const container = $('last-alert-content');
    if (container && !container.querySelector('.alert-entry')) {
      container.className = 'no-alert';
      container.textContent = 'No alerts yet – system is monitoring';
    }
  }

  // ESP32
  const esp = data.esp32 || {};
  updateESPStatus(esp.connected, esp.host, esp.port);
}

function updateBar(id, score, detected) {
  const bar = $(`bar-${id}`);
  const badge = $(`badge-${id}`);
  const scoreEl = $(`score-${id}`);
  const card = $(`dc-${id}`);
  if (!bar || !badge || !scoreEl) return;
  const pct = Math.round(score * 100);

  bar.style.width = `${pct}%`;
  scoreEl.textContent = `${pct}.0%`;

  if (detected) {
    bar.className = 'detect-bar alert';
    badge.textContent = id === 'gesture' ? 'DETECTED' : 'ALERT';
    badge.className = 'detect-badge alert';
    if (card) card.className = 'detect-card alert';
  } else if (score > 0.4) {
    bar.className = 'detect-bar warning';
    badge.className = 'detect-badge warn';
    badge.textContent = 'MONITOR';
    if (card) card.className = 'detect-card warning';
  } else {
    bar.className = 'detect-bar';
    badge.className = 'detect-badge safe';
    badge.textContent = id === 'gesture' ? 'CLEAR' : (id === 'thief' ? 'CLEAR' : 'SAFE');
    if (card) card.className = 'detect-card';
  }
}

function updateThreatPill(level) {
  const pill = $('threat-pill');
  const text = $('threat-text');
  const bar = $('threat-bar');
  if (!pill || !text) return;
  pill.className = 'threat-pill';
  if (level === 'critical') {
    pill.classList.add('critical');
    text.textContent = '🚨 Critical Threat';
    if (bar) { bar.style.display = 'block'; $('threat-bar-text').textContent = '🚨 CRITICAL THREAT DETECTED – Alerts Sent!'; }
  } else if (level === 'warning') {
    pill.classList.add('warning');
    text.textContent = '⚠ Suspicious Activity';
    if (bar) { bar.style.display = 'block'; $('threat-bar-text').textContent = '⚠ Suspicious Activity Detected'; }
  } else {
    text.textContent = '✅ No Threats';
    if (bar) bar.style.display = 'none';
  }
}

function updateLastAlert(alert) {
  const container = $('last-alert-content');
  if (!container) return;
  container.className = '';
  container.innerHTML = `
    <div class="alert-entry">
      <div class="alert-level-dot ${alert.level}"></div>
      <div class="alert-entry-body">
        <div class="alert-entry-type">${alert.type}</div>
        <div class="alert-entry-meta">Detected at ${alert.time}</div>
      </div>
      <div class="alert-entry-conf">${Math.round(alert.confidence * 100)}%</div>
    </div>`;
}

// ── SSE Real-time Events ───────────────────────────────────
function connectSSE() {
  disconnectSSE();
  if (!systemRunning) return;
  sseSource = new EventSource(`${API}/events`);
  sseSource.onmessage = e => {
    try {
      const ev = JSON.parse(e.data);
      if (ev.type === 'threat_detected') {
        const payload = ev.data || ev;
        showThreatOverlay(payload);
        fetchStatus();
        loadIncidents();
        if (payload.level) updateThreatPill(payload.level);
        if (payload.type) {
          updateLastAlert({
            type: payload.type,
            confidence: payload.confidence,
            level: payload.level,
            time: new Date().toLocaleTimeString('en-IN', { hour12: false })
          });
        }
      }
    } catch {}
  };
  sseSource.onerror = () => {
    if (!systemRunning) {
      disconnectSSE();
      return;
    }
    disconnectSSE();
    setTimeout(() => {
      if (systemRunning) connectSSE();
    }, 3000);
  };
}

// ── Threat Overlay ─────────────────────────────────────────
function showThreatOverlay(data) {
  $('threat-modal-title').textContent = data.level === 'critical' ? '🚨 CRITICAL THREAT!' : '⚠ THREAT DETECTED';
  $('threat-modal-body').textContent = `${data.type} – Snapshot saved. Emergency alerts dispatched!`;
  $('threat-modal-conf').textContent = `Confidence: ${Math.round((data.confidence || 0) * 100)}%`;
  $('threat-modal-time').textContent = `Time: ${new Date().toLocaleTimeString()}`;
  const imgEl = $('threat-modal-image');
  if (data.image) {
    imgEl.src = captureUrl(`/api/captures/image/${data.image}`);
    imgEl.classList.remove('hidden');
  } else {
    imgEl.classList.add('hidden');
    imgEl.removeAttribute('src');
  }
  $('threat-overlay').classList.remove('hidden');
}
$('dismiss-overlay-btn').addEventListener('click', () => {
  $('threat-overlay').classList.add('hidden');
});

// ── ESP32 Status ───────────────────────────────────────────
function updateESPStatus(connected, host, port) {
  const dot = $('esp-dot');
  const label = $('esp-label');
  const cardDot = $('esp-card-dot');
  const cardStatus = $('esp-card-status');
  const hostLabel = $('esp-host-label');
  if (connected) {
    dot.className = 'status-dot online';
    label.textContent = 'ESP32 Online';
    if (cardDot) cardDot.className = 'status-dot online';
    if (cardStatus) cardStatus.textContent = 'Connected';
  } else {
    dot.className = 'status-dot offline';
    label.textContent = 'ESP32 Offline';
    if (cardDot) cardDot.className = 'status-dot offline';
    if (cardStatus) cardStatus.textContent = 'Not Connected';
  }
  if (hostLabel && host) hostLabel.textContent = `Host: ${host}:${port || 80}`;
}

async function loadESP32Status() {
  try {
    const res = await fetch(`${API}/esp32/status`);
    const data = await res.json();
    updateESPStatus(data.connected, data.host, data.port);
    if ($('esp-host')) $('esp-host').value = data.host || '';
    if ($('esp-port')) $('esp-port').value = data.port || 80;
    if ($('esp-enabled')) $('esp-enabled').checked = data.enabled !== false;
    renderCommandLog(data.last_commands || []);
  } catch(e) {}
}

function renderCommandLog(cmds) {
  const log = $('esp-cmd-log');
  if (!cmds.length) { log.innerHTML = '<div class="log-empty">No commands sent yet</div>'; return; }
  log.innerHTML = cmds.map(c => `
    <div class="log-entry">
      <span class="log-time">${c.timestamp}</span>
      <span class="log-cmd">${c.command}</span>
      <span class="${c.success ? 'log-ok' : 'log-fail'}">${c.success ? '✓' : '✗'}</span>
    </div>`).join('');
}

// ── ESP32 Control Buttons ──────────────────────────────────
function esp32Action(action, params = {}) {
  return fetch(`${API}/esp32/action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, ...params })
  }).then(r => r.json()).then(d => { loadESP32Status(); return d; });
}

$('ping-esp-btn')?.addEventListener('click', () => esp32Action('ping').then(() => showToast('Ping sent!')));
$('esp-alarm-btn')?.addEventListener('click', () => esp32Action('alarm').then(() => showToast('Alarm triggered!')));
$('esp-critical-btn')?.addEventListener('click', () => esp32Action('alarm').then(() => showToast('Critical alarm sent!')));
$('esp-silence-btn')?.addEventListener('click', () => esp32Action('silence').then(() => showToast('Alarm silenced!')));
$('esp-led-normal-btn')?.addEventListener('click', () => esp32Action('led', { status: 'normal' }).then(() => showToast('LED → Normal')));
$('esp-led-warn-btn')?.addEventListener('click', () => esp32Action('led', { status: 'warning' }).then(() => showToast('LED → Warning')));
$('esp-led-crit-btn')?.addEventListener('click', () => esp32Action('led', { status: 'critical' }).then(() => showToast('LED → Critical')));

$('save-esp-btn')?.addEventListener('click', async () => {
  const cfg = {
    esp32: {
      host: $('esp-host').value.trim(),
      port: parseInt($('esp-port').value) || 80,
      enabled: $('esp-enabled').checked
    }
  };
  await fetch(`${API}/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cfg) });
  showToast('ESP32 settings saved!');
  loadESP32Status();
});

$('test-alarm-btn')?.addEventListener('click', () => esp32Action('alarm').then(() => showToast('Test alarm sent to ESP32!')));
$('silence-btn')?.addEventListener('click', () => esp32Action('silence').then(() => showToast('Alarm silenced!')));

// ── Snapshot ───────────────────────────────────────────────
$('snapshot-btn')?.addEventListener('click', () => {
  const feed = $('video-feed');
  if (!feed || feed.classList.contains('hidden')) { showToast('Start system first', 'error'); return; }
  const canvas = document.createElement('canvas');
  canvas.width = feed.naturalWidth || 640;
  canvas.height = feed.naturalHeight || 480;
  canvas.getContext('2d').drawImage(feed, 0, 0);
  const link = document.createElement('a');
  link.download = `snapshot_${Date.now()}.jpg`;
  link.href = canvas.toDataURL('image/jpeg', 0.9);
  link.click();
  showToast('Snapshot saved!');
});

// ── Settings ───────────────────────────────────────────────
async function loadSettings() {
  try {
    const res = await fetch(`${API}/config`);
    const cfg = await res.json();
    if (cfg.gmail) {
      $('gmail-sender').value = cfg.gmail.sender_email || '';
      $('gmail-password').value = cfg.gmail.sender_password || '';
      $('gmail-recipient').value = cfg.gmail.recipient_email || '';
    }
    if (cfg.phone) {
      $('twilio-sid').value = cfg.phone.twilio_account_sid || '';
      $('twilio-token').value = cfg.phone.twilio_auth_token || '';
      $('twilio-from').value = cfg.phone.twilio_phone_number || '';
      $('twilio-to').value = cfg.phone.recipient_phone_number || '';
    }
    if (cfg.detection) {
      $('violence-thresh').value = Math.round((cfg.detection.violence_threshold || 0.75) * 100);
      $('gesture-conf').value = Math.round((cfg.detection.gesture_confidence || 0.80) * 100);
      if ($('pose-thresh'))       $('pose-thresh').value = Math.round((cfg.detection.pose_safety_threshold || 0.68) * 100);
      $('cooldown').value = cfg.detection.alert_cooldown_seconds || 30;
      $('sensitivity').value = cfg.detection.suspicious_sensitivity || 'medium';
      if ($('face-match-thresh')) $('face-match-thresh').value = Math.round((cfg.detection.face_match_threshold || 0.45) * 100);
      if ($('face-cover-thresh')) $('face-cover-thresh').value = Math.round((cfg.detection.face_cover_threshold || 0.55) * 100);
      if ($('object-conf')) $('object-conf').value = Math.round((cfg.detection.object_confidence || 0.45) * 100);
      $('vt-val').textContent = $('violence-thresh').value + '%';
      $('gc-val').textContent = $('gesture-conf').value + '%';
      if ($('pt-val')) $('pt-val').textContent = $('pose-thresh').value + '%';
      $('cd-val').textContent = $('cooldown').value + 's';
      if ($('fm-val')) $('fm-val').textContent = $('face-match-thresh').value + '%';
      if ($('fc-val')) $('fc-val').textContent = $('face-cover-thresh').value + '%';
      if ($('oc-val')) $('oc-val').textContent = $('object-conf').value + '%';
    }
    if (cfg.camera) {
      $('cam-source').value = cfg.camera.source !== undefined ? cfg.camera.source : 0;
      $('cam-width').value = cfg.camera.width || 640;
      $('cam-height').value = cfg.camera.height || 480;
    }
  } catch(e) { showToast('Could not load settings', 'error'); }
}

// Range slider live labels
['violence-thresh', 'gesture-conf', 'pose-thresh', 'cooldown', 'face-match-thresh', 'face-cover-thresh', 'object-conf'].forEach(id => {
  const el = $(id);
  if (!el) return;
  el.addEventListener('input', () => {
    if (id === 'violence-thresh') $('vt-val').textContent = el.value + '%';
    if (id === 'gesture-conf') $('gc-val').textContent = el.value + '%';
    if (id === 'pose-thresh') $('pt-val').textContent = el.value + '%';
    if (id === 'cooldown') $('cd-val').textContent = el.value + 's';
    if (id === 'face-match-thresh') $('fm-val').textContent = el.value + '%';
    if (id === 'face-cover-thresh') $('fc-val').textContent = el.value + '%';
    if (id === 'object-conf') $('oc-val').textContent = el.value + '%';
  });
});

$('save-settings-btn')?.addEventListener('click', async () => {
  const cfg = {
    gmail: {
      sender_email: $('gmail-sender').value.trim(),
      sender_password: $('gmail-password').value,
      recipient_email: $('gmail-recipient').value.trim()
    },
    phone: {
      twilio_account_sid: $('twilio-sid').value.trim(),
      twilio_auth_token: $('twilio-token').value.trim(),
      twilio_phone_number: $('twilio-from').value.trim(),
      recipient_phone_number: $('twilio-to').value.trim()
    },
    detection: {
      violence_threshold: parseInt($('violence-thresh').value) / 100,
      gesture_confidence: parseInt($('gesture-conf').value) / 100,
      pose_safety_threshold: parseInt($('pose-thresh').value) / 100,
      suspicious_sensitivity: $('sensitivity').value,
      alert_cooldown_seconds: parseInt($('cooldown').value),
      face_match_threshold: $('face-match-thresh') ? parseInt($('face-match-thresh').value) / 100 : 0.45,
      face_cover_threshold: $('face-cover-thresh') ? parseInt($('face-cover-thresh').value) / 100 : 0.55,
      object_confidence: $('object-conf') ? parseInt($('object-conf').value) / 100 : 0.45
    },
    camera: {
      source: $('cam-source').value,
      width: parseInt($('cam-width').value) || 640,
      height: parseInt($('cam-height').value) || 480
    }
  };
  try {
    await fetch(`${API}/config`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg)
    });
    $('save-result').textContent = '✅ Settings saved successfully!';
    setTimeout(() => { $('save-result').textContent = ''; }, 4000);
    showToast('Settings saved!');
  } catch(e) {
    showToast('Save failed – is backend running?', 'error');
  }
});

$('test-email-btn')?.addEventListener('click', async () => {
  $('test-email-result').textContent = '⏳ Sending...';
  try {
    const res = await fetch(`${API}/test-alert`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threat_type: 'Test Email Alert' })
    });
    const d = await res.json();
    $('test-email-result').textContent = d.success ? '✅ Email sent!' : `❌ ${d.message}`;
  } catch { $('test-email-result').textContent = '❌ Backend not reachable'; }
  setTimeout(() => { $('test-email-result').textContent = ''; }, 5000);
});

// ── Twilio Test SMS ─────────────────────────────────────────
$('test-sms-btn')?.addEventListener('click', async () => {
  $('test-sms-result').textContent = '⏳ Sending...';
  try {
    const res = await fetch(`${API}/test-sms`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threat_type: 'Test SMS Alert' })
    });
    const d = await res.json();
    $('test-sms-result').textContent = d.success ? `✅ SMS sent!` : `❌ ${d.message || 'SMS failed'}`;
  } catch {
    $('test-sms-result').textContent = '❌ Backend not reachable';
  }
  setTimeout(() => { $('test-sms-result').textContent = ''; }, 5000);
});

// ── Twilio Test Call ───────────────────────────────────────
$('test-call-btn')?.addEventListener('click', async () => {
  $('test-call-result').textContent = '⏳ Calling...';
  try {
    const res = await fetch(`${API}/test-call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threat_type: 'Test Call Alert' })
    });
    const d = await res.json();
    $('test-call-result').textContent = d.success ? `✅ Call started!` : `❌ ${d.message || 'Call failed'}`;
  } catch {
    $('test-call-result').textContent = '❌ Backend not reachable';
  }
  setTimeout(() => { $('test-call-result').textContent = ''; }, 6000);
});

// ── Thief Database ─────────────────────────────────────────
async function loadThieves() {
  try {
    const res = await fetch(`${API}/thieves`);
    const thieves = await res.json();
    if ($('thief-badge')) $('thief-badge').textContent = thieves.length;
    renderThieves(thieves);
  } catch {
    const list = $('thieves-list');
    if (list) list.innerHTML = '<div class="empty-state"><p>Could not load thief database</p></div>';
  }
}

function renderThieves(thieves) {
  const list = $('thieves-list');
  if (!list) return;
  if (!thieves.length) {
    list.innerHTML = '<div class="empty-state"><div class="empty-icon">🕵️</div><p>No thieves enrolled yet</p></div>';
    return;
  }
  list.innerHTML = thieves.map(t => `
    <div class="thief-row">
      <img class="thief-thumb" src="${captureUrl(t.photo_url || `/api/thieves/photo/${t.id}`)}" alt="${t.name}"/>
      <div class="thief-row-body">
        <div class="thief-row-name">${t.name}${t.alias ? ` <span class="thief-alias">(${t.alias})</span>` : ''}</div>
        <div class="thief-row-meta">Enrolled ${t.enrolled_at || ''}</div>
        ${t.crime_details ? `<div class="thief-row-crime">${t.crime_details}</div>` : ''}
        ${t.notes ? `<div class="thief-row-notes">${t.notes}</div>` : ''}
      </div>
      <button type="button" class="btn btn-danger btn-sm" data-delete-thief="${t.id}">Delete</button>
    </div>`).join('');

  list.querySelectorAll('[data-delete-thief]').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (!confirm('Remove this profile from the database?')) return;
      await fetch(`${API}/thieves/${btn.dataset.deleteThief}`, { method: 'DELETE' });
      showToast('Profile removed');
      loadThieves();
    });
  });
}

$('thief-enroll-form')?.addEventListener('submit', async e => {
  e.preventDefault();
  const result = $('thief-enroll-result');
  const file = $('thief-photo').files[0];
  if (!file) {
    result.textContent = '❌ Please select a photo';
    return;
  }
  const fd = new FormData();
  fd.append('name', $('thief-name').value.trim());
  fd.append('alias', $('thief-alias').value.trim());
  fd.append('crime_details', $('thief-crime').value.trim());
  fd.append('notes', $('thief-notes').value.trim());
  fd.append('photo', file);

  result.textContent = '⏳ Enrolling face...';
  $('thief-submit-btn').disabled = true;
  try {
    const res = await fetch(`${API}/thieves`, { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Enrollment failed');
    result.textContent = '✅ Thief profile saved!';
    $('thief-enroll-form').reset();
    loadThieves();
    showToast(`Enrolled: ${data.thief.name}`);
  } catch (err) {
    result.textContent = `❌ ${err.message}`;
  } finally {
    $('thief-submit-btn').disabled = false;
    setTimeout(() => { result.textContent = ''; }, 5000);
  }
});

// ── Suspicious History (photos) ────────────────────────────
$('refresh-history-btn')?.addEventListener('click', loadIncidents);

async function loadIncidents() {
  const grid = $('incidents-grid');
  if (!grid) return;

  grid.innerHTML = '<div class="empty-state"><div class="empty-icon">⏳</div><p>Loading suspicious activity history…</p></div>';

  try {
    let res = await fetch(`${API}/incidents?limit=100`, {
      cache: 'no-store',
      headers: { Accept: 'application/json' }
    });

    // Some environments redirect or register trailing-slash routes only.
    if (res.status === 404) {
      res = await fetch(`${API}/incidents/?limit=100`, {
        cache: 'no-store',
        headers: { Accept: 'application/json' }
      });
    }

    const contentType = res.headers.get('content-type') || '';
    let body = await res.text();

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }

    if (!contentType.includes('application/json')) {
      throw new Error('API did not return JSON — restart backend and open http://127.0.0.1:5000');
    }

    const incidents = JSON.parse(body);
    if (!Array.isArray(incidents)) {
      if (incidents && Array.isArray(incidents.incidents)) {
        renderIncidents(incidents.incidents);
        return;
      }
      throw new Error('Invalid incidents data');
    }

    if ($('history-badge')) $('history-badge').textContent = incidents.length;
    renderIncidents(incidents);
  } catch (err) {
    grid.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div>
      <p>Could not load history.</p>
      <p class="history-error-detail">${escapeHtml(err.message)}</p>
      <p class="history-error-detail">API: ${escapeHtml(API)}/incidents</p>
      <button type="button" class="btn btn-outline mt-8" id="retry-history-btn">Try again</button></div>`;
    $('retry-history-btn')?.addEventListener('click', loadIncidents);
  }
}

function renderIncidents(incidents) {
  renderedIncidentsList = incidents;
  const grid = $('incidents-grid');
  if (!grid) return;
  if (!incidents.length) {
    grid.innerHTML = '<div class="empty-state"><div class="empty-icon">📷</div><p>No suspicious captures yet — start monitoring and trigger a detection</p></div>';
    return;
  }
  grid.innerHTML = incidents.map(inc => {
    const imgSrc = inc.image_url ? captureUrl(inc.image_url) : (inc.image ? captureUrl(`/api/captures/image/${inc.image}`) : '');
    const level = inc.level === 'critical' ? 'critical' : 'warning';
    const videoLink = inc.video_url
      ? `<a href="${captureUrl(inc.video_url)}" target="_blank" rel="noopener" style="align-self: center;">▶ View clip</a>`
      : (inc.video ? `<a href="${captureUrl(`/api/captures/video/${inc.video}`)}" target="_blank" rel="noopener" style="align-self: center;">▶ View clip</a>` : '');
    const title = escapeHtml(inc.threat_type || 'Suspicious Activity');
    const ts = escapeHtml(inc.timestamp || '');
    const isChecked = selectedIncidentIds.has(inc.id);
    return `
    <article class="incident-card ${level}" data-incident-id="${inc.id}" style="${isChecked && selectModeActive ? 'border-color: var(--accent); box-shadow: 0 0 12px var(--accent-glow);' : ''}">
      <div class="incident-select-wrap ${selectModeActive ? '' : 'hidden'}">
        <input type="checkbox" class="incident-checkbox" data-id="${inc.id}" ${isChecked ? 'checked' : ''}/>
      </div>
      <img class="incident-thumb" src="${imgSrc}" alt="${title}" loading="lazy"
           onerror="this.src='';this.alt='Image unavailable';this.classList.add('thumb-missing')"/>
      <div class="incident-body">
        <div class="incident-title">${title}</div>
        <div class="incident-meta">
          <span>${ts}</span>
          <span>Confidence: ${Math.round((inc.confidence || 0) * 100)}%</span>
          <span>Level: ${escapeHtml(inc.level || 'warning')}</span>
        </div>
        <div class="incident-actions" style="align-items: center;">
          ${imgSrc ? `<a href="${imgSrc}" target="_blank" rel="noopener" style="align-self: center;">Open photo</a>` : ''}
          ${videoLink}
          <button type="button" class="btn btn-danger btn-sm" data-delete-incident="${inc.id}" style="margin-left: auto; ${selectModeActive ? 'display: none !important;' : ''}">Delete</button>
        </div>
      </div>
    </article>`;
  }).join('');

  grid.querySelectorAll('[data-delete-incident]').forEach(btn => {
    btn.addEventListener('click', async () => {
      if (!confirm('Remove this capture from suspicious history?')) return;
      try {
        const res = await fetch(`${API}/incidents/${btn.dataset.deleteIncident}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Delete request failed');
        showToast('Capture removed');
        loadIncidents();
      } catch (err) {
        showToast(err.message, 'error');
      }
    });
  });

  // Handle Select Checkbox changes
  grid.querySelectorAll('.incident-checkbox').forEach(cb => {
    cb.addEventListener('change', e => {
      const id = cb.dataset.id;
      if (cb.checked) {
        selectedIncidentIds.add(id);
      } else {
        selectedIncidentIds.delete(id);
      }
      updateSelectButtonStates();
      
      // Update visual border of the parent card dynamically
      const card = cb.closest('.incident-card');
      if (card) {
        if (cb.checked) {
          card.style.borderColor = 'var(--accent)';
          card.style.boxShadow = '0 0 12px var(--accent-glow)';
        } else {
          card.style.borderColor = '';
          card.style.boxShadow = '';
        }
      }
    });
  });

  // Support clicking the entire card to toggle selection
  grid.querySelectorAll('.incident-card').forEach(card => {
    card.addEventListener('click', e => {
      if (!selectModeActive) return;
      if (e.target.closest('a') || e.target.closest('button')) return;
      
      const cb = card.querySelector('.incident-checkbox');
      if (cb && e.target !== cb) {
        cb.checked = !cb.checked;
        cb.dispatchEvent(new Event('change'));
      }
    });
  });
}

// ── Alerts Page ────────────────────────────────────────────
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    loadAlerts();
  });
});

async function loadAlerts() {
  try {
    const res = await fetch(`${API}/alerts?limit=100`);
    const alerts = await res.json();
    renderAlerts(alerts);
  } catch { $('alerts-list').innerHTML = '<div class="empty-state"><div class="empty-icon">⚠️</div><p>Could not load alerts – is the backend running?</p></div>'; }
}

const METHOD_ICONS = { email: '📧', sms: '📱', call: '📞' };
function renderAlerts(alerts) {
  const list = $('alerts-list');
  const filtered = currentFilter === 'all' ? alerts : alerts.filter(a => a.method === currentFilter);
  if (!filtered.length) {
    list.innerHTML = '<div class="empty-state"><div class="empty-icon">🔔</div><p>No alerts recorded yet</p></div>';
    return;
  }
  list.innerHTML = filtered.map(a => `
    <div class="alert-row ${a.threat_type?.toLowerCase().includes('violence') ? 'critical' : 'warning'}">
      <span class="alert-method">${METHOD_ICONS[a.method] || '🔔'}</span>
      <div class="alert-row-body">
        <div class="alert-row-title">${a.threat_type || 'Unknown Threat'}</div>
        <div class="alert-row-sub">${a.timestamp} · via ${a.method?.toUpperCase()}</div>
        <div class="alert-row-message">${escapeHtml(a.message || '')}</div>
      </div>
      <span class="alert-row-status ${a.success ? 'ok' : 'fail'}">${a.success ? 'SENT' : 'FAILED'}</span>
    </div>`).join('');
}

// ── Multiple Select Controls ────────────────────────────────
function updateSelectButtonStates() {
  const count = selectedIncidentIds.size;
  const delBtn = $('delete-selected-history-btn');
  const countEl = $('selected-count');
  if (countEl) countEl.textContent = count;
  
  if (delBtn) {
    delBtn.disabled = count === 0;
  }
}

function initSelectModeControls() {
  const toggleBtn = $('toggle-select-mode-btn');
  const selectAllBtn = $('select-all-history-btn');
  const deleteBtn = $('delete-selected-history-btn');

  if (!toggleBtn) return;

  toggleBtn.addEventListener('click', () => {
    selectModeActive = !selectModeActive;
    selectedIncidentIds.clear();
    
    if (selectModeActive) {
      toggleBtn.innerHTML = '❌ Exit Select Mode';
      toggleBtn.classList.remove('btn-outline');
      toggleBtn.classList.add('btn-primary');
      selectAllBtn?.classList.remove('hidden');
      deleteBtn?.classList.remove('hidden');
    } else {
      toggleBtn.innerHTML = '☑️ Select Mode';
      toggleBtn.classList.add('btn-outline');
      toggleBtn.classList.remove('btn-primary');
      selectAllBtn?.classList.add('hidden');
      deleteBtn?.classList.add('hidden');
    }
    
    renderIncidents(renderedIncidentsList);
    updateSelectButtonStates();
  });

  selectAllBtn?.addEventListener('click', () => {
    if (!renderedIncidentsList.length) return;
    
    const allSelected = renderedIncidentsList.every(inc => selectedIncidentIds.has(inc.id));
    if (allSelected) {
      selectedIncidentIds.clear();
    } else {
      renderedIncidentsList.forEach(inc => selectedIncidentIds.add(inc.id));
    }
    
    renderIncidents(renderedIncidentsList);
    updateSelectButtonStates();
  });

  deleteBtn?.addEventListener('click', async () => {
    const ids = Array.from(selectedIncidentIds);
    if (!ids.length) return;

    if (!confirm(`Delete all ${ids.length} selected captures from suspicious history?`)) return;

    deleteBtn.disabled = true;
    deleteBtn.innerHTML = '⏳ Deleting...';

    try {
      const res = await fetch(`${API}/incidents/batch-delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids })
      });

      if (!res.ok) {
        throw new Error('Batch delete request failed');
      }

      showToast(`Successfully removed ${ids.length} captures`);
      selectedIncidentIds.clear();
      
      // Auto-exit select mode
      selectModeActive = false;
      toggleBtn.innerHTML = '☑️ Select Mode';
      toggleBtn.classList.add('btn-outline');
      toggleBtn.classList.remove('btn-primary');
      selectAllBtn?.classList.add('hidden');
      deleteBtn?.classList.add('hidden');

      loadIncidents();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      deleteBtn.disabled = false;
      updateSelectButtonStates();
      deleteBtn.innerHTML = `🗑️ Delete Selected (<span id="selected-count">${selectedIncidentIds.size}</span>)`;
    }
  });
}

// ── Toast Notifications ────────────────────────────────────
function showToast(message, type = 'success') {
  const t = document.createElement('div');
  t.style.cssText = `
    position:fixed;bottom:24px;right:24px;z-index:9999;
    background:${type === 'error' ? 'var(--danger)' : 'var(--success)'};
    color:white;padding:12px 20px;border-radius:10px;
    font-size:14px;font-weight:600;
    box-shadow:0 4px 20px rgba(0,0,0,0.4);
    animation:modal-in 0.2s ease;
  `;
  t.textContent = message;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

async function ensureBackendIsThisApp() {
  try {
    const res = await fetch(`${API}/ping`, { cache: 'no-store' });
    if (!res.ok) return;
    const d = await res.json();
    if (!d || d.service !== 'SafeGuard AI') return;
  } catch { /* ignore */ }
}

// ── Analytics & Reports ────────────────────────────────────
let currentAnalyticsRange = 'day';

async function loadAnalytics(range = 'day') {
  currentAnalyticsRange = range;

  // Update toggle button states
  document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
  const activeBtn = document.querySelector(`.range-btn[data-range="${range}"]`);
  if (activeBtn) activeBtn.classList.add('active');

  try {
    const res = await fetch(`${API}/incidents/stats?range=${range}`);
    const data = await res.json();

    // Update summary stat cards
    if ($('ms-total-val'))    $('ms-total-val').textContent = data.total_incidents || 0;
    if ($('ms-critical-val')) $('ms-critical-val').textContent = data.critical_count || 0;
    if ($('ms-warning-val'))  $('ms-warning-val').textContent = data.warning_count || 0;

    // Top threat
    const types = data.threat_types || {};
    const topThreat = Object.entries(types).sort((a, b) => b[1] - a[1])[0];
    if ($('ms-threat-val')) {
      $('ms-threat-val').textContent = topThreat ? topThreat[0] : '—';
    }

    // Render photo gallery
    const gallery = $('photo-gallery');
    const photos = data.recent_photos || [];
    if ($('gallery-count')) $('gallery-count').textContent = `${photos.length} capture${photos.length !== 1 ? 's' : ''}`;

    if (!photos.length || !gallery) {
      if (gallery) gallery.innerHTML = `
        <div class="empty-state" style="grid-column:1/-1;padding:32px;">
          <div class="empty-icon">📷</div>
          <p>No captures in this time range</p>
        </div>`;
      return;
    }

    gallery.innerHTML = photos.map(p => {
      const imgSrc = p.image_url ? captureUrl(p.image_url) : '';
      const level = p.level === 'critical' ? 'critical' : 'warning';
      const conf = Math.round((p.confidence || 0) * 100);
      const ts = (p.timestamp || '').split(' ').pop() || '';
      return `
        <a class="gallery-item" href="${imgSrc}" target="_blank" rel="noopener" title="${escapeHtml(p.threat_type || '')}">
          <img class="gallery-thumb" src="${imgSrc}" alt="${escapeHtml(p.threat_type || 'Capture')}" loading="lazy"
               onerror="this.style.display='none'" />
          <div class="gallery-overlay">
            <div class="gallery-overlay-type">
              <span class="gallery-level-dot ${level}"></span>
              ${escapeHtml(p.threat_type || 'Unknown')}
            </div>
            <div class="gallery-overlay-meta">${ts} · ${conf}%</div>
          </div>
        </a>`;
    }).join('');

  } catch (err) {
    console.warn('Analytics load failed:', err);
  }
}

// Range toggle click handlers
document.querySelectorAll('.range-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    loadAnalytics(btn.dataset.range);
  });
});

// Email report button
$('send-report-btn')?.addEventListener('click', async () => {
  const resultEl = $('report-result');
  const btn = $('send-report-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Sending report...';
  if (resultEl) resultEl.textContent = '';

  try {
    const res = await fetch(`${API}/report/email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ range: currentAnalyticsRange })
    });
    const data = await res.json();
    if (data.success) {
      showToast(data.message);
      if (resultEl) resultEl.textContent = `✅ ${data.message}`;
    } else {
      showToast(data.message || 'Report failed', 'error');
      if (resultEl) resultEl.textContent = `❌ ${data.message}`;
    }
  } catch (err) {
    showToast('Backend not reachable', 'error');
    if (resultEl) resultEl.textContent = '❌ Backend not reachable';
  } finally {
    btn.disabled = false;
    btn.textContent = '📧 Email Report';
    setTimeout(() => { if (resultEl) resultEl.textContent = ''; }, 6000);
  }
});

// ── Initial Load ───────────────────────────────────────────
loadSettings();
loadThieves();
initSelectModeControls();
startStatusPoll();
ensureBackendIsThisApp();
loadAnalytics('day');
fetch(`${API}/status`, { cache: 'no-store' })
  .then(r => r.ok ? r.json() : null)
  .then(data => {
    if (data && data.running) {
      systemRunning = true;
      startTime = data.start_time ? Date.parse(data.start_time) : Date.now();
      setSystemRunning(true);
      connectSSE();
    }
  })
  .catch(() => {});

console.log('%c SafeGuard AI Dashboard Ready', 'color:#6366f1;font-size:16px;font-weight:bold');


