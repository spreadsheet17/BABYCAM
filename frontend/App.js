/**
 * BabyCam Monitor — App.js
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  View, Text, StyleSheet, StatusBar, ScrollView,
  TouchableOpacity, Animated, Vibration, Dimensions,
  Platform, TextInput, KeyboardAvoidingView,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { SafeAreaProvider, SafeAreaView, useSafeAreaInsets } from "react-native-safe-area-context";
import * as Notifications from "expo-notifications";
import * as Device from "expo-device";
import { VLCPlayer } from "react-native-vlc-media-player";

// ── Config defaults (overridden by saved settings) ───────────────────────────
const DEFAULT_CONFIG = {
  serverIp:    "192.168.1.9",
  serverPort:  "5000",
  rtspUser:    "acuitycam",
  rtspPass:    "12345678",
  rtspIp:      "192.168.1.4",
  rtspPort:    "554",
  rtspPath:    "stream1",
};
const STORAGE_KEY = "@babycam_config";

function buildBaseUrl(cfg) {
  return `http://${cfg.serverIp}:${cfg.serverPort}`;
}
function buildRtspUrl(cfg) {
  const auth = cfg.rtspUser ? `${cfg.rtspUser}:${cfg.rtspPass}@` : "";
  return `rtsp://${auth}${cfg.rtspIp}:${cfg.rtspPort}/${cfg.rtspPath}`;
}

const POLL_MS = 2000;

// ── Alert modes ───────────────────────────────────────────────────────────────
const ALERT_MODES = [
  { id: "none",     label: "None",          icon: "🔕", desc: "No alerts" },
  { id: "vibrate",  label: "Vibrate",       icon: "📳", desc: "Vibrate only" },
  { id: "sound",    label: "Sound",         icon: "🔔", desc: "Notification sound only" },
  { id: "both",     label: "Both",          icon: "🚨", desc: "Vibrate + notification sound" },
];

// ── Notification channel setup (required for Android) ────────────────────────
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge:  true,
  }),
});

async function setupNotificationChannel() {
  if (Platform.OS !== "android") return;
  await Notifications.setNotificationChannelAsync("babycam-alerts", {
    name:              "BabyCam Safety Alerts",
    importance:        Notifications.AndroidImportance.MAX,
    vibrationPattern:  [0, 400, 200, 400],
    lightColor:        "#ef4444",
    sound:             "default",
    enableVibrate:     true,
    showBadge:         true,
  });
}

async function requestNotifPermission() {
  if (!Device.isDevice) return false;
  const { status } = await Notifications.requestPermissionsAsync();
  return status === "granted";
}

async function fireAlert(position, alertMode) {
  if (alertMode === "none") return;

  const bodies = {
    PRONE:       "Baby may be face-down — check immediately!",
    SIDE_UNSAFE: "Baby rolled to an unsafe side position — check now.",
    UNKNOWN:     "Baby's position is unclear — please check the crib.",
  };

  const useSound   = alertMode === "sound" || alertMode === "both";
  const useVibrate = alertMode === "vibrate" || alertMode === "both";

  // Vibrate
  if (useVibrate) Vibration.vibrate([0, 400, 200, 400, 200, 400]);

  // Push notification
  await Notifications.scheduleNotificationAsync({
    content: {
      title:    "🚨 BabyCam Safety Alert",
      body:     bodies[position] ?? "Unsafe sleeping position detected!",
      sound:    useSound ? "default" : null,
      priority: Notifications.AndroidNotificationPriority.MAX,
      color:    "#ef4444",
    },
    trigger: {
      channelId: "babycam-alerts",   // required on Android
    },
  });
}

// ── Position definitions ──────────────────────────────────────────────────────
const POS = {
  SAFE:        { label: "Safe",          emoji: "😴", safe: true,  color: "#4ade80", bg: "#052e16", desc: "Baby is on their back" },
  PRONE:       { label: "Prone",         emoji: "🚨", safe: false, color: "#f87171", bg: "#2d0a0a", desc: "Face-down — check immediately!" },
  SIDE_UNSAFE: { label: "Side / Rolled", emoji: "⚠️", safe: false, color: "#fbbf24", bg: "#1c1206", desc: "Rolled to side — check baby" },
  UNKNOWN:     { label: "Unknown",       emoji: "❓", safe: false, color: "#fb923c", bg: "#1f1106", desc: "Position unclear — check baby" },
  NO_BABY:     { label: "No Baby",       emoji: "🔍", safe: true,  color: "#94a3b8", bg: "#0f1117", desc: "No baby detected in frame" },
};

// ── Pulse ring ────────────────────────────────────────────────────────────────
function PulseRing({ active, color }) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (active) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(anim, { toValue: 1, duration: 800, useNativeDriver: true }),
          Animated.timing(anim, { toValue: 0, duration: 800, useNativeDriver: true }),
        ])
      ).start();
    } else {
      anim.stopAnimation();
      anim.setValue(0);
    }
  }, [active]);

  if (!active) return null;
  return (
    <Animated.View style={[styles.pulseRing, {
      borderColor: color,
      transform:   [{ scale: anim.interpolate({ inputRange: [0, 1], outputRange: [1, 1.25] }) }],
      opacity:     anim.interpolate({ inputRange: [0, 1], outputRange: [0.8, 0] }),
    }]} />
  );
}

function Pill({ label, value }) {
  return (
    <View style={styles.pill}>
      <Text style={styles.pillVal}>{value}</Text>
      <Text style={styles.pillLbl}>{label}</Text>
    </View>
  );
}

// ── Monitor tab ───────────────────────────────────────────────────────────────
function MonitorTab({ status, connected }) {
  const cfg    = POS[status.position] ?? POS.UNKNOWN;
  const unsafe = !cfg.safe && status.position !== "NO_BABY";

  return (
    <>
      <View style={[styles.statusCard, { backgroundColor: cfg.bg, borderColor: cfg.color }]}>
        <View style={styles.emojiWrap}>
          <PulseRing active={unsafe} color={cfg.color} />
          <Text style={styles.bigEmoji}>{cfg.emoji}</Text>
        </View>
        <Text style={[styles.posLabel, { color: cfg.color }]}>{cfg.label.toUpperCase()}</Text>
        <Text style={styles.posDesc}>{cfg.desc}</Text>

        <View style={styles.pillRow}>
          <Pill label="Confidence" value={`${Math.round((status.confidence ?? 0) * 100)}%`} />
          <Pill label="FPS"        value={status.fps ?? "–"} />
          <Pill label="Alerts"     value={status.alert_count ?? 0} />
        </View>

        {status.reason ? (
          <Text style={styles.reasonText}>
            {status.keypoints_used ? "🦴 " : "📐 "}{status.reason}
          </Text>
        ) : null}

        <View style={[styles.liveBadge, { backgroundColor: connected ? "#14532d" : "#450a0a" }]}>
          <View style={[styles.liveDot, { backgroundColor: connected ? "#4ade80" : "#f87171" }]} />
          <Text style={[styles.liveText, { color: connected ? "#4ade80" : "#f87171" }]}>
            {connected ? "CONNECTED" : "OFFLINE"}
          </Text>
        </View>
      </View>

      <View style={styles.refCard}>
        <Text style={styles.secTitle}>Position Reference</Text>
        {Object.entries(POS).filter(([k]) => k !== "NO_BABY").map(([key, c]) => (
          <View key={key} style={styles.refRow}>
            <Text style={styles.refEmoji}>{c.emoji}</Text>
            <View style={{ flex: 1 }}>
              <Text style={[styles.refLabel, { color: c.color }]}>{c.label}</Text>
              <Text style={styles.refDesc}>{c.desc}</Text>
            </View>
            <Text style={{ fontSize: 18, fontWeight: "800", color: c.safe ? "#4ade80" : "#f87171" }}>
              {c.safe ? "✓" : "✗"}
            </Text>
          </View>
        ))}
      </View>
    </>
  );
}

// ── Feed tab ──────────────────────────────────────────────────────────────────
function FeedTab({ status, rtspUrl }) {
  const cfg = POS[status.position] ?? POS.UNKNOWN;
  return (
    <View style={styles.feedWrapper}>
      <VLCPlayer
        source={{
          uri: rtspUrl,
          initOptions: [
            "--network-caching=150",
            "--rtsp-tcp",
            "--no-drop-late-frames",
            "--no-skip-frames",
          ],
        }}
        autoplay={true}
        hwDecoderEnabled={1}
        hwDecoderForced={1}
        style={styles.vlcPlayer}
        videoAspectRatio="16:9"
        resizeMode="contain"
      />
      <View style={[styles.feedBadge, { backgroundColor: cfg.bg + "ee", borderColor: cfg.color }]}>
        <Text style={styles.feedBadgeEmoji}>{cfg.emoji}</Text>
        <Text style={[styles.feedBadgeLabel, { color: cfg.color }]}>{cfg.label.toUpperCase()}</Text>
      </View>
    </View>
  );
}

// ── Alerts tab ────────────────────────────────────────────────────────────────
function AlertsTab({ alerts, onClear }) {
  if (alerts.length === 0) {
    return (
      <View style={styles.emptyWrap}>
        <Text style={styles.emptyEmoji}>✅</Text>
        <Text style={styles.emptyTitle}>All clear</Text>
        <Text style={styles.emptySub}>No unsafe positions detected yet.</Text>
      </View>
    );
  }
  return (
    <View>
      <View style={styles.alertsHeader}>
        <Text style={styles.secTitle}>Alert History ({alerts.length})</Text>
        <TouchableOpacity onPress={onClear}>
          <Text style={styles.clearBtn}>Clear all</Text>
        </TouchableOpacity>
      </View>
      {alerts.slice().reverse().map((a, i) => {
        const c = POS[a.position] ?? POS.UNKNOWN;
        return (
          <View key={i} style={[styles.alertRow, { borderLeftColor: c.color }]}>
            <Text style={styles.alertEmoji}>{c.emoji}</Text>
            <View style={{ flex: 1 }}>
              <Text style={[styles.alertPos, { color: c.color }]}>{c.label}</Text>
              <Text style={styles.alertReason} numberOfLines={2}>{a.reason}</Text>
              <Text style={styles.alertTime}>{a.time}</Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

// ── Settings tab ──────────────────────────────────────────────────────────────
function AlertModeSelector({ value, onChange }) {
  return (
    <View style={styles.modeCard}>
      <Text style={styles.secTitle}>Alert Mode</Text>
      <View style={styles.modeGrid}>
        {ALERT_MODES.map(m => {
          const active = value === m.id;
          return (
            <TouchableOpacity
              key={m.id}
              style={[styles.modeBtn, active && styles.modeBtnActive]}
              onPress={() => onChange(m.id)}
            >
              <Text style={styles.modeIcon}>{m.icon}</Text>
              <Text style={[styles.modeLabel, active && styles.modeLabelActive]}>{m.label}</Text>
              <Text style={styles.modeDesc}>{m.desc}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}

function SettingsTab({ alertMode, setAlertMode, config, onSaveConfig }) {
  const [draft, setDraft] = useState({ ...config });
  const [saved, setSaved] = useState(false);

  // sync if parent config changes (e.g. on first load)
  useEffect(() => { setDraft({ ...config }); }, [config]);

  function field(label, key, opts = {}) {
    return (
      <View style={styles.inputRow}>
        <Text style={styles.inputLabel}>{label}</Text>
        <TextInput
          style={styles.inputField}
          value={draft[key]}
          onChangeText={v => setDraft(prev => ({ ...prev, [key]: v }))}
          placeholderTextColor="#475569"
          autoCapitalize="none"
          autoCorrect={false}
          {...opts}
        />
      </View>
    );
  }

  function handleSave() {
    onSaveConfig(draft);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  const previewBase = buildBaseUrl(draft);
  const previewRtsp = buildRtspUrl(draft);

  return (
    <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : undefined}>
      <AlertModeSelector value={alertMode} onChange={setAlertMode} />

      {/* ── Connection settings ── */}
      <View style={styles.modeCard}>
        <Text style={styles.secTitle}>🖥 Server</Text>
        {field("IP Address", "serverIp", { keyboardType: "numeric" })}
        {field("Port",       "serverPort", { keyboardType: "numeric" })}
      </View>

      <View style={styles.modeCard}>
        <Text style={styles.secTitle}>📷 Camera (RTSP)</Text>
        {field("Camera IP",   "rtspIp",   { keyboardType: "numeric" })}
        {field("Port",        "rtspPort", { keyboardType: "numeric" })}
        {field("Username",    "rtspUser")}
        {field("Password",    "rtspPass", { secureTextEntry: true })}
        {field("Stream path", "rtspPath", { placeholder: "stream1" })}
      </View>

      {/* preview */}
      <View style={styles.card}>
        <Text style={styles.serverLbl}>Server endpoint</Text>
        <Text style={styles.serverVal} numberOfLines={1}>{previewBase}</Text>
        <Text style={[styles.serverLbl, { marginTop: 10 }]}>RTSP URL</Text>
        <Text style={styles.serverVal} numberOfLines={1}>{previewRtsp}</Text>
      </View>

      <TouchableOpacity style={[styles.saveBtn, saved && styles.saveBtnDone]} onPress={handleSave}>
        <Text style={styles.saveBtnTxt}>{saved ? "✓ Saved" : "Save & Apply"}</Text>
      </TouchableOpacity>

      <View style={styles.divider} />
      <Text style={styles.secTitle}>Position Reference</Text>
      {Object.entries(POS).filter(([k]) => k !== "NO_BABY").map(([key, c]) => (
        <View key={key} style={styles.refRow}>
          <Text style={styles.refEmoji}>{c.emoji}</Text>
          <View style={{ flex: 1 }}>
            <Text style={[styles.refLabel, { color: c.color }]}>{c.label}</Text>
            <Text style={styles.refDesc}>{c.desc}</Text>
          </View>
          <Text style={{ fontSize: 18, fontWeight: "800", color: c.safe ? "#4ade80" : "#f87171" }}>
            {c.safe ? "✓" : "✗"}
          </Text>
        </View>
      ))}
    </KeyboardAvoidingView>
  );
}

// ── Tab bar ───────────────────────────────────────────────────────────────────
const TABS = [
  { id: "monitor",  label: "Monitor",  icon: "📊" },
  { id: "feed",     label: "Feed",     icon: "📷" },
  { id: "alerts",   label: "Alerts",   icon: "🔔" },
  { id: "settings", label: "Settings", icon: "⚙️"  },
];

function TabBar({ active, onChange, badgeCount }) {
  const insets = useSafeAreaInsets();
  return (
    <View style={[styles.tabBar, { paddingBottom: Math.max(insets.bottom, 8) }]}>
      {TABS.map(t => (
        <TouchableOpacity key={t.id} style={styles.tabItem} onPress={() => onChange(t.id)}>
          <Text style={styles.tabIcon}>{t.icon}</Text>
          <Text style={[styles.tabLbl, active === t.id && styles.tabLblActive]}>{t.label}</Text>
          {t.id === "alerts" && badgeCount > 0 && (
            <View style={styles.badge}>
              <Text style={styles.badgeTxt}>{badgeCount > 9 ? "9+" : badgeCount}</Text>
            </View>
          )}
          {active === t.id && <View style={styles.tabLine} />}
        </TouchableOpacity>
      ))}
    </View>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────
function Main() {
  const [tab,        setTab]        = useState("monitor");
  const [status,     setStatus]     = useState({ position: "NO_BABY", confidence: 0, fps: 0, alert_count: 0, reason: "", keypoints_used: false });
  const [connected,  setConnected]  = useState(false);
  const [alerts,     setAlerts]     = useState([]);
  const [alertMode,  setAlertMode]  = useState("both");
  const [granted,    setGranted]    = useState(false);
  const [config,     setConfig]     = useState(DEFAULT_CONFIG);

  const prevAlertCount = useRef(0);
  const cooldown       = useRef(false);
  const configRef      = useRef(DEFAULT_CONFIG);  // always-current config for poll()

  // Load saved config on mount
  useEffect(() => {
    setupNotificationChannel();
    requestNotifPermission().then(setGranted);

    AsyncStorage.getItem(STORAGE_KEY).then(raw => {
      if (raw) {
        try {
          const saved = { ...DEFAULT_CONFIG, ...JSON.parse(raw) };
          setConfig(saved);
          configRef.current = saved;
        } catch (_) {}
      }
    });
  }, []);

  async function handleSaveConfig(draft) {
    const merged = { ...DEFAULT_CONFIG, ...draft };
    setConfig(merged);
    configRef.current = merged;
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
  }

  const poll = useCallback(async () => {
    const baseUrl = buildBaseUrl(configRef.current);
    try {
      const controller = new AbortController();
      const timeout    = setTimeout(() => controller.abort(), 3000);
      const res        = await fetch(`${baseUrl}/status`, { signal: controller.signal });
      clearTimeout(timeout);
      const data = await res.json();
      setStatus(data);
      setConnected(true);

      const newAlerts = (data.alert_count ?? 0) - prevAlertCount.current;
      if (newAlerts > 0) {
        prevAlertCount.current = data.alert_count;
        setAlerts(prev => [...prev, {
          time:     new Date().toLocaleTimeString(),
          position: data.position,
          reason:   data.reason ?? "",
        }]);

        if (granted && alertMode !== "none" && !cooldown.current) {
          cooldown.current = true;
          fireAlert(data.position, alertMode);
          setTimeout(() => { cooldown.current = false; }, 30000);
        }
      }
    } catch (e) {
      console.log("POLL ERROR:", e.message);
      setConnected(false);
    }
  }, [granted, alertMode]);

  useEffect(() => {
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, [poll]);

  const cfg    = POS[status.position] ?? POS.UNKNOWN;
  const unsafe = !cfg.safe && status.position !== "NO_BABY";

  return (
    <SafeAreaView style={styles.root} edges={["top"]}>
      <StatusBar barStyle="light-content" backgroundColor="#080810" />

      {tab !== "feed" && (
        <View style={[styles.header, unsafe && { backgroundColor: "#2d0a0a" }]}>
          <View>
            <Text style={styles.headerTitle}>👶 BabyCam</Text>
            <Text style={styles.headerSub}>Sleep Safety Monitor</Text>
          </View>
          {unsafe && (
            <View style={styles.headerBadge}>
              <Text style={styles.headerBadgeTxt}>⚠️ CHECK BABY</Text>
            </View>
          )}
        </View>
      )}

      {tab === "feed" ? (
        <FeedTab status={status} rtspUrl={buildRtspUrl(config)} />
      ) : (
        <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
          {tab === "monitor"  && <MonitorTab status={status} connected={connected} />}
          {tab === "alerts"   && <AlertsTab alerts={alerts} onClear={() => setAlerts([])} />}
          {tab === "settings" && <SettingsTab alertMode={alertMode} setAlertMode={setAlertMode} config={config} onSaveConfig={handleSaveConfig} />}
        </ScrollView>
      )}

      <TabBar active={tab} onChange={setTab} badgeCount={alerts.length} />
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <Main />
    </SafeAreaProvider>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const C = { dark: "#080810", card: "#0f172a", border: "#1e293b", muted: "#475569", text: "#e2e8f0", sub: "#64748b" };

const styles = StyleSheet.create({
  root:           { flex: 1, backgroundColor: C.dark },
  header:         { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 20, paddingVertical: 14 },
  headerTitle:    { fontSize: 22, fontWeight: "900", color: C.text, letterSpacing: 0.5 },
  headerSub:      { fontSize: 11, color: C.muted, marginTop: 1 },
  headerBadge:    { backgroundColor: "#7f1d1d", paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8 },
  headerBadgeTxt: { color: "#fca5a5", fontWeight: "800", fontSize: 12, letterSpacing: 1 },
  scroll:         { padding: 16, paddingBottom: 28 },

  feedWrapper:    { flex: 1, backgroundColor: "#000" },
  vlcPlayer:      { flex: 1 },
  feedBadge:      { position: "absolute", top: 16, left: 16, flexDirection: "row", alignItems: "center", gap: 8, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 10, borderWidth: 1.5 },
  feedBadgeEmoji: { fontSize: 16 },
  feedBadgeLabel: { fontSize: 12, fontWeight: "800", letterSpacing: 1 },

  statusCard:     { borderRadius: 24, borderWidth: 2, padding: 24, alignItems: "center", marginBottom: 14 },
  emojiWrap:      { width: 100, height: 100, alignItems: "center", justifyContent: "center", marginBottom: 10 },
  pulseRing:      { position: "absolute", width: 100, height: 100, borderRadius: 50, borderWidth: 3 },
  bigEmoji:       { fontSize: 60 },
  posLabel:       { fontSize: 26, fontWeight: "900", letterSpacing: 3, marginBottom: 4 },
  posDesc:        { fontSize: 14, color: "#94a3b8", marginBottom: 18, textAlign: "center" },
  pillRow:        { flexDirection: "row", gap: 8, marginBottom: 14 },
  pill:           { backgroundColor: "#0f172a", borderRadius: 12, paddingHorizontal: 12, paddingVertical: 9, alignItems: "center", minWidth: 76 },
  pillVal:        { fontSize: 18, fontWeight: "800", color: C.text },
  pillLbl:        { fontSize: 10, color: C.sub, marginTop: 2, textAlign: "center" },
  reasonText:     { fontSize: 11, color: "#64748b", textAlign: "center", marginBottom: 14, paddingHorizontal: 12, fontStyle: "italic" },
  liveBadge:      { flexDirection: "row", alignItems: "center", gap: 6, paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20 },
  liveDot:        { width: 8, height: 8, borderRadius: 4 },
  liveText:       { fontSize: 11, fontWeight: "800", letterSpacing: 1.5 },

  refCard:        { backgroundColor: C.card, borderRadius: 16, padding: 16, marginBottom: 14 },
  refRow:         { flexDirection: "row", alignItems: "center", paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: C.border, gap: 12 },
  refEmoji:       { fontSize: 22, width: 32 },
  refLabel:       { fontSize: 14, fontWeight: "700" },
  refDesc:        { fontSize: 12, color: C.sub, marginTop: 1 },

  card:           { backgroundColor: C.card, borderRadius: 16, padding: 16, marginBottom: 14 },

  // Alert mode selector
  modeCard:       { backgroundColor: C.card, borderRadius: 16, padding: 16, marginBottom: 14 },
  modeGrid:       { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 4 },
  modeBtn:        { flex: 1, minWidth: "45%", backgroundColor: "#0a0a14", borderRadius: 12, padding: 14, alignItems: "center", borderWidth: 1.5, borderColor: C.border },
  modeBtnActive:  { borderColor: "#4ade80", backgroundColor: "#052e16" },
  modeIcon:       { fontSize: 24, marginBottom: 6 },
  modeLabel:      { fontSize: 13, fontWeight: "700", color: C.sub, marginBottom: 2 },
  modeLabelActive:{ color: "#4ade80" },
  modeDesc:       { fontSize: 10, color: C.muted, textAlign: "center" },

  alertsHeader:   { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  alertRow:       { flexDirection: "row", alignItems: "flex-start", backgroundColor: C.card, borderRadius: 12, padding: 14, marginBottom: 8, borderLeftWidth: 3, gap: 12 },
  alertEmoji:     { fontSize: 22, marginTop: 2 },
  alertPos:       { fontSize: 15, fontWeight: "700" },
  alertReason:    { fontSize: 11, color: C.sub, marginTop: 2, fontStyle: "italic" },
  alertTime:      { fontSize: 11, color: C.muted, marginTop: 4 },
  clearBtn:       { color: C.sub, fontSize: 13, fontWeight: "600" },
  emptyWrap:      { alignItems: "center", paddingVertical: 60 },
  emptyEmoji:     { fontSize: 48, marginBottom: 12 },
  emptyTitle:     { fontSize: 20, fontWeight: "800", color: C.text, marginBottom: 6 },
  emptySub:       { fontSize: 14, color: C.muted },

  divider:        { height: 1, backgroundColor: C.border, marginVertical: 16 },
  serverLbl:      { fontSize: 11, color: C.sub, marginBottom: 3 },
  serverVal:      { fontSize: 13, color: "#7dd3fc", fontFamily: Platform.OS === "android" ? "monospace" : "Courier" },

  inputRow:       { marginBottom: 12 },
  inputLabel:     { fontSize: 11, color: C.sub, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.5 },
  inputField:     { backgroundColor: "#0a0a14", borderWidth: 1.5, borderColor: C.border, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10, color: C.text, fontSize: 14, fontFamily: Platform.OS === "android" ? "monospace" : "Courier" },
  saveBtn:        { backgroundColor: "#1e3a5f", borderRadius: 12, paddingVertical: 14, alignItems: "center", marginBottom: 8 },
  saveBtnDone:    { backgroundColor: "#14532d" },
  saveBtnTxt:     { color: "#7dd3fc", fontWeight: "800", fontSize: 15, letterSpacing: 0.5 },

  tabBar:         { flexDirection: "row", backgroundColor: "#0a0a14", borderTopWidth: 1, borderTopColor: C.border, paddingTop: 6 },
  tabItem:        { flex: 1, alignItems: "center", position: "relative", paddingTop: 4, paddingBottom: 4 },
  tabIcon:        { fontSize: 20, marginBottom: 2 },
  tabLbl:         { fontSize: 10, color: C.muted, fontWeight: "600" },
  tabLblActive:   { color: "#7dd3fc" },
  tabLine:        { position: "absolute", bottom: -4, width: 24, height: 2, backgroundColor: "#7dd3fc", borderRadius: 1 },
  badge:          { position: "absolute", top: 0, right: 8, backgroundColor: "#ef4444", borderRadius: 8, minWidth: 16, height: 16, alignItems: "center", justifyContent: "center", paddingHorizontal: 3 },
  badgeTxt:       { color: "#fff", fontSize: 9, fontWeight: "800" },

  secTitle:       { fontSize: 12, fontWeight: "700", color: C.sub, letterSpacing: 1, textTransform: "uppercase", marginBottom: 10 },
});
