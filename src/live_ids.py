# ============================================================
# live_ids.py — Real-time IDS using trained IDSNet model
# Works like Snort but uses your ML model
# ============================================================

import torch
import numpy as np
import joblib
import time
import datetime
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP, ICMP, conf

# ── Load your trained model ──────────────────────────────────
import sys
sys.path.insert(0, '.')
from ids_model import IDSNet

# ── Configuration ────────────────────────────────────────────
MODEL_PATH      = '../models/ids_adversarial.pth'   # Use hardened model
SCALER_PATH     = '../models/scaler_standard.pkl'   # Feature scaler
THRESHOLD       = 0.5                               # Attack probability threshold
INTERFACE       = None                              # None = auto detect
LOG_FILE        = '../results/live_ids_alerts.log'

# ── Colours for terminal output ──────────────────────────────
RED    = '\033[91m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
BLUE   = '\033[94m'
RESET  = '\033[0m'
BOLD   = '\033[1m'

# ── Flow tracker (tracks connection statistics) ──────────────
flow_stats = defaultdict(lambda: {
    'count'         : 0,
    'src_bytes'     : 0,
    'dst_bytes'     : 0,
    'serror_count'  : 0,
    'duration_start': time.time(),
    'syn_count'     : 0,
    'fin_count'     : 0,
    'rst_count'     : 0,
})

alert_count   = 0
packet_count  = 0

# ── Load Model ───────────────────────────────────────────────
def load_model():
    print(f"{BLUE}[*] Loading IDSNet model...{RESET}")
    try:
        model = IDSNet(input_dim=38)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu',
                                weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"{GREEN}[✓] Model loaded: {MODEL_PATH}{RESET}")
        return model
    except Exception as e:
        print(f"{RED}[✗] Model load failed: {e}{RESET}")
        sys.exit(1)

def load_scaler():
    print(f"{BLUE}[*] Loading feature scaler...{RESET}")
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"{GREEN}[✓] Scaler loaded{RESET}")
        return scaler
    except Exception as e:
        print(f"{YELLOW}[!] Scaler not found — using raw features: {e}{RESET}")
        return None

# ── Feature Extraction ───────────────────────────────────────
# Converts a raw packet into 38 NSL-KDD-style features
def extract_features(packet, flow_key):

    stats = flow_stats[flow_key]

    # Update flow stats
    stats['count'] += 1
    stats['src_bytes'] += len(packet) if IP in packet else 0

    duration = time.time() - stats['duration_start']

    # TCP flag analysis
    tcp_flags = 0
    if TCP in packet:
        flags = packet[TCP].flags
        if flags & 0x02:  stats['syn_count'] += 1   # SYN
        if flags & 0x01:  stats['fin_count'] += 1   # FIN
        if flags & 0x04:  stats['rst_count'] += 1   # RST
        tcp_flags = int(flags)

    count      = stats['count']
    src_bytes  = stats['src_bytes']
    syn_count  = stats['syn_count']
    fin_count  = stats['fin_count']
    rst_count  = stats['rst_count']

    # Build 38 features matching NSL-KDD structure
    features = [
        # Basic (4)
        duration,                                        # 0  duration
        1 if TCP in packet else 0,                       # 1  protocol_type
        src_bytes,                                       # 2  src_bytes
        stats['dst_bytes'],                              # 3  dst_bytes

        # Content (13)
        0,                                               # 4  land
        0,                                               # 5  wrong_fragment
        0,                                               # 6  urgent
        0,                                               # 7  hot
        0,                                               # 8  num_failed_logins
        0,                                               # 9  logged_in
        0,                                               # 10 num_compromised
        0,                                               # 11 root_shell
        0,                                               # 12 su_attempted
        0,                                               # 13 num_root
        0,                                               # 14 num_file_creations
        0,                                               # 15 num_shells
        0,                                               # 16 num_access_files

        # Traffic same host (9)
        count,                                           # 17 count
        syn_count,                                       # 18 srv_count
        rst_count / max(count, 1),                       # 19 serror_rate
        rst_count / max(count, 1),                       # 20 srv_serror_rate
        fin_count / max(count, 1),                       # 21 rerror_rate
        fin_count / max(count, 1),                       # 22 srv_rerror_rate
        1.0,                                             # 23 same_srv_rate
        0.0,                                             # 24 diff_srv_rate
        syn_count / max(count, 1),                       # 25 srv_diff_host_rate

        # Traffic destination host (10)
        min(count * 2, 255),                             # 26 dst_host_count
        min(syn_count * 2, 255),                         # 27 dst_host_srv_count
        syn_count / max(count, 1),                       # 28 dst_host_same_srv_rate
        0.0,                                             # 29 dst_host_diff_srv_rate
        syn_count / max(count, 1),                       # 30 dst_host_same_src_port_rate
        0.0,                                             # 31 dst_host_srv_diff_host_rate
        rst_count / max(count, 1),                       # 32 dst_host_serror_rate
        rst_count / max(count, 1),                       # 33 dst_host_srv_serror_rate
        fin_count / max(count, 1),                       # 34 dst_host_rerror_rate
        fin_count / max(count, 1),                       # 35 dst_host_srv_rerror_rate

        # Extra (2)
        tcp_flags,                                       # 36 tcp_flags
        len(packet),                                     # 37 packet_size
    ]

    return np.array(features, dtype=np.float32)


# ── Classify using IDSNet ────────────────────────────────────
def classify_packet(features, model, scaler):
    features_2d = features.reshape(1, -1)

    if scaler:
        try:
            features_2d = scaler.transform(features_2d)
        except Exception:
            pass  # Use raw if scaler fails

    with torch.no_grad():
        tensor = torch.tensor(features_2d, dtype=torch.float32)
        prob   = model(tensor).item()

    return prob


# ── Alert System ─────────────────────────────────────────────
def raise_alert(packet, flow_key, prob, log_file):
    global alert_count
    alert_count += 1

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    src_ip    = packet[IP].src  if IP in packet else 'unknown'
    dst_ip    = packet[IP].dst  if IP in packet else 'unknown'
    protocol  = 'TCP' if TCP in packet else 'UDP' if UDP in packet else 'OTHER'

    src_port  = packet[TCP].sport if TCP in packet else \
                packet[UDP].sport if UDP in packet else 0
    dst_port  = packet[TCP].dport if TCP in packet else \
                packet[UDP].dport if UDP in packet else 0

    # Terminal alert
    print(f"\n{RED}{BOLD}{'═'*60}{RESET}")
    print(f"{RED}{BOLD}  🚨 INTRUSION DETECTED — Alert #{alert_count}{RESET}")
    print(f"{RED}{BOLD}{'═'*60}{RESET}")
    print(f"  {YELLOW}Time     :{RESET} {timestamp}")
    print(f"  {YELLOW}Source   :{RESET} {src_ip}:{src_port}")
    print(f"  {YELLOW}Target   :{RESET} {dst_ip}:{dst_port}")
    print(f"  {YELLOW}Protocol :{RESET} {protocol}")
    print(f"  {YELLOW}Confidence:{RESET} {prob*100:.1f}% attack probability")
    print(f"  {YELLOW}Severity :{RESET} {'HIGH' if prob > 0.8 else 'MEDIUM'}")
    print(f"{RED}{BOLD}{'═'*60}{RESET}\n")

    # Log to file
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] ALERT #{alert_count} | "
                f"{src_ip}:{src_port} → {dst_ip}:{dst_port} | "
                f"{protocol} | Confidence: {prob*100:.1f}%\n")


# ── Packet Handler ───────────────────────────────────────────
def packet_handler(packet, model, scaler):
    global packet_count

    # Only process IP packets
    if IP not in packet:
        return

    packet_count += 1

    # Create flow key (src_ip:port → dst_ip:port)
    src_ip   = packet[IP].src
    dst_ip   = packet[IP].dst
    src_port = packet[TCP].sport if TCP in packet else \
               packet[UDP].sport if UDP in packet else 0
    dst_port = packet[TCP].dport if TCP in packet else \
               packet[UDP].dport if UDP in packet else 0
    flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"

    # Extract features
    features = extract_features(packet, flow_key)

    # Classify
    prob = classify_packet(features, model, scaler)

    # Status update every 10 packets
    if packet_count % 10 == 0:
        print(f"  {GREEN}[{datetime.datetime.now().strftime('%H:%M:%S')}] "
              f"Packets: {packet_count} | "
              f"Alerts: {alert_count} | "
              f"Last prob: {prob:.3f}{RESET}")

    # Raise alert if attack detected
    if prob > THRESHOLD:
        raise_alert(packet, flow_key, prob, LOG_FILE)


# ── Main ─────────────────────────────────────────────────────
def main():
    print(f"""
{BLUE}{BOLD}
╔══════════════════════════════════════════════════════════╗
║          Adversarial IDS — Live Network Monitor          ║
║              Powered by IDSNet (CDAC ITISS)              ║
╚══════════════════════════════════════════════════════════╝
{RESET}""")

    model  = load_model()
    scaler = load_scaler()

    print(f"\n{BLUE}[*] Configuration:{RESET}")
    print(f"    Model     : Adversarially Hardened IDSNet")
    print(f"    Threshold : {THRESHOLD} (>{THRESHOLD*100:.0f}% = ATTACK)")
    print(f"    Log file  : {LOG_FILE}")
    print(f"    Interface : {'Auto-detect' if not INTERFACE else INTERFACE}")

    print(f"\n{GREEN}{BOLD}[✓] Live IDS Started — Monitoring network traffic...{RESET}")
    print(f"{YELLOW}    Press Ctrl+C to stop{RESET}\n")
    print(f"  {'─'*58}")

    try:
        sniff(
            iface=INTERFACE,
            prn=lambda pkt: packet_handler(pkt, model, scaler),
            store=False,          # Don't store packets in memory
            filter="ip",          # Only IP packets
        )
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}[!] IDS Stopped by user{RESET}")
        print(f"{BLUE}[*] Session Summary:{RESET}")
        print(f"    Total packets analysed : {packet_count}")
        print(f"    Total alerts raised    : {alert_count}")
        print(f"    Alert log saved to     : {LOG_FILE}")
        print(f"{GREEN}[✓] Goodbye!{RESET}\n")


if __name__ == '__main__':
    main()
