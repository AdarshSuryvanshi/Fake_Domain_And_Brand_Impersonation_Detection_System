#importing required libraries

from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
import requests
from urllib.parse import urlparse
import ssl
import socket
from datetime import datetime
from bs4 import BeautifulSoup
import os
hf_token = os.getenv("HF_TOKEN")
warnings.filterwarnings('ignore')
from feature import FeatureExtraction
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

file = open("pickle/model.pkl","rb")
gbc = pickle.load(file)
file.close()

from dotenv import load_dotenv
load_dotenv()  # This looks for the .env file and loads the variables
app = Flask(__name__)

def normalize_url(raw: str) -> str:
    try:
        parsed = urlparse(raw)
        if not parsed.scheme:
            return f"http://{raw}"
        return raw
    except Exception:
        return raw

def check_site(url: str):
    info = {"reachable": False, "status_code": None, "https": False, "error": None}
    try:
        parsed = urlparse(url)
        info["https"] = parsed.scheme.lower() == "https"
        resp = requests.head(url, timeout=6, allow_redirects=True)
        info["status_code"] = resp.status_code
        info["reachable"] = True
    except requests.exceptions.SSLError as e:
        info["error"] = "SSL error"
    except requests.exceptions.RequestException as e:
        info["error"] = str(e)
    return info

def get_ssl_status(url: str):
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            return {"enabled": False}
        host = parsed.hostname
        port = parsed.port or 443
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=6) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                tls_version = ssock.version()
                cipher_name, cipher_protocol, cipher_bits = ssock.cipher()
                not_after = cert.get('notAfter')
                not_before = cert.get('notBefore')
                issuer_t = cert.get('issuer')
                subject_t = cert.get('subject')
                san = cert.get('subjectAltName')
                def flatten(t):
                    return ', '.join('='.join(x) for x in sum(t, ())) if t else None
                issuer = flatten(issuer_t)
                subject = flatten(subject_t)
                expires_in_days = None
                if not_after:
                    try:
                        exp_dt = datetime.strptime(not_after, '%b %d %H:%M:%S %Y %Z')
                        expires_in_days = (exp_dt - datetime.utcnow()).days
                    except Exception:
                        pass
                # Hostname match check: SAN first, else CN
                hostname_match = None
                if san:
                    dns_names = [v for (k, v) in san if k == 'DNS']
                    hostname_match = host in dns_names
                if hostname_match is None:
                    cn = None
                    try:
                        for tup in subject_t or []:
                            for k, v in tup:
                                if k == 'commonName':
                                    cn = v
                                    break
                    except Exception:
                        pass
                    hostname_match = (cn == host) if cn else None
                # Self-signed heuristic: issuer == subject
                self_signed = (issuer == subject) if (issuer and subject) else None
                # TLS policy: TLSv1.2 or TLSv1.3 considered OK
                tls_ok = tls_version in ('TLSv1.2', 'TLSv1.3')
                return {
                    "enabled": True,
                    "valid": True,
                    "issuer": issuer,
                    "subject": subject,
                    "not_after": not_after,
                    "not_before": not_before,
                    "expires_in_days": expires_in_days,
                    "tls_version": tls_version,
                    "tls_ok": tls_ok,
                    "cipher_name": cipher_name,
                    "cipher_bits": cipher_bits,
                    "hostname_match": hostname_match,
                    "self_signed": self_signed
                }
    except ssl.SSLError as e:
        return {"enabled": True, "valid": False, "error": f"SSL error: {e}"}
    except Exception as e:
        return {"enabled": True, "valid": False, "error": str(e)}

def check_local_blacklist(url: str) -> bool:
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or url
        # First, try structured CSV with pandas
        try:
            df = pd.read_csv('phishing.csv', dtype=str, low_memory=False)
            cols = [c for c in df.columns]
            # Common column guesses
            candidates = [
                'url','URL','domain','Domain','hostname','Hostname','site','Site','web','Web','Address','address'
            ]
            target_cols = [c for c in candidates if c in cols]
            if target_cols:
                for c in target_cols:
                    series = df[c].dropna().astype(str).str.lower()
                    if (series == domain.lower()).any() or series.str.contains(domain.lower()).any():
                        return True
            # Fallback: scan all object columns for domain substring
            obj_cols = df.select_dtypes(include=['object']).columns
            for c in obj_cols:
                series = df[c].dropna().astype(str).str.lower()
                if series.str.contains(domain.lower()).any():
                    return True
        except Exception:
            # Lightweight fallback: search file lines for domain
            with open('phishing.csv', 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if domain.lower() in line.lower():
                        return True
        return False
    except Exception:
        return False

FEATURE_LABELS = [
    "IP in URL","Long URL","Shortener","@ symbol","Multiple //",
    "Prefix/Suffix -","Subdomains","HTTPS","Domain reg length","Favicon",
    "Non-standard port","HTTPS in domain","External resource ratio","Unsafe anchors",
    "Script/link external ratio","Form handler","Info email","Abnormal URL",
    "Forwarding","Status bar trick","Disable right click","Popup alerts",
    "Iframe/frame","Domain age","DNS record age","Website traffic","PageRank",
    "Google indexed","Links pointing","Blacklist/Stats"
]

FEATURE_REASONS_NEG = {
    0: "URL looks like an IP address",
    1: "URL length unusually long",
    2: "Uses a URL shortener",
    3: "Contains @ symbol",
    4: "Suspicious redirect (// in path)",
    5: "Domain has hyphen (-)",
    6: "Too many subdomains",
    7: "Not using HTTPS",
    8: "Short domain registration",
    9: "Favicon from external domain",
    10: "Non-standard port",
    11: "'https' inside domain name",
    12: "Externally loaded resources",
    13: "Many unsafe anchor links",
    14: "Excess external scripts/links",
    15: "Form submits to unknown host",
    16: "Info email detected",
    17: "Abnormal URL vs WHOIS",
    18: "Multiple redirects",
    19: "On-hover status bar script",
    20: "Right-click disabled via script",
    21: "Popup alerts present",
    22: "Iframes present",
    23: "New domain (young age)",
    24: "DNS newly recorded",
    25: "Low/unknown traffic",
    26: "Low page rank",
    27: "Not indexed by Google",
    28: "Too many outbound links",
    29: "Matches known bad patterns/IPs"
}

FEATURE_REASONS_NEU = {
    1: "Medium URL length",
    6: "Two subdomains",
    12: "Some external resources",
    14: "Some external scripts/links",
    18: "Few redirects",
    25: "Medium traffic",
    28: "Few outbound links"
}

def compute_risk_and_reasons(features: np.ndarray):
    vals = features.flatten().tolist()
    neg = sum(1 for v in vals if v == -1)
    neu = sum(1 for v in vals if v == 0)
    risk_features = ((neg * 2) + (neu * 1)) / (len(vals) * 2) * 100.0
    reasons = []
    for idx, v in enumerate(vals):
        if v == -1 and idx in FEATURE_REASONS_NEG:
            reasons.append(f"{FEATURE_LABELS[idx]}: {FEATURE_REASONS_NEG[idx]}")
        elif v == 0 and idx in FEATURE_REASONS_NEU:
            reasons.append(f"{FEATURE_LABELS[idx]}: {FEATURE_REASONS_NEU[idx]}")
    return round(risk_features, 2), reasons[:8]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        raw_url = request.form["url"].strip()
        result = analyze_url(raw_url)
        return render_template('index.html', **result)
    return render_template("index.html", xx =-1)


def analyze_url(raw_url: str):
    url = normalize_url(raw_url)
    site = check_site(url)
    if not site["reachable"]:
        return {
            'xx': -1,
            'url': raw_url,
            'risk': None,
            'category': 'Unknown / Not Reachable',
            'ssl': 'HTTPS' if site["https"] else 'HTTP',
            'status_code': site["status_code"],
            'reasons': ["Site didn’t load", str(site.get("error") or "")],
            'model_confidence': None,
            'ssl_status': get_ssl_status(url),
            'risk_explanation': 'Site unreachable.'
        }

    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1,30)
    y_pred = gbc.predict(x)[0]
    y_pro_phishing = gbc.predict_proba(x)[0,0]
    y_pro_non_phishing = gbc.predict_proba(x)[0,1]
    risk_features, reasons = compute_risk_and_reasons(x)
    risk_model = round((1.0 - float(y_pro_non_phishing)) * 100.0, 2)
    risk = round((risk_features + risk_model) / 2.0, 2)

    if risk <= 30:
        category = 'Legitimate / Safe website'
    elif risk <= 70:
        category = 'Suspicious'
    else:
        category = 'Phishing / Malicious '

    ssl_proto = 'HTTPS' if site["https"] else 'HTTP'
    ssl_status = get_ssl_status(url)
    # Google index feature at position 27 (0-based)
    google_indexed = None
    try:
        google_indexed = bool(x[0][27] == 1)
    except Exception:
        google_indexed = None
    # Local blacklist check
    is_blacklisted = check_local_blacklist(url)

    if risk <= 30:
        if ssl_status.get("enabled"):
            ssl_text = "Valid SSL" if ssl_status.get("valid") else "Invalid SSL"
        else:
            ssl_text = "No SSL (HTTP)"
        risk_explanation = f"Low risk. {ssl_text}. Few or no red flags detected."
        reasons_to_show = []
    else:
        extra = []
        if ssl_status.get('enabled') and not ssl_status.get('valid'):
            extra.append('Invalid SSL certificate')
        if ssl_status.get('tls_version') and not ssl_status.get('tls_ok'):
            extra.append(f"Weak TLS ({ssl_status.get('tls_version')})")
        if ssl_status.get('hostname_match') is False:
            extra.append('Certificate hostname mismatch')
        if ssl_status.get('self_signed') is True:
            extra.append('Self-signed certificate')
        exp_days = ssl_status.get('expires_in_days')
        if isinstance(exp_days, int) and exp_days <= 0:
            extra.append('Certificate expired')
        if is_blacklisted:
            extra.insert(0, 'Blacklisted domain (local list)')
        top_reasons = (extra + reasons)[:5]
        risk_explanation = "Top signals: " + ", ".join(top_reasons) if top_reasons else "Model + features indicate elevated risk."
        reasons_to_show = top_reasons

    return {
        'xx': round(y_pro_non_phishing,2),
        'url': url,
        'risk': risk,
        'category': category,
        'ssl': ssl_proto,
        'status_code': site["status_code"],
        'reasons': reasons_to_show,
        'model_confidence': round(float(y_pro_non_phishing)*100.0,2),
        'ssl_status': ssl_status,
        'risk_explanation': risk_explanation,
        'google_indexed': google_indexed,
        'blacklisted': is_blacklisted
    }


@app.route('/api/check', methods=['POST'])
def api_check():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    raw_url = str(data['url']).strip()
    result = analyze_url(raw_url)
    return jsonify(result), 200


import os
from huggingface_hub import InferenceClient

def get_hf_client():
    if InferenceClient is None:
        return None, "huggingface_hub not installed"
    
    # Standardize on one variable name: hf_token
    # It will look for HF_TOKEN in your environment variables
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        return None, "Missing Hugging Face API token (set HF_TOKEN environment variable)"
    
    try:
        # Pass the hf_token variable here
        client = InferenceClient(api_key=hf_token)
        return client, None
    except Exception as e:
        return None, str(e)


def classify_message(text: str, model: str = "ealvaradob/bert-finetuned-phishing"):
    client, err = get_hf_client()
    if err or client is None:
        return None, err or "Client init failed"
    try:
        outputs = client.text_classification(text, model=model)
        # Normalize outputs to list of dicts
        norm = []
        for o in outputs:
            try:
                norm.append({"label": getattr(o, 'label', None), "score": float(getattr(o, 'score', 0.0))})
            except Exception:
                try:
                    norm.append({"label": o.get('label'), "score": float(o.get('score', 0.0))})
                except Exception:
                    pass
        # Determine top label and phishing score
        top = max(norm, key=lambda x: x['score']) if norm else {"label": None, "score": 0.0}
        score_phishing = next((x['score'] for x in norm if x['label'] == 'phishing'), None)
        score_benign = next((x['score'] for x in norm if x['label'] == 'benign'), None)
        is_phishing = (top['label'] == 'phishing' and top['score'] >= 0.5)
        return {
            "text": text,
            "label": top['label'],
            "score": round(float(top['score']), 6),
            "is_phishing": bool(is_phishing),
            "score_phishing": round(float(score_phishing), 6) if score_phishing is not None else None,
            "score_benign": round(float(score_benign), 6) if score_benign is not None else None,
            "raw": norm
        }, None
    except Exception as e:
        return None, str(e)


@app.route('/api/checkmsg', methods=['POST'])
def api_checkmsg():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    text = str(data['text']).strip()
    res, err = classify_message(text)
    if err:
        return jsonify({"error": err}), 500
    return jsonify(res), 200


def extract_page_text(url: str, max_chars: int = 1000):
    meta = {"content_type": None, "status_code": None, "title": None}
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; URL-Inspector/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        meta["status_code"] = resp.status_code
        meta["content_type"] = resp.headers.get("content-type")
        if resp.status_code >= 400:
            return None, "HTTP error while fetching content", meta
        # Parse as HTML when possible
        text = ""
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove non-content tags
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            # Capture title if present
            if soup.title and soup.title.string:
                meta["title"] = soup.title.string.strip()
            # Get text
            text = " ".join(s for s in soup.stripped_strings)
        except Exception:
            text = resp.text
        # Normalize whitespace and truncate
        text = " ".join(text.split())
        if not text:
            return None, "No extractable text found", meta
        if len(text) > max_chars:
            text = text[:max_chars]
        return text, None, meta
    except Exception as e:
        return None, str(e), meta


@app.route('/api/verifyurl', methods=['POST'])
def api_verifyurl():
    data = request.get_json(silent=True)
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    raw_url = str(data['url']).strip()
    # Step 1: normal URL analysis
    url_result = analyze_url(raw_url)
    normalized_url = url_result.get('url') or raw_url
    # Step 2: scrape and classify text if reachable
    can_fetch = url_result.get('category') != 'Unknown / Not Reachable'
    text_payload = None
    text_error = None
    content_meta = None
    message_result = None
    if can_fetch:
        text_payload, text_error, content_meta = extract_page_text(normalized_url)
        if text_payload:
            message_result, msg_err = classify_message(text_payload)
            if msg_err:
                message_result = {"error": msg_err}
        else:
            message_result = {"error": text_error or "Unable to extract text"}
    else:
        message_result = {"error": "Site unreachable; cannot analyze content"}

    return jsonify({
        "url": normalized_url,
        "url_result": url_result,
        "content": {
            "scraped": bool(text_payload is not None),
            "title": content_meta.get('title') if content_meta else None,
            "status_code": content_meta.get('status_code') if content_meta else url_result.get('status_code'),
            "content_type": content_meta.get('content_type') if content_meta else None,
            "error": None if text_payload else (text_error or (None if can_fetch else "Site unreachable"))
        },
        "message_result": message_result
    }), 200


if __name__ == "__main__":
    app.run(debug=True)