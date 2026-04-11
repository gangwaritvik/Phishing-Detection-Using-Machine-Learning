import os
import pickle
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from feature_extractor import get_feature_dict

app = Flask(__name__, template_folder='.')

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

model = None
feature_names = None


def load_model():
    global model, feature_names

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            model = data.get("model")
            feature_names = data.get("feature_names")
        else:
            model = data

    if model:
        print("✓ ML model loaded")
    else:
        print("⚠ No model found — using rule-based fallback")


def expand_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        r = requests.get(
            url,
            headers=headers,
            allow_redirects=False,
            timeout=8
        )

        if "Location" in r.headers:
            expanded = r.headers["Location"]
            return expanded, 1

        return url, 0

    except Exception as e:
        print("Expansion error:", e)
        return url, 0


def rule_based_score(features):

    score = 0
    reasons = []

    if features.get("IsDomainIP"):
        score += 25
        reasons.append("IP address used")

    if features.get("IsHTTPS") == 0:
        score += 15
        reasons.append("No HTTPS")

    if features.get("HasAtSymbol"):
        score += 20
        reasons.append("@ symbol in URL")

    if features.get("PrefixSuffixInDomain"):
        score += 15
        reasons.append("Hyphen in domain")

    if features.get("SuspiciousTLD"):
        score += 20
        reasons.append("Suspicious TLD")

    if features.get("IsShortenedURL"):
        score += 20
        reasons.append("Shortened URL")

    if features.get("URLLength", 0) > 100:
        score += 10
        reasons.append("Very long URL")

    if features.get("SubDomainCount", 0) > 2:
        score += 10
        reasons.append("Many subdomains")

    score = min(score, 100)

    label = 1 if score >= 40 else 0
    probability = score / 100

    return label, probability, reasons


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    url = (data or {}).get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    expanded_url, redirect_count = expand_url(url)

    print("Original URL:", url)
    print("Expanded URL:", expanded_url)

    target_url = expanded_url if redirect_count > 0 and expanded_url != url else url

    features = get_feature_dict(target_url)

    features["RedirectCount"] = redirect_count
    features["HasRedirect"] = 1 if redirect_count > 0 else 0
    features["MultipleRedirects"] = 1 if redirect_count > 2 else 0

    if features.get("IsShortenedURL") and redirect_count == 0:
        features["RedirectFailure"] = 1
    else:
        features["RedirectFailure"] = 0

    reasons = []

    if model is not None:
        try:
            names = feature_names or list(features.keys())
            X = np.array([[features.get(n, 0) for n in names]])

            ml_pred = int(model.predict(X)[0])
            ml_prob = float(model.predict_proba(X)[0][1])

            rule_pred, rule_prob, reasons = rule_based_score(features)

            prob_phishing = max(ml_prob, rule_prob)

            pred = 1 if prob_phishing > 0.35 else 0
            source = "ml_model"

        except Exception as e:
            print("Prediction error:", e)
            pred, prob_phishing, reasons = rule_based_score(features)
            source = "fallback"

    else:
        pred, prob_phishing, reasons = rule_based_score(features)
        source = "rule_based"

    key_features = {
        k: features[k]
        for k in [
            "IsHTTPS",
            "IsDomainIP",
            "HasAtSymbol",
            "PrefixSuffixInDomain",
            "SuspiciousTLD",
            "IsShortenedURL",
            "HasBrandKeyword",
            "URLLength",
            "SubDomainCount",
            "RedirectCount",
        ]
        if k in features
    }

    return jsonify({
        "url": target_url,
        "original_url": url,
        "prediction": pred,
        "label": "Phishing" if pred else "Legitimate",
        "phishing_probability": prob_phishing,
        "confidence": round(abs(prob_phishing - 0.5) * 2, 4),
        "key_features": key_features,
        "all_features": features,
        "reasons": reasons,
        "source": source
    })


load_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
