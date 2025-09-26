import os
import json, io, time, logging
import numpy as np
import torch, torchaudio, torchcrepe, librosa, soundfile as sf, joblib
from transformers import AutoFeatureExtractor, AutoModel

# ---------------- Config ----------------
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
TARGET_SR  = 16000
WIN_SEC    = float(os.getenv("WIN_SEC", 1.0))
HOP_SEC    = float(os.getenv("HOP_SEC", 0.5))
TRIM_P     = float(os.getenv("VAL_TRIM_P", 0.10))
CAL_LEVEL  = os.getenv("CAL_LEVEL", "clip").lower()

# ---------------- Logging ----------------
log = logging.getLogger("vad-api")
DEBUG = bool(int(os.getenv("DEBUG", "1")))
def dbg(msg, **kw):
    if DEBUG:
        log.debug(msg + ((" | " + json.dumps(kw)) if kw else ""))

# ---------------- Lazy-load wav2vec2 ----------------
feature_extractor = None
encoder = None
def get_models():
    global feature_extractor, encoder
    if feature_extractor is None or encoder is None:
        logging.info(f"Cargando modelo {MODEL_NAME} desde HuggingFace...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        encoder = AutoModel.from_pretrained(MODEL_NAME).eval()
        logging.info("Modelo cargado correctamente.")
    return feature_extractor, encoder

# ---------------- Load regressors & calibrators ----------------
def _load_reg(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra: {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict):
        for k in ("xgb_best", "model", "regressor", "estimator"):
            if k in obj: return obj[k]
        if len(obj) == 1: return next(iter(obj.values()))
    return obj

reg_v = _load_reg("models/valence/xgb_model.joblib")
reg_a = _load_reg("models/arousal/xgb_model.joblib")
reg_d = _load_reg("models/dominance/xgb_model.joblib")

iso_v = joblib.load("models/valence/iso_per_lang.joblib") if os.path.exists("models/valence/iso_per_lang.joblib") else None
iso_a = joblib.load("models/arousal/iso_global.joblib") if os.path.exists("models/arousal/iso_global.joblib") else None
iso_d = joblib.load("models/dominance/iso_global.joblib") if os.path.exists("models/dominance/iso_global.joblib") else None

# ---------------- Load META & acoustic scaler ----------------
META = {}
try:
    with open("models/meta.json", "r") as f:
        META = json.load(f)
    log.info("Meta cargado correctamente.")
except FileNotFoundError:
    log.warning("Meta no encontrado. Usando defaults.")

MU_AC, SD_AC = None, None
try:
    with open("models/valence/acoustic_val.json", "r") as f:
        norm = json.load(f)
        MU_AC = np.array(norm["mu_ac"], np.float32)
        SD_AC = np.array(norm["sd_ac"], np.float32)
except FileNotFoundError:
    MU_AC = SD_AC = None
    log.warning("No se encontró acoustic_norm.json, se omite normalización acústica.")

# === Precarga de modelo al iniciar Celery ===
try:
    _ = get_models()
    print("✅ Modelo Wav2Vec2 precargado correctamente en el worker.")
except Exception as e:
    print(f"No se pudo precargar el modelo Wav2Vec2: {e}")

# ---------------- Feature extraction helpers ----------------
def to_mono(w): return w if w.ndim == 1 else w.mean(axis=1)

def resample_16k(w, sr):
    return w if sr == TARGET_SR else librosa.resample(y=w, orig_sr=sr, target_sr=TARGET_SR)

def frame_windows(wave, sr):
    win = int(WIN_SEC*sr); hop = int(HOP_SEC*sr)
    out = []
    for i in range(0, len(wave) - win + 1, hop):
        out.append(wave[i:i+win])
    return out or [wave]

def rms(x): return float(np.sqrt(np.mean(x*x) + 1e-12))

@torch.inference_mode()
def embed_batch(wavs_16k):
    log.info(f"[embed] start batch={len(wavs_16k)}")
    global feature_extractor, encoder
    fe, enc = feature_extractor, encoder
    log.info("[embed] got models")
    inputs = fe(wavs_16k, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    log.info("[embed] feature extractor done")
    hs = enc(**inputs).last_hidden_state
    log.info("[embed] forward done")
    mean, std = hs.mean(dim=1), hs.std(dim=1)
    emb = torch.cat([mean, std], dim=1).cpu().numpy().astype(np.float32)
    log.info("[embed] pooled ok")
    return emb


def extract_acoustic_features(wavs_16k):
    feats = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mfcc_tfm = torchaudio.transforms.MFCC(
        sample_rate=TARGET_SR, n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
    ).to(device)
    for w in wavs_16k:
        y = torch.from_numpy(w).to(device)
        with torch.no_grad():
            mfcc = mfcc_tfm(y)
            mfcc_mean, mfcc_std = mfcc.mean(dim=1), mfcc.std(dim=1)
            f0 = librosa.yin(w, fmin=60, fmax=500, sr=TARGET_SR)
            p_feats = np.array([np.nanmean(f0), np.nanstd(f0), np.nanmin(f0), np.nanmax(f0)], np.float32)
            p_feats = torch.tensor(p_feats, device=device)
            f = torch.cat([mfcc_mean, mfcc_std, p_feats]).cpu().numpy()
        feats.append(f)
    return np.stack(feats, axis=0).astype(np.float32)

def build_features_v3(emb, rms_list, acoustic_feats, lang="es"):
    # RMS z-score
    mu_rms = float(META.get("mu", 0.0))
    sd_rms = float(META.get("sd", 1.0))
    zrms = ((rms_list - mu_rms) / sd_rms)[:, None].astype(np.float32)

    # log-RMS z-score
    log_rms = np.log1p(np.clip(rms_list, 1e-9, None))
    mu_log, sd_log = log_rms.mean(), log_rms.std(ddof=1) + 1e-9
    zrms_log = ((log_rms - mu_log) / sd_log)[:, None].astype(np.float32)

    # normalize acoustic feats
    if MU_AC is not None and SD_AC is not None:
        acoustic_feats = (acoustic_feats - MU_AC) / (SD_AC + 1e-8)

    # language flag
    is_es = np.ones((len(emb), 1), np.float32) if lang == "es" else np.zeros((len(emb), 1), np.float32)

    return np.concatenate([emb, is_es, zrms, zrms_log, acoustic_feats], axis=1).astype(np.float32)

def aggregate_vector(x, mode="mean", rms_weights=None, trim_p=0.10):
    if mode == "median": return float(np.median(x))
    if mode == "trimmed": 
        k = int(len(x)*trim_p)
        xs = np.sort(x)
        return float(np.mean(xs[k:len(xs)-k]))
    if mode == "rms_weighted_mean" and rms_weights is not None:
        wsum = float(np.sum(rms_weights)) + 1e-12
        return float(np.sum(x * rms_weights) / wsum)
    if mode.startswith("p"):
        q = float(mode[1:])
        return float(np.percentile(x, q))
    return float(np.mean(x))

def apply_iso(model_or_dict, value, lang="es"):
    if model_or_dict is None:
        return value
    if isinstance(model_or_dict, dict):
        iso = model_or_dict.get(lang) or list(model_or_dict.values())[0]
        return iso.predict([value])[0]
    return model_or_dict.predict([value])[0]

# ---------------- Prediction pipeline ----------------
def predict_from_wave(wave, sr):
    try:
        w = resample_16k(to_mono(wave), sr)
        rms_list = np.array([rms(w)], np.float32)
        emb = embed_batch([w])
        acoustic_feats = extract_acoustic_features([w])
        X = build_features_v3(emb, rms_list, acoustic_feats, lang="es")

        # Predict
        v_w_raw = np.asarray(reg_v.predict(X), np.float32)
        log.info(f"Ventanas predichas: valence min={v_w_raw.min():.4f}, max={v_w_raw.max():.4f}, mean={v_w_raw.mean():.4f}")
        a_w_raw = np.asarray(reg_a.predict(X), np.float32)
        log.info(f"Ventanas predichas: arousal min={a_w_raw.min():.4f}, max={a_w_raw.max():.4f}, mean={a_w_raw.mean():.4f}")
        d_w_raw = np.asarray(reg_d.predict(X), np.float32)
        log.info(f"Ventanas predichas: dominance min={d_w_raw.min():.4f}, max={d_w_raw.max():.4f}, mean={d_w_raw.mean():.4f}")

        if CAL_LEVEL == "window":
            v_w, a_w, d_w = [np.clip(iso_v.predict(v_w_raw),0,1),
                             np.clip(iso_a.predict(a_w_raw),0,1),
                             np.clip(iso_d.predict(d_w_raw),0,1)]
        else:
            v_w, a_w, d_w = v_w_raw, a_w_raw, d_w_raw

        # Aggregate to clip
        v_clip_raw = aggregate_vector(v_w, "trimmed", rms_list, TRIM_P)
        a_clip_raw = aggregate_vector(a_w, "trimmed", rms_list, TRIM_P)
        d_clip_raw = aggregate_vector(d_w, "trimmed", rms_list, TRIM_P)

        v_clip = apply_iso(iso_v, v_clip_raw)
        a_clip = apply_iso(iso_a, a_clip_raw)
        d_clip = apply_iso(iso_d, d_clip_raw)
        
        log.info(f"Clip predicho: valence={v_clip:.4f}, arousal={a_clip:.4f}, dominance={d_clip:.4f}")

        return {"clip": {"valence": float(v_clip), "arousal": float(a_clip), "dominance": float(d_clip)}}

    except Exception as e:
        log.exception("Fallo en predict_from_wave")
        raise

# ---------------- Anxiety score ----------------
def anxiety_percent_from_vad(v, a, d, decimals=2):
    ANX, SAFE = np.array([1,5,1]), np.array([5,1,5])
    dist_max = np.linalg.norm(ANX - SAFE)
    sim = 1.0 - np.linalg.norm(np.array([v,a,d]) - ANX) / (dist_max + 1e-12)
    return round(sim*100, decimals)

def run_anxiety_analysis(audio_bytes, decimals=2):
    wave, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if wave.ndim > 1: wave = wave.mean(axis=1)
    log.info(f"Audio cargado: {len(wave)/sr:.2f}s a {sr}Hz")
    out = predict_from_wave(wave, sr)
    v, a, d = [out["clip"][k] for k in ("valence","arousal","dominance")]
    return {"anxiety_pct": anxiety_percent_from_vad(v,a,d,decimals)}
