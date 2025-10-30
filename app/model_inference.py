import os
import json, io, time, logging
import numpy as np
import torch, torchaudio, torchcrepe, librosa, soundfile as sf, joblib
from transformers import AutoFeatureExtractor, AutoModel

# configuración
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
TARGET_SR  = 16000
WIN_SEC    = float(os.getenv("WIN_SEC", 1.0))
HOP_SEC    = float(os.getenv("HOP_SEC", 0.5))
TRIM_P     = float(os.getenv("VAL_TRIM_P", 0.10))
CAL_LEVEL  = os.getenv("CAL_LEVEL", "clip").lower()

# debug logger
log = logging.getLogger("vad-api")
DEBUG = bool(int(os.getenv("DEBUG", "1")))
def dbg(msg, **kw):
    if DEBUG:
        log.debug(msg + ((" | " + json.dumps(kw)) if kw else ""))

# carga modelo Wav2Vec2
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

# carga modelos de regresión
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

# carga META y normalización acústica
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

# Precarga de modelo al iniciar Celery ===
if os.getenv("ROLE", "").lower() == "worker":
    try:
        _ = get_models()
        print("Modelo Wav2Vec2 precargado correctamente en el worker.")
    except Exception as e:
        print(f"No se pudo precargar el modelo Wav2Vec2: {e}")

# funciones de procesamiento de audio
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
    global feature_extractor, encoder
    if feature_extractor is None or encoder is None:
        logging.warning("[embed] Modelo no estaba cargado, inicializando...")
        feature_extractor, encoder = get_models()
        logging.info("[embed] Modelo cargado dinámicamente dentro del worker.")

    log.info(f"[embed] start batch={len(wavs_16k)}")
    log.info("[embed] got models")
    inputs = feature_extractor(wavs_16k, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    log.info("[embed] feature extractor done")
    hs = encoder(**inputs).last_hidden_state
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

def chunk_audio(w, sr, chunk_sec=15):
    hop = int(chunk_sec * sr)
    return [w[i:i+hop] for i in range(0, len(w), hop)]

# ---------------- Prediction pipeline ----------------
def predict_from_wave(wave, sr):
    try:
        w = resample_16k(to_mono(wave), sr)
        chunks = chunk_audio(w, TARGET_SR)
        rms_list = np.array([rms(c) for c in chunks], np.float32)
        emb_parts, BATCH_SIZE = [], 2
        for i in range(0, len(chunks), BATCH_SIZE):
            emb_chunk = embed_batch(chunks[i:i+BATCH_SIZE])
            emb_parts.append(emb_chunk)
        emb = np.concatenate(emb_parts, axis=0)
        acoustic_feats = extract_acoustic_features(chunks)
        X = build_features_v3(emb, rms_list, acoustic_feats, lang="es")

        # Predice
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

# cálculo de ansiedad
import numpy as np

def anxiety_percent_from_vad(v, a, d, decimals=2):
    """
    % anxiety (v,a,d) 1–5
    """
    # polos emocionales    
    ANX  = np.array([1.5, 4.3, 1.5])   # ansiedad
    SAFE = np.array([4.5, 1.5, 4.5])   # relajado / seguro
    CONF = np.array([3.8, 2.8, 4.2])   # confiado / equilibrado
    vec  = np.array([v, a, d])

    # distancias euclidianas
    dist_anx  = np.linalg.norm(vec - ANX)
    dist_safe = np.linalg.norm(vec - SAFE)

    # Polaridad de ansiedad relativa  (-1 … +1)
    raw = (dist_safe - dist_anx) / (dist_safe + dist_anx + 1e-12)

    # Ajustes basados en VAD
    # Low valence + high arousal = tension
    valence_stress   = np.clip((3 - v) / 2, 0, 1)
    arousal_tension  = np.clip((a - 3) / 2, 0, 1)
    dominance_relief = np.clip((d - 3) / 2, 0, 1)  # más control ↓ ansiedad

    # ajuste lineal
    adj = (
        raw
        + 0.25 * arousal_tension * valence_stress   # más tensión → mayor ansiedad
        - 0.40 * dominance_relief                   # control/agency reduce ansiedad
        - 0.10 * (v - 3)                            # penaliza ligeramente la valencia positiva
    )

    # normaliza a 0–100%
    score = np.clip((adj + 1) / 2, 0, 1) * 100
    return round(score, decimals)

def count_meaningful_pauses(
    wave, sr,
    frame_len=0.032,          # 32 ms por frame
    hop_len=0.010,            # salto de 10 ms
    min_pause_dur=0.15,       # pausa mínima 150 ms
    min_speech_gap=0.15,      # une pausas separadas por <150 ms
    rms_med_win=5,            # mediana 5 frames (~70 ms)
    thr_percentile=25,        # umbral adaptativo bajo
    std_factor=1.0            # sensibilidad al ruido
):
    """
    Detecta pausas significativas en una señal de voz.
    Devuelve: (pause_count, pause_ratio, pauses_per_min, silence_seconds, audio_seconds)
    """

    FL = int(frame_len * sr)
    HL = int(hop_len * sr)
    audio_dur = len(wave) / sr

    if np.abs(wave).max() < 1e-3: # energía para analizar?
        # ruido muy bajo, mantener igual
        padded_wave = wave
    else:
        pad_dur = 1.0  # segundos
        pad_len = int(sr * pad_dur)
        padded_wave = np.concatenate([wave, np.zeros(pad_len, dtype=wave.dtype)])
    wave = padded_wave

    # --- RMS por frame ---
    rms = librosa.feature.rms(y=wave, frame_length=FL, hop_length=HL)[0]

    # Suavizado mediano para evitar respiraciones y clics
    if rms_med_win > 1:
        pad = rms_med_win // 2
        rms_sm = np.pad(rms, (pad, pad), mode="edge")
        rms_sm = np.array([np.median(rms_sm[i:i+rms_med_win])
                           for i in range(len(rms_sm)-rms_med_win+1)])
    else:
        rms_sm = rms

    # --- Umbral adaptativo ---
    thr_base = np.percentile(rms_sm, thr_percentile)
    std_rms  = np.std(rms_sm) if len(rms_sm) > 1 else 0
    thr = max(thr_base, std_factor * std_rms * 0.2)
    silent = rms_sm < thr

    # --- Agrupar regiones de silencio ---
    pauses = []
    min_frames = int(min_pause_dur / hop_len)
    cur_len = 0

    for s in silent:
        if s:
            cur_len += 1
        else:
            if cur_len >= min_frames:
                pauses.append(cur_len)
            cur_len = 0
    if cur_len >= min_frames:
        pauses.append(cur_len)

    pause_durs = np.array(pauses) * hop_len
    pause_count = len(pause_durs)
    silence_seconds = pause_durs.sum() if pause_count > 0 else 0.0

    pause_ratio = silence_seconds / max(audio_dur, 1e-6)
    pauses_per_min = pause_count / max(audio_dur / 60.0, 1e-6)

    return pause_count, pause_ratio, pauses_per_min, silence_seconds, audio_dur

def run_anxiety_analysis(audio_bytes, decimals=2):
    """Analiza ansiedad de voz (VAD + pausas)."""
    try:
        wave, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception as e:
        log.warning(f"sf.read falló ({type(e).__name__}): intentando con librosa...")
        try:
            wave, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        except Exception as e2:
            log.error(f"No se pudo decodificar el audio: {e2}")
            return {
                "status": "invalid_audio",
                "valence": None, "arousal": None, "dominance": None,
                "pause_count": 0, "pauses_per_min": 0.0, "pause_ratio": 1.0,
                "silence_seconds": 0.0, "audio_seconds": 0.0,
                "anxiety_base_vad": None, "anxiety_pct": None,
            }

    # Si llega aquí, el audio es válido:
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    audio_dur = len(wave) / sr
    rms_global = float(np.sqrt(np.mean(wave ** 2) + 1e-12))
    log.info(f"Audio cargado correctamente: {audio_dur:.2f}s a {sr}Hz (RMS={rms_global:.6f})")

    if rms_global < 5e-3:  # silencio?
        log.warning("Audio RMS muy bajo — sin voz detectable.")
        return {
            "valence": None, "arousal": None, "dominance": None,
            "pause_count": None, "pauses_per_min": None, "pause_ratio": None,
            "silence_seconds": audio_dur, "audio_seconds": audio_dur,
            "anxiety_base_vad": None, "anxiety_pct": None,
    }

    # --- pausas ---
    pc, pr, ppm, sil_sec, dur_sec = count_meaningful_pauses(wave, sr)

    if dur_sec <= 31:
        if sil_sec > 16:
            log.warning("Audio sin señal detectable (silencio total o ruido mínimo)")
            return {
                "valence": None,
                "arousal": None,
                "dominance": None,
                "pause_count": int(pc),
                "pauses_per_min": 0.0,
                "pause_ratio": 1.0,
                "silence_seconds": round(sil_sec, 2),
                "audio_seconds": round(dur_sec, 2),
                "anxiety_base_vad": None,
                "anxiety_pct": None,
            }
    elif dur_sec - sil_sec < 30:
        log.warning("Audio sin señal detectable (silencio total o ruido mínimo)")
        return {
            "valence": None,
            "arousal": None,
            "dominance": None,
            "pause_count": None,
            "pauses_per_min": None,
            "pause_ratio": None,
            "silence_seconds": round(sil_sec, 2),
            "audio_seconds": round(dur_sec, 2),
            "anxiety_base_vad": None,
            "anxiety_pct": None,
        }

    # predicción VAD
    out = predict_from_wave(wave, sr)
    v, a, d = [out["clip"][k] for k in ("valence", "arousal", "dominance")]

    base = anxiety_percent_from_vad(v, a, d, decimals)

    # corregir escalado poco realista para clips más cortos de 60 s
    if audio_dur < 60:
        ppm = pc / max(audio_dur, 1e-6)  # pausas por segundo
        log.info(f"[Short clip] {pc} pausas en {audio_dur:.2f}s → {ppm:.2f} pausas/s")
        pauses_per_min = ppm * 60 * (audio_dur / 60)  # neutralizar extrapolación
    else:
        pauses_per_min = pc / max(audio_dur/60.0, 1e-6)

    # reducir peso de pausa para clips cortos (< 60 s)
    dur_factor = np.clip(audio_dur / 60.0, 0, 1)

    pause_rate_factor  = np.clip((pauses_per_min - 4) / 8, 0, 1)
    pause_ratio_factor = np.clip((pr - 0.08) / 0.15, 0, 1)
    pause_factor = (0.6 * pause_rate_factor + 0.4 * pause_ratio_factor) * dur_factor

    adjusted = base * (1 + 0.35 * pause_factor)
    adjusted = float(np.clip(adjusted, 0, 100))

    return {
        "valence": round(v, 3),
        "arousal": round(a, 3),
        "dominance": round(d, 3),
        "pause_count": int(pc),
        "pauses_per_min": round(pauses_per_min, 2),
        "pause_ratio": round(pr, 3),
        "silence_seconds": round(sil_sec, 2),
        "audio_seconds": round(dur_sec, 2),
        "anxiety_base_vad": round(base, decimals),
        "anxiety_pct": round(adjusted, decimals),
    }
