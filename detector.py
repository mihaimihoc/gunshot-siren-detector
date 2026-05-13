import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import time
from collections import deque

SAMPLE_RATE          = 16000
WINDOW_SECONDS       = 1.0
HOP_SECONDS          = 0.5
CONFIDENCE_THRESHOLD = 0.80   
REQUIRED_HITS = {
    "Gunshot": 1,  # Gunshots are extremely short, trigger immediately
    "Siren": 3     # Sirens are continuous, 3 consecutive hits to avoid false alarms
}
MIN_RMS_ENERGY       = 0.002 
LABELS               = ["Gunshot", "Siren", "Background"]

BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SIZE    = int(SAMPLE_RATE * HOP_SECONDS)

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

print("Loading your trained classifier...")
classifier = tf.keras.models.load_model("model/sound_classifier.keras")

print("Ready.\n")

audio_buffer = deque(maxlen=BUFFER_SIZE)

def audio_callback(indata, frames, time_info, status):
    mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    audio_buffer.extend(mono.tolist())

consecutive_counts = {"Gunshot": 0, "Siren": 0}

def update_smoothing(label_detected: str | None):
    for key in consecutive_counts:
        if key == label_detected:
            consecutive_counts[key] += 1
        else:
            consecutive_counts[key] = 0  # reset if not detected this window

    # Check against the specific requirement for the detected label
    if label_detected and consecutive_counts[label_detected] >= REQUIRED_HITS[label_detected]:
        return label_detected
    return None

def run_inference():
    if len(audio_buffer) < BUFFER_SIZE: # not enough audio yet for a full window
        return None

    waveform = np.array(list(audio_buffer), dtype=np.float32) # deque to numpy array

    rms = np.sqrt(np.mean(waveform ** 2)) # skip silent windows and reduce false positives
    if rms < MIN_RMS_ENERGY:
        update_smoothing(None)  
        return {
            "gunshot":    0.0,
            "siren":      0.0,
            "background": 1.0,
            "alert":      None,
            "gated":      True,
        }

    # YAMNet to get embeddings
    _, embeddings, _ = yamnet(waveform)
    mean_embedding = np.mean(embeddings.numpy(), axis=0)

    # Keras classifier gives probabilities
    probs = classifier.predict(mean_embedding[np.newaxis, :], verbose=0)[0]

    gunshot_conf = float(probs[0])
    siren_conf   = float(probs[1])

    # Find top detection above threshold
    candidates = [
        ("Gunshot", gunshot_conf),
        ("Siren",   siren_conf),
    ]
    candidates = [(l, c) for l, c in candidates if c >= CONFIDENCE_THRESHOLD]

    # Determine the most confident label among candidates
    top_label = None
    if candidates:
        top_label = max(candidates, key=lambda x: x[1])[0]

    # Apply temporal smoothing
    confirmed_alert = update_smoothing(top_label)

    return {
        "gunshot":    gunshot_conf,
        "siren":      siren_conf,
        "background": float(probs[2]),
        "alert":      confirmed_alert,  
        "gated":      False,
    }

# streaming loop
def start_stream(callback_fn):
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=HOP_SIZE,
        callback=audio_callback
    ):
        while True:
            time.sleep(HOP_SECONDS)
            result = run_inference()
            if result:
                callback_fn(result)