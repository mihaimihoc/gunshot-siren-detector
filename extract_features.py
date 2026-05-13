import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import scipy.io.wavfile as wav
import scipy.signal as signal
from tqdm import tqdm


DATASET_PATH        = "UrbanSound8K"
CUSTOM_BACKGROUNDS  = [
    "custom_backgrounds/Voices",   # voice files
]
OUTPUT_PATH         = "features"
SAMPLE_RATE         = 16000
TARGET_CLASSES      = {6: "gunshot", 8: "siren"} # UrbanSound8K class IDs for gunshot and siren
AUGMENT_TARGETS     = True

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading YAMNet...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
print("Loaded.\n")

# add noise, time stretch, and pitch shift augmentations for gunshot and siren classes
def add_noise(audio, factor=0.005):
    noise = np.random.randn(len(audio)).astype(np.float32) # array of white noise
    return audio + factor * noise

def time_stretch(audio, rate=None):
    if rate is None:
        rate = np.random.choice([0.85, 0.9, 1.1, 1.15])
    num_samples = int(len(audio) / rate)
    return signal.resample(audio, num_samples)

def pitch_shift_simple(audio, semitones=None):
    if semitones is None:
        semitones = np.random.choice([-2, -1, 1, 2])
    rate = 2 ** (semitones / 12.0)
    shifted = signal.resample(audio, int(len(audio) / rate))
    if len(shifted) > len(audio):
        return shifted[:len(audio)]
    else:
        return np.pad(shifted, (0, len(audio) - len(shifted)))

def get_augmentations(audio):
    return [
        add_noise(audio, factor=0.003),
        add_noise(audio, factor=0.008),
        time_stretch(audio, rate=0.9),
        time_stretch(audio, rate=1.1),
        pitch_shift_simple(audio, semitones=1),
        pitch_shift_simple(audio, semitones=-1),
    ]

def load_audio(filepath):
    sr, audio = wav.read(filepath) # returns sample rate and audio data as numpy array
    if audio.ndim > 1:
        audio = audio.mean(axis=1) # mono mix if stereo
    audio = audio.astype(np.float32)
    if audio.max() > 1.0:
        audio = audio / 32768.0 # scales volume of 16-bit audio to [-1, 1]
    if sr != SAMPLE_RATE:
        num_samples = int(len(audio) * SAMPLE_RATE / sr) # resample to target yamnet sample rate
        audio = signal.resample(audio, num_samples)
    return audio

def get_embedding(audio):
    if len(audio) < SAMPLE_RATE * 0.5: # skip very short clips that may cause YAMNet to error out
        return None
    _, embeddings, _ = yamnet(audio) # class scores, embeddings, and a spectrogram (we only need embeddings)
    return np.mean(embeddings.numpy(), axis=0) # average over time frames to get a single embedding

all_embeddings = []
all_labels     = []
skipped        = 0

# UrbanSound8K metadata CSV processing
meta = pd.read_csv(os.path.join(DATASET_PATH, "metadata", "UrbanSound8K.csv"))

background = meta[~meta["classID"].isin(TARGET_CLASSES.keys())]
targets    = meta[meta["classID"].isin(TARGET_CLASSES.keys())]
subset     = pd.concat([targets, background]).reset_index(drop=True)

print(f"UrbanSound8K — Files to process: {len(subset)}")
print(f"  Gunshot    : {len(subset[subset.classID == 6])}")
print(f"  Siren      : {len(subset[subset.classID == 8])}")
print(f"  Background : {len(background)}\n")

for _, row in tqdm(subset.iterrows(), total=len(subset), desc="UrbanSound8K"):
    filepath = os.path.join(
        DATASET_PATH, "audio",
        f"fold{row['fold']}",
        row["slice_file_name"]
    )
    try:
        audio = load_audio(filepath)

        if row["classID"] == 6:
            label = 0
        elif row["classID"] == 8:
            label = 1
        else:
            label = 2

        emb = get_embedding(audio)
        if emb is None:
            skipped += 1
            continue

        all_embeddings.append(emb)
        all_labels.append(label)

        if AUGMENT_TARGETS and label in [0, 1]:
            for aug_audio in get_augmentations(audio):
                emb_aug = get_embedding(aug_audio.astype(np.float32))
                if emb_aug is not None:
                    all_embeddings.append(emb_aug)
                    all_labels.append(label)

    except Exception as e:
        skipped += 1

# Custom background files for voice processing
for folder in CUSTOM_BACKGROUNDS:
    if not os.path.exists(folder):
        print(f"\n⚠️  Folder not found, skipping: {folder}")
        continue

    wav_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    print(f"\nCustom background — '{folder}': {len(wav_files)} files")

    for filename in tqdm(wav_files, desc=os.path.basename(folder)):
        filepath = os.path.join(folder, filename)
        try:
            audio = load_audio(filepath)

            emb = get_embedding(audio)
            if emb is None:
                skipped += 1
                continue

            # All voice files are background (class 2)
            all_embeddings.append(emb)
            all_labels.append(2)

        except Exception as e:
            skipped += 1

# ── SAVE ─────────────────────────────────────────
print(f"\nDone. Skipped: {skipped}")

X = np.array(all_embeddings, dtype=np.float32)
y = np.array(all_labels,     dtype=np.int32)

np.save(os.path.join(OUTPUT_PATH, "X.npy"), X)
np.save(os.path.join(OUTPUT_PATH, "y.npy"), y)

print(f"\nSaved: X={X.shape}, y={y.shape}")
print(f"  Class 0 (gunshot)   : {np.sum(y==0)}")
print(f"  Class 1 (siren)     : {np.sum(y==1)}")
print(f"  Class 2 (background): {np.sum(y==2)}")