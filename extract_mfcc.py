import os
import wave
import numpy as np

# =========================================================
# DATASET PATH
# =========================================================

DATASET_DIR = r"C:\Users\adi\OneDrive\Documents\PS26172\dataset"

OUTPUT_FILE = os.path.join(DATASET_DIR, "mfcc_dataset_new.npz")

CLASSES = {
    "keyword": 0,
    "unknown": 1,
    "silence": 2
}

SAMPLES_PER_CLASS = 30

# =========================================================
# AUDIO / MFCC SETTINGS
# SAME AS ESP32
# =========================================================

SAMPLE_RATE = 16000

FRAME_SIZE = 400
HOP_SIZE = 160

FFT_SIZE = 512

NUM_MEL = 26
NUM_MFCC = 13

LOW_FREQ = 20
HIGH_FREQ = 8000

PRE_EMPHASIS = 0.97


# =========================================================
# WAV LOADER
# =========================================================

def load_wav(filename):

    with wave.open(filename, "rb") as wav:

        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()

        audio = wav.readframes(frames)

    if channels != 1:
        raise ValueError("Audio is not mono")

    if sample_width != 2:
        raise ValueError("Audio is not 16-bit")

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Wrong sample rate: {sample_rate}"
        )

    samples = np.frombuffer(
        audio,
        dtype=np.int16
    ).astype(np.float32)

    return samples


# =========================================================
# PRE-EMPHASIS
# =========================================================

def pre_emphasis(signal):

    output = np.empty_like(signal)

    output[0] = signal[0]

    output[1:] = (
        signal[1:]
        - PRE_EMPHASIS * signal[:-1]
    )

    return output


# =========================================================
# MEL SCALE
# =========================================================

def hz_to_mel(hz):

    return 2595.0 * np.log10(
        1.0 + hz / 700.0
    )


def mel_to_hz(mel):

    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


# =========================================================
# MEL FILTER BANK
# =========================================================

def create_mel_filterbank():

    low_mel = hz_to_mel(LOW_FREQ)
    high_mel = hz_to_mel(HIGH_FREQ)

    mel_points = np.linspace(
        low_mel,
        high_mel,
        NUM_MEL + 2
    )

    hz_points = mel_to_hz(mel_points)

    bins = np.floor(
        (FFT_SIZE + 1)
        * hz_points
        / SAMPLE_RATE
    ).astype(int)

    filterbank = np.zeros(
        (NUM_MEL, FFT_SIZE // 2 + 1),
        dtype=np.float32
    )

    for m in range(1, NUM_MEL + 1):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center == left:
            center += 1

        if right == center:
            right += 1

        for k in range(left, center):

            if k < filterbank.shape[1]:

                filterbank[m - 1, k] = (
                    (k - left)
                    / (center - left)
                )

        for k in range(center, right):

            if k < filterbank.shape[1]:

                filterbank[m - 1, k] = (
                    (right - k)
                    / (right - center)
                )

    return filterbank


# =========================================================
# DCT
# =========================================================

def dct_type_2(values):

    n = len(values)

    result = np.zeros(
        NUM_MFCC,
        dtype=np.float32
    )

    for k in range(NUM_MFCC):

        total = 0.0

        for i in range(n):

            angle = (
                np.pi
                * k
                * (2 * i + 1)
                / (2 * n)
            )

            total += (
                values[i]
                * np.cos(angle)
            )

        result[k] = total

    return result


# =========================================================
# MFCC EXTRACTION
# =========================================================

def extract_mfcc(signal, mel_filterbank):

    # Make sure we have at least 1 second
    if len(signal) < SAMPLE_RATE:

        padded = np.zeros(
            SAMPLE_RATE,
            dtype=np.float32
        )

        padded[:len(signal)] = signal
        signal = padded

    else:

        signal = signal[:SAMPLE_RATE]


    # Pre-emphasis
    signal = pre_emphasis(signal)

    frames = []

    # 97 frames exactly
    for frame_index in range(97):

        start = frame_index * HOP_SIZE

        frame = signal[
            start:start + FRAME_SIZE
        ]

        # Safety padding
        if len(frame) < FRAME_SIZE:

            padded = np.zeros(
                FRAME_SIZE,
                dtype=np.float32
            )

            padded[:len(frame)] = frame
            frame = padded

        # Hamming window
        window = np.hamming(FRAME_SIZE)

        frame = frame * window

        # Zero padding to FFT_SIZE
        fft_input = np.zeros(
            FFT_SIZE,
            dtype=np.float32
        )

        fft_input[:FRAME_SIZE] = frame

        # FFT
        spectrum = np.fft.rfft(
            fft_input,
            n=FFT_SIZE
        )

        power = (
            np.abs(spectrum) ** 2
        ) / FFT_SIZE

        # Mel filter bank
        mel_energy = np.dot(
            mel_filterbank,
            power
        )

        # Avoid log(0)
        mel_energy = np.maximum(
            mel_energy,
            1e-10
        )

        log_mel = np.log(
            mel_energy
        )

        # DCT
        mfcc_frame = dct_type_2(
            log_mel
        )

        frames.append(mfcc_frame)

    return np.array(
        frames,
        dtype=np.float32
    )


# =========================================================
# MAIN
# =========================================================

print()
print("=" * 60)
print("           PS26172 MFCC EXTRACTION")
print("=" * 60)
print()

print("Dataset:")
print(DATASET_DIR)

print()
print("MFCC configuration:")
print("Sample rate :", SAMPLE_RATE)
print("Frame size  :", FRAME_SIZE)
print("Hop size    :", HOP_SIZE)
print("FFT size    :", FFT_SIZE)
print("Mel filters :", NUM_MEL)
print("MFCC        :", NUM_MFCC)
print("Frames      : 97")

print()

mel_filterbank = create_mel_filterbank()

X = []
y = []

total = 0

# =========================================================
# PROCESS 30 FILES FROM EACH CLASS
# =========================================================

for class_name, label in CLASSES.items():

    folder = os.path.join(
        DATASET_DIR,
        class_name
    )

    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])

    # Only first 30
    files = files[:SAMPLES_PER_CLASS]

    print("-" * 60)
    print(
        f"{class_name.upper()} : "
        f"{len(files)} files"
    )
    print("-" * 60)

    if len(files) < SAMPLES_PER_CLASS:

        raise RuntimeError(
            f"Only {len(files)} files found "
            f"for {class_name}"
        )

    for index, filename in enumerate(files):

        path = os.path.join(
            folder,
            filename
        )

        signal = load_wav(path)

        features = extract_mfcc(
            signal,
            mel_filterbank
        )

        X.append(features)
        y.append(label)

        total += 1

        print(
            f"[{index + 1:02d}/30] "
            f"{filename}  ->  "
            f"{features.shape}"
        )


# =========================================================
# CONVERT TO NUMPY
# =========================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)


# =========================================================
# SAVE
# =========================================================

np.savez(
    OUTPUT_FILE,
    X=X,
    y=y
)


# =========================================================
# FINAL INFORMATION
# =========================================================

print()
print("=" * 60)
print("             EXTRACTION COMPLETE")
print("=" * 60)

print()
print("Total samples :", len(X))
print("X shape       :", X.shape)
print("y shape       :", y.shape)

print()
print("Expected:")
print("X = (90, 97, 13)")
print("y = (90,)")

print()
print("MFCC min      :", np.min(X))
print("MFCC max      :", np.max(X))
print("MFCC mean     :", np.mean(X))
print("MFCC std      :", np.std(X))

print()
print("Saved file:")
print(OUTPUT_FILE)

print()
print("STATUS: MFCC DATASET READY")
print("STATUS: 30 KEYWORD + 30 UNKNOWN + 30 SILENCE")
print()