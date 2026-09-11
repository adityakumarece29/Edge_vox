import serial
import wave
import os
import time

# ============================================================
# PS26172 DATASET COLLECTOR
# ESP32 + INMP441
# ============================================================

# =========================
# ESP32 SETTINGS
# =========================

PORT = "COM9"
BAUDRATE = 921600

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
BYTES_PER_SAMPLE = 2

# Exactly 1 second = 16000 samples
EXPECTED_BYTES = NUM_SAMPLES * BYTES_PER_SAMPLE

# =========================
# DATASET SETTINGS
# =========================

OUTPUT_DIR = "dataset"
SAMPLES_PER_CLASS = 30

CLASSES = {
    "1": "keyword",
    "2": "unknown",
    "3": "silence"
}

# ============================================================
# CREATE DATASET FOLDERS
# ============================================================

for folder in CLASSES.values():
    os.makedirs(
        os.path.join(OUTPUT_DIR, folder),
        exist_ok=True
    )

# ============================================================
# START PROGRAM
# ============================================================

print()
print("==============================================")
print("        PS26172 DATASET COLLECTOR")
print("==============================================")
print()

print("Connecting to ESP32...")
print("Port     :", PORT)
print("Baudrate :", BAUDRATE)
print()

# ============================================================
# CONNECT ESP32
# ============================================================

try:

    ser = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        timeout=3,
        write_timeout=3
    )

except Exception as e:

    print()
    print("ERROR: Could not open", PORT)
    print()
    print(e)
    print()
    print("Check:")
    print("1. Arduino Serial Monitor is CLOSED")
    print("2. Arduino Serial Plotter is CLOSED")
    print("3. Correct COM port is selected")
    print("4. ESP32 is connected")
    print()

    input("Press ENTER to exit...")
    raise SystemExit

# Give ESP32 time to initialize
time.sleep(2)

# Clear old serial data
ser.reset_input_buffer()
ser.reset_output_buffer()

print("ESP32 CONNECTED!")
print()

# ============================================================
# RECORD ONE SAMPLE
# ============================================================

def record_sample(option, label, number):

    print()
    print("==============================================")
    print("Class  :", label.upper())
    print("Sample :", f"{number:02d}/{SAMPLES_PER_CLASS}")
    print("==============================================")

    # --------------------------------------------------------
    # User starts recording
    # --------------------------------------------------------

    input("Press ENTER to start recording...")

    print()
    print("Recording...")
    print("Please speak / make the required sound.")
    print()

    # --------------------------------------------------------
    # Clear any previous serial data
    # --------------------------------------------------------

    ser.reset_input_buffer()

    # --------------------------------------------------------
    # Send class command to ESP32
    #
    # 1 = keyword
    # 2 = unknown
    # 3 = silence
    # --------------------------------------------------------

    ser.write(option.encode("ascii"))
    ser.flush()

    # --------------------------------------------------------
    # Receive exactly 32000 bytes
    # 16000 samples × 2 bytes
    # --------------------------------------------------------

    audio_data = bytearray()

    start_time = time.time()

    while len(audio_data) < EXPECTED_BYTES:

        remaining = EXPECTED_BYTES - len(audio_data)

        chunk = ser.read(
            min(4096, remaining)
        )

        if chunk:

            audio_data.extend(chunk)

        # Safety timeout
        if time.time() - start_time > 10:

            print()
            print("ERROR: Recording timeout!")
            print(
                "Received:",
                len(audio_data),
                "/",
                EXPECTED_BYTES,
                "bytes"
            )

            return False

    # --------------------------------------------------------
    # Verify received data
    # --------------------------------------------------------

    print(
        "Audio received:",
        len(audio_data),
        "bytes"
    )

    if len(audio_data) != EXPECTED_BYTES:

        print("ERROR: Wrong audio size!")

        return False

    # --------------------------------------------------------
    # Create filename
    # --------------------------------------------------------

    filename = os.path.join(
        OUTPUT_DIR,
        label,
        f"{label}_{number:02d}.wav"
    )

    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    try:

        with wave.open(filename, "wb") as wav:

            wav.setnchannels(1)       # Mono
            wav.setsampwidth(2)      # 16-bit
            wav.setframerate(16000)  # 16 kHz
            wav.writeframes(audio_data)

    except Exception as e:

        print()
        print("ERROR while saving WAV:")
        print(e)

        return False

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print()
    print("SUCCESS!")
    print("Saved:", os.path.abspath(filename))

    return True


# ============================================================
# SHOW MENU
# ============================================================

print("----------------------------------------------")
print("DATASET COLLECTION")
print("----------------------------------------------")
print()
print("1. KEYWORD")
print("2. UNKNOWN")
print("3. SILENCE")
print()
print("30 samples will be collected for each class.")
print()

# ============================================================
# COLLECT EACH CLASS
# ============================================================

for option, label in CLASSES.items():

    print()
    print()
    print("##############################################")
    print("        STARTING:", label.upper())
    print("##############################################")
    print()

    for number in range(1, SAMPLES_PER_CLASS + 1):

        success = record_sample(
            option,
            label,
            number
        )

        if not success:

            print()
            print("Recording failed.")
            print("Stopping collector.")
            print()

            ser.close()

            input("Press ENTER to exit...")

            raise SystemExit

        # Small gap before next sample
        time.sleep(0.5)


# ============================================================
# FINISH
# ============================================================

ser.close()

print()
print()
print("==============================================")
print("           DATASET COMPLETE")
print("==============================================")
print()

print("Keyword :", SAMPLES_PER_CLASS)
print("Unknown :", SAMPLES_PER_CLASS)
print("Silence :", SAMPLES_PER_CLASS)

print()
print("Total   :", SAMPLES_PER_CLASS * 3)
print()

print("Dataset location:")
print(os.path.abspath(OUTPUT_DIR))
print()

input("Press ENTER to exit...")