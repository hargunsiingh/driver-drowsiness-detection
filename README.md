# Driver Drowsiness Detection System

Real-time driver drowsiness detection using **OpenCV** and **MediaPipe Face Mesh**.

## Features

| Feature | How it works |
|---|---|
| **Eye Closure Detection** | Calculates Eye Aspect Ratio (EAR) from 6 landmarks per eye |
| **Yawn Detection** | Calculates Mouth Aspect Ratio (MAR) from lip landmarks |
| **Head Droop Detection** | Estimates head pitch angle via `solvePnP` |
| **Audio Alarm** | Two-tone pulsing alert via `pygame` (auto-generated, no sound files needed) |
| **Visual Alerts** | Color-coded border (green/yellow/red), flashing overlay, status badges |
| **Face Mesh Overlay** | Live wireframe showing face oval, eye, and mouth outlines |

## Requirements

- Python 3.8+
- Webcam

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

- **Green border** → Driver is awake
- **Yellow border** → Warning — yawning or brief eye closure
- **Red border + alarm** → Drowsy — prolonged eye closure or multiple fatigue indicators
- Press **`q`** to quit

## Configuration

Thresholds can be tuned in `detector.py`:

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | 0.22 | EAR below this = eyes closed |
| `MAR_THRESHOLD` | 0.75 | MAR above this = yawning |
| `HEAD_TILT_THRESHOLD` | 30° | Pitch above this = head drooping |
| `CONSEC_FRAMES_DROWSY` | 20 | Frames of eye closure before drowsy alert |

## Project Structure

```
_driverDrowsyness/
├── main.py           # Entry point — webcam loop and UI
├── detector.py       # Drowsiness detection engine (EAR, MAR, head pose)
├── alert.py          # Audio alarm manager
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
