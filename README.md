# Driver Drowsiness Detection System

Real-time driver drowsiness detection using OpenCV, MediaPipe Face Mesh, and a browser-based MediaPipe demo.

## Live Demo

Try the browser version here:

https://hargunsiingh.github.io/driver-drowsiness-detection/

The live demo runs fully in the browser. Camera frames stay on the user's device and are processed client-side with MediaPipe face landmarks.

## Features

| Feature | How it works |
|---|---|
| Eye Closure Detection | Calculates Eye Aspect Ratio (EAR) from 6 landmarks per eye |
| Yawn Detection | Calculates Mouth Aspect Ratio (MAR) from lip landmarks |
| Head Tilt Detection | Tracks face orientation changes across consecutive frames |
| Desktop Audio Alarm | Two-tone pulsing alert via pygame in the Python app |
| Visual Alerts | Green/yellow/red state indicators, overlay landmarks, and warning display |
| Face Mesh Overlay | Live wireframe showing face oval, eye, and mouth outlines |

## Browser Demo

The deployable demo is a static site:

```text
index.html
styles.css
app.js
face_landmarker.task
```

It can be hosted on GitHub Pages without a backend because webcam access and model inference happen in the visitor's browser.

## Python Requirements

- Python 3.8+
- Webcam

## Python Setup

```bash
pip install -r requirements.txt
```

## Python Usage

```bash
python main.py
```

- Green border: driver is awake
- Yellow border: warning, such as yawning or brief eye closure
- Red border + alarm: drowsy, such as prolonged eye closure or multiple fatigue indicators
- Press `q` to quit

## Configuration

Thresholds can be tuned in `detector.py`:

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | 0.22 | EAR below this means eyes closed |
| `MAR_THRESHOLD` | 0.75 | MAR above this means yawning |
| `HEAD_TILT_THRESHOLD` | 30 deg | Pitch above this means head drooping in the Python app |
| `CONSEC_FRAMES_DROWSY` | 20 | Frames of eye closure before drowsy alert |

## Project Structure

```text
driver-drowsiness-detection/
|-- index.html            # Browser demo entry point
|-- styles.css            # Browser demo styling
|-- app.js                # Browser demo detection logic
|-- main.py               # Python entry point: webcam loop and UI
|-- detector.py           # Python detection engine
|-- alert.py              # Python audio alarm manager
|-- face_landmarker.task  # MediaPipe model
|-- requirements.txt      # Python dependencies
`-- README.md
```
