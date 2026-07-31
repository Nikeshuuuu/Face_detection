# Real-Time Object, Person & Face Detection with YOLOv11

A real-time computer vision app that uses **YOLOv11** to simultaneously perform:
- **Object detection + multi-object tracking** (via ByteTrack) using `yolo11l.pt`
- **Face detection** using `yolo11l-face.pt`

Both run live on a webcam feed, with on-screen overlays for bounding boxes, track IDs, confidence scores, FPS, person count, and face count.



## Features

- Live object detection with class labels and confidence scores
- Persistent multi-object tracking (ByteTrack) with unique track IDs
- Dedicated face detection model running alongside object detection
- Real-time FPS counter
- Live person and face counters
- Apple Silicon (MPS) GPU acceleration support, with automatic CPU fallback

## Requirements

- Python 3.9+
- A webcam
- macOS with Apple Silicon (for MPS acceleration) — otherwise runs on CPU

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Model Weights

This project needs two YOLOv11 weight files, placed in the project root:

| File | Source |
|---|---|
| `yolo11l.pt` | Downloads automatically via `ultralytics` on first run |
| `yolo11l-face.pt` | Download manually from [link to source/repo] and place in the project root |

> Weight files are excluded from this repo via `.gitignore` since they're large binaries.

## Usage

```bash
python main.py
```

- Press **`x`** to quit the application.

## Project Structure

```
.
├── main.py            # Main detection + tracking script
├── requirements.txt    # Python dependencies
├── README.md
└── .gitignore
```

## How It Works

1. Captures frames from the default webcam (`cv2.VideoCapture(0)`).
2. Runs `object_model.track()` with ByteTrack for persistent object tracking.
3. Runs `face_model()` separately for face detection.
4. Draws bounding boxes, labels, track IDs, and confidence scores on a copy of the frame.
5. Overlays FPS, person count, and face count.
6. Displays the annotated frame in a live OpenCV window.

## Tech Stack

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/) (with MPS support for Apple Silicon)
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (via Ultralytics tracker integration)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Ultralytics for the YOLOv11 models and framework
- ByteTrack for the tracking algorithm
