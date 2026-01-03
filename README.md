# YOLO8 Sponsor Detection Pipeline

**Automated sponsor logo detection and exposure metrics computation for squash men's final.**

<img width="1470" height="956" alt="Screenshot 2026-01-03 at 00 21 05" src="https://github.com/user-attachments/assets/a827edae-345a-4c50-992d-2f17b39d2c52" />

<img width="1470" height="956" alt="Screenshot 2026-01-03 at 00 21 55" src="https://github.com/user-attachments/assets/ab577bd1-2f7d-4dd7-819e-f0ebffe9c4e6" />


## Overview

This pipeline analyzes broadcast video segments to quantify sponsor exposure using computer vision. It detects sponsor logos throughout the video, tracks their appearance duration, computes prominence metrics, and generates comprehensive analytics reports for sponsor valuation.

The system is designed for sports broadcast analysis, with support for multiple sponsor categories (host, gold, silver, bronze, jersey) and comprehensive metrics including Share of Voice (SOV), exposure duration, and visual prominence.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               Input: Video + Config                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Frame Processing (Sampling)                    │
│  • Extract frames at specified FPS                          │
│  • Support time-based segment selection                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Logo Detection (YOLO Model)                    │
│  • Run object detection on each frame                       │
│  • Map detected classes to brand IDs                        │
│  • Filter by confidence threshold                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Temporal Smoothing & Validation                     │
│  • Confirm detections across multiple frames                │
│  • Filter out transient false positives                     │
│  • Maintain detection continuity                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         Exposure Segment Computation                        │
│  • Group continuous detections into segments                │
│  • Calculate segment duration and metrics                   │
│  • Compute average confidence and prominence                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Metrics & KPI Generation                       │
│  • Brand-level exposure duration                            │
│  • Share of Voice (SOV) calculations                        │
│  • Average continuous exposure                              │
│  • Visual prominence (area ratio)                           │
│  • Detection confidence metrics                             │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Files                             │
│  • Frame-level predictions (CSV)                            │
│  • Exposure segments (CSV)                                  │
│  • Brand KPIs (CSV)                                         │
│  • Annotated video (optional)                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Automated Logo Detection**: Deep learning-based object detection for sponsor logos
- **Temporal Smoothing**: Prevents flicker and false positives through multi-frame confirmation
- **Multi-Category Support**: Handles host, gold, silver, bronze, and jersey sponsor categories
- **Comprehensive Metrics**: Exposure duration, SOV, prominence, confidence scores
- **Flexible Configuration**: JSON-based brand and detection parameter configuration
- **Annotated Video Output**: Visual validation with detection bounding boxes
- **Frame-Level Tracking**: Detailed per-frame detection data for analysis

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

### Setup

1. Clone or download this repository

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- `ultralytics` - YOLO model inference
- `opencv-python` - Video processing
- `numpy` - Numerical operations
- `pandas` - Data processing and CSV export

## Quick Start

### Basic Usage

```bash
python3 yolo_pipeline.py \
  --video "2025 China Open Squash Mens Final Segment.mp4" \
  --config brand_dictionary.json \
  --fps 25 \
  --out_dir outputs \
  --save_annotated_video
```

### Process Specific Time Segment

```bash
python3 yolo_pipeline.py \
  --video "2025 China Open Squash Mens Final Segment.mp4" \
  --config brand_dictionary.json \
  --t_start 100 \
  --t_end 120 \
  --fps 25 \
  --out_dir outputs
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--video` | Input video file path (required) | - |
| `--config` | Brand configuration JSON file (required) | - |
| `--t_start` | Start time in seconds | 0 |
| `--t_end` | End time in seconds | End of video |
| `--fps` | Processing frame rate | 10.0 |
| `--out_dir` | Output directory | `outputs` |
| `--save_annotated_video` | Save annotated video with detections | False |

## Configuration

The `brand_dictionary.json` file configures:

### Detector Configuration

```json
{
  "detector_config": {
    "type": "yolo",
    "model_path": "runs/detect/squash_sponsors_v2/weights/best.pt",
    "min_confidence": 0.40,
    "enabled": true
  }
}
```

### Temporal Smoothing

```json
{
  "temporal_smoothing": {
    "enabled": true,
    "confirm_frames": 2,
    "drop_frames": 15,
    "max_label_switch_per_second": 3
  }
}
```

- `confirm_frames`: Number of consecutive frames required to confirm a detection
- `drop_frames`: Frames without detection before dropping a brand
- `max_label_switch_per_second`: Maximum label switches allowed per second

### Brand Definition

Each brand entry maps YOLO detection classes to brand identifiers:

```json
{
  "brand_id": "GOLD_JOYUS",
  "display_name": "JOYUS",
  "category": "gold",
  "yolo_classes": ["joyus_logo"]
}
```

**Categories:**
- `host`: Event hosts/organizers
- `gold`: Primary sponsors (highest tier)
- `silver`: Secondary sponsors
- `bronze`: Tertiary sponsors
- `jersey`: Player jersey sponsors

### Reporting Configuration

```json
{
  "reporting": {
    "include_hosts_in_sov": false,
    "include_jersey_in_sov": false,
    "include_other_in_sov": false
  }
}
```

Controls which categories are included in Share of Voice calculations.

## Output Files

### 1. `frame_level_predictions.csv`

Detailed detection data for every frame:
- `frame_idx`: Frame number
- `time`: Timestamp in seconds
- `brand_id`: Brand identifier
- `display_name`: Brand display name
- `category`: Sponsor category
- `confidence`: Detection confidence score
- `bbox_x1, bbox_y1, bbox_x2, bbox_y2`: Bounding box coordinates
- `area_ratio`: Normalized area (prominence indicator)

### 2. `exposure_segments.csv`

Continuous exposure segments:
- `brand_id`, `display_name`, `category`
- `start_time`, `end_time`: Segment time range
- `duration_sec`: Total exposure duration
- `start_frame`, `end_frame`: Frame range
- `avg_confidence`: Average detection confidence
- `avg_area_ratio`: Average visual prominence

### 3. `brand_kpis.csv`

Brand-level key performance indicators:

**KPI 1: Exposure Duration**
- `num_appearances`: Number of distinct exposure segments
- `total_exposure_time_sec`: Total exposure time across all segments
- `avg_exposure_duration_sec`: Average duration per continuous segment

**KPI 2: Size & Prominence**
- `avg_area_percentage`: Average logo size as percentage of screen (0-100%)
- `prominence_score`: Overall prominence metric (0-1) combining size, clarity, and position

**KPI 3: Clarity**
- `avg_sharpness_score`: Logo sharpness/clarity score (0-1, higher = clearer)

**KPI 4: Share of Voice (SOV)**
- `SOV_all_sponsors`: Share of Voice among all sponsors (%)
- `SOV_category`: Share of Voice within brand category (%)

### Model Performance

- **mAP@50**: 76.2% (v2 model)
- **Classes**: 17 sponsor logo classes
- **Inference Speed**: ~65ms/frame on CPU

## Performance

- **Processing Speed**: 2-3 frames/second on CPU (depends on video resolution and FPS setting)
- **Memory Usage**: Low - processes frames sequentially
- **Accuracy**: Depends on model performance and confidence thresholds
- **Scalability**: Can process videos of any length with time-based segmentation

## Project Structure

```
.
├── yolo_pipeline.py          # Main pipeline script
├── brand_dictionary.json      # Brand configuration
├── requirements.txt           # Python dependencies
├── start_training.sh         # Model training script
├── datasets/                  # Training datasets
│   └── squash-sponsor-detection_v2/
├── runs/                      # Model training outputs
│   └── detect/
│       └── squash_sponsors_v2/
│           └── weights/
│               └── best.pt    # Trained model
└── outputs/                   # Analysis results
    ├── frame_level_predictions.csv
    ├── exposure_segments.csv
    ├── brand_kpis.csv
    └── annotated_video.mp4
```

## Future Enhancements

- [ ] GPU acceleration optimization
- [ ] Batch processing for multiple videos
- [ ] Advanced visualization dashboards
