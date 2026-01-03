#!/usr/bin/env python3
"""
YOLO-Only Sponsor Analytics Pipeline
Simple, clean pipeline that trusts the YOLO model.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

# Simple logging functions
def log_info(msg: str):
    """Print info message."""
    print(f"ℹ️  {msg}")

def log_success(msg: str):
    """Print success message."""
    print(f"✅ {msg}")

def log_warning(msg: str):
    """Print warning message."""
    print(f"⚠️  {msg}")

def log_error(msg: str):
    """Print error message."""
    print(f"❌ {msg}", file=sys.stderr)


@dataclass
class Brand:
    """Brand configuration."""
    brand_id: str
    display_name: str
    category: str
    yolo_classes: List[str]


@dataclass
class Detection:
    """Single detection."""
    brand_id: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    frame_idx: int
    time: float
    sharpness: float = 0.0  # Clarity score
    center_weight: float = 0.0  # Position weighting


@dataclass
class ExposureSegment:
    """Continuous exposure segment."""
    brand_id: str
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int
    avg_confidence: float
    avg_area_ratio: float
    avg_area_percentage: float = 0.0  # Area as % of screen
    avg_sharpness: float = 0.0  # Clarity score
    avg_center_weight: float = 0.0  # Position prominence
    prominence_score: float = 0.0  # Combined prominence metric


class TemporalSmoother:
    """Simple temporal smoothing to prevent flicker."""
    
    def __init__(self, confirm_frames: int = 2, drop_frames: int = 15):
        self.confirm_frames = confirm_frames
        self.drop_frames = drop_frames
        self.active_detections: Dict[str, Dict] = {}  # brand_id -> {frames: [], last_seen: int}
    
    def update(self, frame_idx: int, detections: List[Detection]) -> List[Detection]:
        """Update with new detections and return confirmed ones."""
        # Track which brands were detected this frame
        detected_brands = {d.brand_id for d in detections}
        
        # Update active detections
        for brand_id in list(self.active_detections.keys()):
            if brand_id in detected_brands:
                # Brand still detected
                self.active_detections[brand_id]['frames'].append(frame_idx)
                self.active_detections[brand_id]['last_seen'] = frame_idx
            else:
                # Brand not detected - check if we should drop it
                frames_since_last = frame_idx - self.active_detections[brand_id]['last_seen']
                if frames_since_last > self.drop_frames:
                    del self.active_detections[brand_id]
        
        # Add new detections
        for det in detections:
            if det.brand_id not in self.active_detections:
                self.active_detections[det.brand_id] = {
                    'frames': [frame_idx],
                    'last_seen': frame_idx,
                    'detections': []
                }
            self.active_detections[det.brand_id]['detections'].append(det)
        
        # Return confirmed detections (seen for at least confirm_frames)
        confirmed = []
        for brand_id, info in self.active_detections.items():
            if len(info['frames']) >= self.confirm_frames:
                # Use the most recent detection
                if info['detections']:
                    confirmed.append(info['detections'][-1])
        
        return confirmed


class YOLOAnalytics:
    """YOLO-only sponsor analytics pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize with config."""
        self.config = self._load_config(config_path)
        self.brands = self._load_brands()
        self.yolo_class_to_brand = self._build_class_mapping()
        
        # Load YOLO model
        model_path = self.config['detector_config']['model_path']
        log_info(f"Loading YOLO model: {Path(model_path).name}")
        self.model = YOLO(model_path)
        self.min_confidence = self.config['detector_config']['min_confidence']
        
        # Temporal smoother
        ts_config = self.config.get('temporal_smoothing', {})
        self.smoother = TemporalSmoother(
            confirm_frames=ts_config.get('confirm_frames', 2),
            drop_frames=ts_config.get('drop_frames', 15)
        )
        
        # Storage
        self.all_detections: List[Detection] = []
        self.frame_detections: Dict[int, List[Detection]] = defaultdict(list)
        
        log_success(f"Initialized with {len(self.brands)} brands")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load JSON config."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _load_brands(self) -> List[Brand]:
        """Load brand configurations."""
        brands = []
        for b in self.config['brands']:
            brands.append(Brand(
                brand_id=b['brand_id'],
                display_name=b['display_name'],
                category=b['category'],
                yolo_classes=b['yolo_classes']
            ))
        return brands
    
    def _build_class_mapping(self) -> Dict[str, Brand]:
        """Build mapping from YOLO class names to Brand objects."""
        mapping = {}
        for brand in self.brands:
            for yolo_class in brand.yolo_classes:
                if yolo_class in mapping:
                    log_warning(f"YOLO class '{yolo_class}' mapped to multiple brands!")
                mapping[yolo_class] = brand
        return mapping
    
    def process_video(self, video_path: str, t_start: float = 0, t_end: Optional[float] = None, 
                     fps: float = 10.0, out_dir: str = "outputs", save_annotated: bool = False):
        """Process video and generate analytics."""
        out_path = Path(out_dir)
        out_path.mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range
        start_frame = int(t_start * video_fps)
        end_frame = int(t_end * video_fps) if t_end else total_frames
        frame_skip = max(1, int(video_fps / fps))
        num_frames_to_process = (end_frame - start_frame) // frame_skip
        
        print(f"\n📹 Video: {width}x{height} @ {video_fps:.1f}fps ({total_frames} frames)")
        if t_end:
            print(f"⏱️  Segment: {t_start:.1f}s - {t_end:.1f}s ({num_frames_to_process} frames @ {fps}fps)")
        else:
            print(f"⏱️  Segment: {t_start:.1f}s - end ({num_frames_to_process} frames @ {fps}fps)")
        print()
        
        # Set video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Video writer for annotated output
        writer = None
        if save_annotated:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video_path = out_path / "annotated_video.mp4"
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = start_frame
        processed_count = 0
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if (frame_idx - start_frame) % frame_skip != 0:
                frame_idx += 1
                continue
            
            time = frame_idx / video_fps
            
            # Run YOLO detection
            detections = self._detect_frame(frame, frame_idx, time)
            
            # Apply temporal smoothing
            confirmed = self.smoother.update(frame_idx, detections)
            
            # Store detections
            self.frame_detections[frame_idx] = confirmed
            self.all_detections.extend(confirmed)
            
            # Draw annotations if requested
            if save_annotated:
                annotated = self._draw_annotations(frame.copy(), confirmed)
                writer.write(annotated)
            
            processed_count += 1
            # Show progress every 50 frames
            if processed_count % 50 == 0:
                progress = min(100.0, (processed_count / num_frames_to_process) * 100)
                print(f"  Progress: {processed_count}/{num_frames_to_process} frames ({progress:.1f}%) - {len(confirmed)} detections", end='\r')
            
            frame_idx += 1
        
        cap.release()
        if writer:
            writer.release()
            log_success(f"Saved annotated video: {out_video_path.name}")
        
        # Clear progress line and print final summary
        print()  # New line after progress
        print(f"✅ Processed {processed_count} frames, {len(self.all_detections)} total detections\n")
        
        # Generate outputs
        self._generate_outputs(out_path, video_fps)
    
    def _compute_sharpness(self, roi: np.ndarray) -> float:
        """Compute sharpness score using Laplacian variance."""
        if roi.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Compute Laplacian variance (higher = sharper)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (using sigmoid-like function)
        # Typical sharp images have variance > 100, blurry < 50
        normalized = min(1.0, variance / 200.0)
        
        return normalized
    
    def _compute_center_weighting(self, bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> float:
        """Compute center weighting (logos closer to center are more prominent)."""
        x1, y1, x2, y2 = bbox
        
        # Get center of bbox
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Get frame center
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2
        
        # Compute distance from center (normalized)
        dx = (bbox_center_x - frame_center_x) / frame_width
        dy = (bbox_center_y - frame_center_y) / frame_height
        distance = np.sqrt(dx**2 + dy**2)
        
        # Convert to weight (1.0 at center, decreases with distance)
        weight = max(0.0, 1.0 - distance)
        
        return weight
    
    def _detect_frame(self, frame: np.ndarray, frame_idx: int, time: float) -> List[Detection]:
        """Run YOLO detection on frame."""
        results = self.model(frame, verbose=False)
        detections = []
        
        height, width = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.min_confidence:
                    continue
                
                # Get YOLO class name
                cls_id = int(box.cls[0])
                yolo_class = result.names[cls_id]
                
                # Map to brand
                if yolo_class not in self.yolo_class_to_brand:
                    continue
                
                brand = self.yolo_class_to_brand[yolo_class]
                
                # Get bbox
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                
                # Extract ROI for sharpness computation
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                roi = frame[y1:y2, x1:x2]
                
                # Compute metrics
                sharpness = self._compute_sharpness(roi)
                center_weight = self._compute_center_weighting(bbox, width, height)
                
                detections.append(Detection(
                    brand_id=brand.brand_id,
                    confidence=conf,
                    bbox=bbox,
                    frame_idx=frame_idx,
                    time=time,
                    sharpness=sharpness,
                    center_weight=center_weight
                ))
        
        return detections
    
    def _draw_annotations(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes on frame."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            brand = next(b for b in self.brands if b.brand_id == det.brand_id)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{brand.display_name} {det.confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def _generate_outputs(self, out_path: Path, video_fps: float):
        """Generate output CSV files."""
        # 1. Frame-level predictions
        frame_data = []
        for frame_idx, detections in self.frame_detections.items():
            time = frame_idx / video_fps
            for det in detections:
                brand = next(b for b in self.brands if b.brand_id == det.brand_id)
                x1, y1, x2, y2 = det.bbox
                
                # Calculate area metrics
                area_pixels = (x2 - x1) * (y2 - y1)
                area_ratio = area_pixels / (1920 * 1080)  # Normalized
                area_percentage = area_ratio * 100  # As percentage
                
                frame_data.append({
                    'frame_idx': frame_idx,
                    'time': time,
                    'brand_id': det.brand_id,
                    'display_name': brand.display_name,
                    'category': brand.category,
                    'confidence': det.confidence,
                    'bbox_x1': x1,
                    'bbox_y1': y1,
                    'bbox_x2': x2,
                    'bbox_y2': y2,
                    'area_pixels': area_pixels,
                    'area_percentage': area_percentage,
                    'sharpness': det.sharpness,
                    'center_weight': det.center_weight
                })
        
        if frame_data:
            df_frames = pd.DataFrame(frame_data)
            df_frames.to_csv(out_path / 'frame_level_predictions.csv', index=False)
            log_success(f"Frame predictions: {len(df_frames)} detections → {out_path / 'frame_level_predictions.csv'}")
        
        # 2. Exposure segments
        segments = self._compute_exposure_segments(video_fps)
        if segments:
            df_segments = pd.DataFrame([{
                'brand_id': s.brand_id,
                'display_name': next(b.display_name for b in self.brands if b.brand_id == s.brand_id),
                'category': next(b.category for b in self.brands if b.brand_id == s.brand_id),
                'start_time': s.start_time,
                'end_time': s.end_time,
                'duration_sec': s.duration,
                'start_frame': s.start_frame,
                'end_frame': s.end_frame,
                'avg_confidence': s.avg_confidence,
                'avg_area_percentage': s.avg_area_percentage,
                'avg_sharpness': s.avg_sharpness,
                'avg_center_weight': s.avg_center_weight,
                'prominence_score': s.prominence_score
            } for s in segments])
            df_segments.to_csv(out_path / 'exposure_segments.csv', index=False)
            log_success(f"Exposure segments: {len(segments)} segments → {out_path / 'exposure_segments.csv'}")
        
        # 3. Brand KPIs
        kpis = self._compute_brand_kpis(segments, video_fps)
        if kpis:
            df_kpis = pd.DataFrame(kpis)
            df_kpis.to_csv(out_path / 'brand_kpis.csv', index=False)
            log_success(f"Brand KPIs: {len(kpis)} brands → {out_path / 'brand_kpis.csv'}")
    
    def _compute_exposure_segments(self, video_fps: float) -> List[ExposureSegment]:
        """Compute continuous exposure segments."""
        segments = []
        
        # Group detections by brand
        by_brand = defaultdict(list)
        for det in self.all_detections:
            by_brand[det.brand_id].append(det)
        
        for brand_id, detections in by_brand.items():
            if not detections:
                continue
            
            # Sort by frame
            detections.sort(key=lambda d: d.frame_idx)
            
            # Find continuous segments
            current_segment = [detections[0]]
            
            for det in detections[1:]:
                # Check if continuous (within 1 second)
                gap = (det.frame_idx - current_segment[-1].frame_idx) / video_fps
                if gap <= 1.0:
                    current_segment.append(det)
                else:
                    # Save segment
                    if len(current_segment) >= 2:
                        segments.append(self._create_segment(current_segment, video_fps))
                    current_segment = [det]
            
            # Save last segment
            if len(current_segment) >= 2:
                segments.append(self._create_segment(current_segment, video_fps))
        
        return segments
    
    def _create_segment(self, detections: List[Detection], video_fps: float) -> ExposureSegment:
        """Create exposure segment from detections."""
        start_frame = detections[0].frame_idx
        end_frame = detections[-1].frame_idx
        start_time = detections[0].time
        end_time = detections[-1].time
        duration = end_time - start_time
        
        avg_confidence = np.mean([d.confidence for d in detections])
        
        # Calculate average area metrics
        areas = []
        area_percentages = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area_pixels = (x2 - x1) * (y2 - y1)
            area_ratio = area_pixels / (1920 * 1080)
            areas.append(area_ratio)
            area_percentages.append(area_ratio * 100)
        
        avg_area_ratio = np.mean(areas) if areas else 0.0
        avg_area_percentage = np.mean(area_percentages) if area_percentages else 0.0
        
        # Calculate clarity and position metrics
        avg_sharpness = np.mean([d.sharpness for d in detections])
        avg_center_weight = np.mean([d.center_weight for d in detections])
        
        # Compute prominence score (weighted combination)
        # Prominence = 40% size + 30% clarity + 20% position + 10% confidence
        prominence_score = (
            0.40 * min(1.0, avg_area_percentage / 10.0) +  # Normalize to ~10% screen = max
            0.30 * avg_sharpness +
            0.20 * avg_center_weight +
            0.10 * avg_confidence
        )
        
        return ExposureSegment(
            brand_id=detections[0].brand_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            start_frame=start_frame,
            end_frame=end_frame,
            avg_confidence=avg_confidence,
            avg_area_ratio=avg_area_ratio,
            avg_area_percentage=avg_area_percentage,
            avg_sharpness=avg_sharpness,
            avg_center_weight=avg_center_weight,
            prominence_score=prominence_score
        )
    
    def _compute_brand_kpis(self, segments: List[ExposureSegment], video_fps: float) -> List[Dict]:
        """Compute brand-level KPIs."""
        by_brand = defaultdict(list)
        for seg in segments:
            by_brand[seg.brand_id].append(seg)
        
        # Get reporting config
        reporting = self.config.get('reporting', {})
        include_hosts = reporting.get('include_hosts_in_sov', False)
        include_jersey = reporting.get('include_jersey_in_sov', False)
        
        kpis = []
        total_sponsor_time = 0.0
        
        for brand in self.brands:
            brand_segments = by_brand.get(brand.brand_id, [])
            
            if not brand_segments:
                continue
            
            # Include ALL brands in KPIs (hosts and jerseys are detected and should be reported)
            # Only exclude them from SOV calculations (handled below)
            total_duration = sum(s.duration for s in brand_segments)
            num_appearances = len(brand_segments)
            avg_continuous = total_duration / num_appearances if num_appearances > 0 else 0.0
            avg_confidence = np.mean([s.avg_confidence for s in brand_segments])
            avg_area_percentage = np.mean([s.avg_area_percentage for s in brand_segments])
            avg_sharpness = np.mean([s.avg_sharpness for s in brand_segments])
            avg_center_weight = np.mean([s.avg_center_weight for s in brand_segments])
            prominence_score = np.mean([s.prominence_score for s in brand_segments])
            
            # Only count gold/silver/bronze sponsors in total_sponsor_time for SOV
            if brand.category in ['gold', 'silver', 'bronze']:
                total_sponsor_time += total_duration
            
            kpis.append({
                'brand_id': brand.brand_id,
                'display_name': brand.display_name,
                'category': brand.category,
                'num_appearances': num_appearances,
                # KPI 1：出现时长（Total & Avg）
                'total_exposure_time_sec': total_duration,
                'avg_exposure_duration_sec': avg_continuous,
                # KPI 2：Logo大小与显著性（Size + Prominence）
                'avg_area_percentage': avg_area_percentage,
                'prominence_score': prominence_score,
                # KPI 3：清晰/模糊度（Clarity）
                'avg_sharpness_score': avg_sharpness,
                #'avg_center_weight': avg_center_weight,
                #'avg_confidence': avg_confidence
            })
        
        # KPI 4：Share of Voice (SOV)
        # Calculate SOV (only for brands that should be included)
        for kpi in kpis:
            brand = next(b for b in self.brands if b.brand_id == kpi['brand_id'])
            
            # SOV for all sponsors (only gold/silver/bronze)
            if brand.category in ['gold', 'silver', 'bronze'] and total_sponsor_time > 0:
                kpi['SOV_all_sponsors'] = (kpi['total_exposure_time_sec'] / total_sponsor_time) * 100.0
            else:
                kpi['SOV_all_sponsors'] = 0.0
            
            # Category SOV (calculate for all categories)
            category_segments = [s for s in segments 
                               if next(b.category for b in self.brands if b.brand_id == s.brand_id) == brand.category]
            category_time = sum(s.duration for s in category_segments)
            if category_time > 0:
                kpi['SOV_category'] = (kpi['total_exposure_time_sec'] / category_time) * 100.0
            else:
                kpi['SOV_category'] = 0.0
        
        return kpis


def main():
    parser = argparse.ArgumentParser(description='YOLO-Only Sponsor Analytics')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--config', required=True, help='Config JSON path')
    parser.add_argument('--t_start', type=float, default=0, help='Start time (seconds)')
    parser.add_argument('--t_end', type=float, default=None, help='End time (seconds)')
    parser.add_argument('--fps', type=float, default=10.0, help='Processing FPS')
    parser.add_argument('--out_dir', default='outputs', help='Output directory')
    parser.add_argument('--save_annotated_video', action='store_true', help='Save annotated video')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 YOLO-Only Sponsor Analytics Pipeline")
    print("=" * 60)
    print()
    
    # Initialize pipeline
    pipeline = YOLOAnalytics(args.config)
    
    # Process video
    pipeline.process_video(
        video_path=args.video,
        t_start=args.t_start,
        t_end=args.t_end,
        fps=args.fps,
        out_dir=args.out_dir,
        save_annotated=args.save_annotated_video
    )
    
    print()
    print("=" * 60)
    print("✨ Processing Complete!")
    print("=" * 60)
    print(f"📁 Output directory: {args.out_dir}")
    print()


if __name__ == '__main__':
    main()

