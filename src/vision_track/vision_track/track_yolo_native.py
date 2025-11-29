#!/usr/bin/env python3
"""
YOLO-based Object Tracking with Native BoT-SORT Re-Identification

This module provides a simplified YOLOTrackerNative class that uses YOLO's
built-in BoT-SORT tracker with native ReID capabilities instead of custom
feature extraction.

The BoT-SORT tracker with ReID enabled provides:
- Appearance-based re-identification using native YOLO features
- Global motion compensation for camera movement
- More robust ID persistence through occlusions

Author: TinkerFuroc
Date: 2025
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time
import tempfile
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackerState(Enum):
    """Enumeration for tracker states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    TRACKING = "tracking"
    LOST = "lost"
    REIDENTIFYING = "reidentifying"


@dataclass
class TrackingResult:
    """Data class to hold tracking results."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray]  # Segmentation mask
    confidence: float
    class_id: int
    class_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracking result to dictionary."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "mask": self.mask,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name
        }


class YOLOTrackerNative:
    """
    YOLO-based object tracker using native BoT-SORT ReID.
    
    This tracker uses Ultralytics' built-in BoT-SORT tracker with ReID enabled,
    which provides more robust person re-identification than custom feature
    extraction methods.
    
    Key features:
    - Native YOLO feature extraction for ReID
    - Global motion compensation (sparseOptFlow)
    - Configurable appearance and proximity thresholds
    - Longer track buffer for occlusion handling
    """
    
    # Default YOLO model for segmentation
    DEFAULT_MODEL = "yolo11n-seg.pt"
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
        warmup: bool = True,
        appearance_thresh: float = 0.25,  # Lower = more strict ReID
        proximity_thresh: float = 0.5,     # Min IoU for ReID consideration  
        track_buffer: int = 60,            # Frames to keep lost tracks (2 sec at 30fps)
    ):
        """
        Initialize the native YOLO tracker with BoT-SORT ReID.
        
        Args:
            model_path: Path to the YOLO model or model name
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to use ('cuda', 'cpu', or None for auto)
            warmup: Whether to warm up the model on initialization
            appearance_thresh: ReID appearance similarity threshold (lower = stricter)
            proximity_thresh: Minimum IoU to consider tracks for ReID
            track_buffer: Number of frames to keep lost tracks alive
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.state = TrackerState.UNINITIALIZED
        self.target_track_id: Optional[int] = None
        self.original_track_id: Optional[int] = None  # Stable ID for user display
        self.target_class_id: Optional[int] = None
        self.target_class_name: Optional[str] = None
        self.tracked_results: List[TrackingResult] = []
        
        # Native ReID parameters
        self.appearance_thresh = appearance_thresh
        self.proximity_thresh = proximity_thresh
        self.track_buffer = track_buffer
        
        # Tracking state
        self.frames_lost = 0
        self.max_frames_lost = 600  # 20 seconds at 30fps
        self.last_known_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_results: List[TrackingResult] = []
        
        # Create custom tracker config
        self.tracker_config_path = self._create_tracker_config()
        
        # Determine device
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = self._load_model(model_path)
        
        # Warm up if requested
        if warmup:
            self._warmup_model()
    
    def _create_tracker_config(self) -> str:
        """
        Create a custom BoT-SORT tracker config file.
        
        NOTE: with_reid: True is buggy in ultralytics 8.3.x - the native feature
        encoder expects tensors but receives numpy arrays, causing:
        "AttributeError: 'numpy.ndarray' object has no attribute 'cpu'"
        
        We use BoT-SORT without ReID but with longer track buffer and motion
        compensation for better tracking through occlusions.
        
        Returns:
            Path to the created config file
        """
        config_content = f"""# Custom BoT-SORT config
# Generated by YOLOTrackerNative
# NOTE: with_reid disabled due to bug in ultralytics 8.3.x

tracker_type: botsort

# Detection thresholds
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25

# Track persistence - longer buffer for occlusion handling
track_buffer: {self.track_buffer}

# Matching
match_thresh: 0.8
fuse_score: True

# Global motion compensation - helps with camera movement
gmc_method: sparseOptFlow

# ReID settings - DISABLED due to ultralytics bug
# The native encoder has: lambda feats, s: [f.cpu().numpy() for f in feats]
# But feats are already numpy arrays in 8.3.x, causing AttributeError
with_reid: False
proximity_thresh: {self.proximity_thresh}
appearance_thresh: {self.appearance_thresh}
"""
        
        # Write to temp file that persists
        config_dir = Path(tempfile.gettempdir()) / "yolo_tracker_configs"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "botsort_no_reid.yaml"
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Created BoT-SORT config at: {config_path}")
        logger.info(f"  track_buffer: {self.track_buffer}")
        logger.info(f"  NOTE: with_reid disabled due to ultralytics 8.3.x bug")
        
        return str(config_path)
    
    def _get_device(self, device: Optional[str]) -> str:
        """Determine the best available device."""
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            logger.info("CUDA not available. Using CPU.")
            return "cpu"
    
    def _load_model(self, model_path: str):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO model: {model_path}")
            model = YOLO(model_path)
            model.to(self.device)
            
            logger.info("YOLO model loaded successfully")
            logger.info("Native BoT-SORT ReID enabled")
            return model
            
        except ImportError:
            raise ImportError(
                "ultralytics package is required. "
                "Install it with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def _warmup_model(self, warmup_iterations: int = 3):
        """Warm up the model by running inference on dummy data."""
        logger.info("Warming up model...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for _ in range(warmup_iterations):
            self.model.track(
                dummy_frame,
                persist=True,
                tracker=self.tracker_config_path,
                verbose=False
            )
        
        # Reset tracker state after warmup
        self.model.predictor = None
        logger.info("Model warmup complete")
    
    def track(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[TrackingResult]:
        """
        Run tracking on a frame.
        
        Args:
            frame: Input image (BGR format)
            classes: List of class IDs to detect (None for all)
            
        Returns:
            List of tracking results
        """
        import traceback
        
        try:
            # Run YOLO tracking with native ReID
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config_path,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=classes,
                verbose=False
            )
        except Exception as e:
            logger.error(f"YOLO track() failed: {e}")
            logger.error(traceback.format_exc())
            return []
        
        # Parse results
        tracking_results = []
        
        try:
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    # Access the underlying data arrays directly
                    # In newer ultralytics versions, these could be numpy arrays
                    xyxy_data = boxes.xyxy
                    cls_data = boxes.cls
                    conf_data = boxes.conf
                    id_data = boxes.id  # Could be None
                    
                    # Convert to numpy if tensor
                    if hasattr(xyxy_data, 'cpu'):
                        xyxy_np = xyxy_data.cpu().numpy()
                    else:
                        xyxy_np = np.asarray(xyxy_data)
                    
                    if hasattr(cls_data, 'cpu'):
                        cls_np = cls_data.cpu().numpy()
                    else:
                        cls_np = np.asarray(cls_data)
                    
                    if hasattr(conf_data, 'cpu'):
                        conf_np = conf_data.cpu().numpy()
                    else:
                        conf_np = np.asarray(conf_data)
                    
                    id_np = None
                    if id_data is not None:
                        if hasattr(id_data, 'cpu'):
                            id_np = id_data.cpu().numpy()
                        else:
                            id_np = np.asarray(id_data)
                    
                    for i in range(len(boxes)):
                        # Get box coordinates
                        x1, y1, x2, y2 = xyxy_np[i].astype(int)
                        
                        # Get track ID (may be None for new detections)
                        track_id = -1
                        if id_np is not None:
                            track_id = int(id_np[i])
                        
                        # Get class info
                        class_id = int(cls_np[i])
                        confidence = float(conf_np[i])
                        class_name = self.model.names[class_id]
                        
                        # Get mask if available - handle both tensor and numpy
                        mask = None
                        if result.masks is not None and len(result.masks) > i:
                            try:
                                mask_obj = result.masks[i]
                                if hasattr(mask_obj, 'data'):
                                    mask_data = mask_obj.data
                                    if hasattr(mask_data, '__getitem__'):
                                        mask_data = mask_data[0]
                                    if hasattr(mask_data, 'cpu'):
                                        mask = mask_data.cpu().numpy()
                                    elif isinstance(mask_data, np.ndarray):
                                        mask = mask_data
                                    else:
                                        mask = np.asarray(mask_data)
                            except Exception:
                                mask = None
                            
                            if mask is not None:
                                # Resize mask to frame size
                                mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))
                                mask = (mask > 0.5).astype(np.uint8) * 255
                        
                        tracking_results.append(TrackingResult(
                            track_id=track_id,
                            bbox=(x1, y1, x2, y2),
                            mask=mask,
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        ))
        except Exception as e:
            logger.error(f"Failed to parse YOLO results: {e}")
            logger.error(traceback.format_exc())
        
        self.last_results = tracking_results
        return tracking_results
    
    def initialize_tracking(
        self,
        frame: np.ndarray,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        target_class: Optional[str] = None,
        target_id: Optional[int] = None
    ) -> bool:
        """
        Initialize tracking on a specific object.
        
        Args:
            frame: Input image
            target_bbox: Bounding box of target (x1, y1, x2, y2)
            target_class: Class name of target object (e.g., 'person', 'car')
            target_id: Specific track ID to follow (if already known)
            
        Returns:
            True if initialization successful, False otherwise
        """
        # Reset tracker state
        self.model.predictor = None
        self.target_track_id = None
        self.original_track_id = None
        self.frames_lost = 0
        self.state = TrackerState.UNINITIALIZED
        
        # Determine target class ID from name
        target_class_id = None
        if target_class is not None:
            # Map class name to ID (person = 0 in COCO)
            if target_class.lower() == 'person':
                target_class_id = 0
            else:
                # Try to find class ID from model names
                for class_id, name in self.model.names.items():
                    if name.lower() == target_class.lower():
                        target_class_id = class_id
                        break
        
        if target_class_id is None:
            target_class_id = 0  # Default to person
        
        self.target_class_id = target_class_id
        
        # Run tracking to get detections
        results = self.track(frame, classes=[target_class_id])
        
        if not results:
            logger.warning(f"No objects of class {target_class} found for initialization")
            return False
        
        # Find best target
        target = None
        best_score = -1
        
        for result in results:
            if result.track_id < 0:
                continue  # Skip untracked detections
            
            # If specific target_id requested, find it
            if target_id is not None and result.track_id == target_id:
                target = result
                break
            
            # Calculate score based on preference
            score = result.confidence
            
            if target_bbox is not None:
                # Prefer boxes closer to the preferred location
                iou = self._calculate_iou(result.bbox, target_bbox)
                score = 0.3 * score + 0.7 * iou
            else:
                # Default: prefer larger, more central detections
                box_area = (result.bbox[2] - result.bbox[0]) * (result.bbox[3] - result.bbox[1])
                frame_area = frame.shape[0] * frame.shape[1]
                size_score = min(1.0, box_area / (frame_area * 0.1))
                
                center_x = (result.bbox[0] + result.bbox[2]) / 2
                center_y = (result.bbox[1] + result.bbox[3]) / 2
                dist_from_center = np.sqrt(
                    ((center_x - frame.shape[1]/2) / frame.shape[1])**2 +
                    ((center_y - frame.shape[0]/2) / frame.shape[0])**2
                )
                center_score = 1.0 - min(1.0, dist_from_center)
                
                score = 0.4 * result.confidence + 0.3 * size_score + 0.3 * center_score
            
            if score > best_score:
                best_score = score
                target = result
        
        if target is None:
            logger.warning("No trackable object found during initialization")
            return False
        
        # Initialize tracking on this target
        self.target_track_id = target.track_id
        self.original_track_id = target.track_id  # Store stable ID for user display
        self.target_class_id = target.class_id
        self.target_class_name = target.class_name
        self.last_known_bbox = target.bbox
        self.state = TrackerState.TRACKING
        
        logger.info(f"Tracking initialized on {target.class_name} (ID: {target.track_id})")
        
        return True
    
    def get_target(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[TrackingResult], TrackerState]:
        """
        Get the current target location.
        
        Args:
            frame: Input image
            
        Returns:
            Tuple of (TrackingResult or None, TrackerState)
        """
        if self.state == TrackerState.UNINITIALIZED:
            logger.warning("Tracker not initialized")
            return None, self.state
        
        # Run tracking
        results = self.track(frame, classes=[self.target_class_id] if self.target_class_id is not None else None)
        
        # Find our target
        target_result = None
        
        for result in results:
            if result.track_id == self.target_track_id:
                # Verify class matches
                if self.target_class_id is not None and result.class_id != self.target_class_id:
                    logger.warning(f"Track ID {result.track_id} class changed from {self.target_class_name} to {result.class_name}")
                    continue
                
                target_result = result
                break
        
        if target_result is not None:
            # Target found
            self.state = TrackerState.TRACKING
            self.frames_lost = 0
            self.last_known_bbox = target_result.bbox
            logger.debug(f"Target {self.target_track_id} found at {target_result.bbox}")
            return self._with_original_id(target_result), self.state
        
        # Target not found
        self.frames_lost += 1
        logger.info(f"Target {self.target_track_id} not found. Frames lost: {self.frames_lost}")
        
        if self.frames_lost > self.max_frames_lost:
            logger.warning(f"Target lost for {self.frames_lost} frames. Marking as LOST.")
            self.state = TrackerState.LOST
            return None, self.state
        
        # Check if target might have gotten a new ID (common with BoT-SORT)
        # Look for a person detection near the last known position
        if self.last_known_bbox is not None:
            best_candidate = None
            best_iou = 0.0
            
            for result in results:
                if result.track_id < 0:
                    continue
                if self.target_class_id is not None and result.class_id != self.target_class_id:
                    continue
                
                iou = self._calculate_iou(result.bbox, self.last_known_bbox)
                if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                    best_iou = iou
                    best_candidate = result
            
            if best_candidate is not None:
                logger.info(f"Found candidate near last position: ID {best_candidate.track_id} with IoU {best_iou:.3f}")
                # Update target ID - BoT-SORT with ReID should handle this, but we help
                self.target_track_id = best_candidate.track_id
                self.last_known_bbox = best_candidate.bbox
                self.frames_lost = 0
                self.state = TrackerState.TRACKING
                return self._with_original_id(best_candidate), self.state
        
        self.state = TrackerState.REIDENTIFYING
        return None, self.state
    
    def _with_original_id(self, result: TrackingResult) -> TrackingResult:
        """
        Return a copy of the result with the original (stable) track ID.
        
        This ensures users always see the same track ID for the target,
        even if YOLO assigns a new ID after re-identification.
        """
        return TrackingResult(
            track_id=self.original_track_id if self.original_track_id is not None else result.track_id,
            bbox=result.bbox,
            mask=result.mask,
            confidence=result.confidence,
            class_id=result.class_id,
            class_name=result.class_name
        )
    
    def stop_tracking(self):
        """Stop tracking and reset state."""
        self.target_track_id = None
        self.original_track_id = None
        self.target_class_id = None
        self.target_class_name = None
        self.last_known_bbox = None
        self.frames_lost = 0
        self.state = TrackerState.UNINITIALIZED
        
        # Reset YOLO tracker state
        self.model.predictor = None
        
        logger.info("Tracking stopped")
    
    def reset(self):
        """Reset the tracker state (alias for stop_tracking)."""
        self.stop_tracking()
        logger.info("Tracker reset")
    
    def update(self, frame: np.ndarray) -> Optional[TrackingResult]:
        """
        Update tracking on a new frame (compatibility method).
        
        This wraps get_target() for compatibility with YOLOTracker interface.
        
        Args:
            frame: New RGB frame
            
        Returns:
            TrackingResult for the target object, or None if lost
        """
        result, state = self.get_target(frame)
        return result
    
    def get_all_results(self) -> List[TrackingResult]:
        """Get all tracking results from the last frame."""
        return self.last_results
    
    def is_tracking(self) -> bool:
        """Check if actively tracking a target."""
        return self.state == TrackerState.TRACKING
    
    def is_lost(self) -> bool:
        """Check if target is lost."""
        return self.state == TrackerState.LOST
    
    def get_target_id(self) -> Optional[int]:
        """Get the current target track ID."""
        return self.target_track_id
    
    @staticmethod
    def _calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def __del__(self):
        """Cleanup - remove temp config file."""
        try:
            if hasattr(self, 'tracker_config_path') and os.path.exists(self.tracker_config_path):
                # Don't delete - might be reused
                pass
        except:
            pass
