#!/usr/bin/env python3
"""
YOLO-based Object Tracking with Segmentation

This module provides a YOLOTracker class that uses YOLO models for object tracking
with segmentation capabilities. It supports GPU acceleration when available.

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackerState(Enum):
    """Enumeration for tracker states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    TRACKING = "tracking"
    LOST = "lost"


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


class YOLOTracker:
    """
    YOLO-based object tracker with segmentation support.
    
    This class provides functionality for:
    - Object detection and segmentation using YOLO
    - Object tracking across frames using ByteTrack
    - GPU acceleration when available
    
    Attributes:
        model: The YOLO model instance
        device: The device (CPU/GPU) used for inference
        state: Current state of the tracker
        target_track_id: ID of the object being tracked
    """
    
    # Default YOLO model for segmentation
    DEFAULT_MODEL = "yolo11n-seg.pt"
    
    # Supported YOLO segmentation models
    SUPPORTED_MODELS = [
        "yolo11n-seg.pt",
        "yolo11s-seg.pt", 
        "yolo11m-seg.pt",
        "yolo11l-seg.pt",
        "yolo11x-seg.pt",
        "yolov8n-seg.pt",
        "yolov8s-seg.pt",
        "yolov8m-seg.pt",
        "yolov8l-seg.pt",
        "yolov8x-seg.pt",
    ]
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
        warmup: bool = True
    ):
        """
        Initialize the YOLO tracker.
        
        Args:
            model_path: Path to the YOLO model or model name
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to use ('cuda', 'cpu', or None for auto)
            warmup: Whether to warm up the model on initialization
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.state = TrackerState.UNINITIALIZED
        self.target_track_id: Optional[int] = None
        self.target_class_id: Optional[int] = None
        self.tracked_results: List[TrackingResult] = []
        
        # Determine device
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = self._load_model(model_path)
        
        # Warm up the model if requested
        if warmup:
            self._warmup_model()
    
    def _get_device(self, device: Optional[str]) -> str:
        """
        Determine the best available device.
        
        Args:
            device: Requested device or None for auto-detection
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            logger.info("CUDA not available. Using CPU.")
            return "cpu"
    
    def _load_model(self, model_path: str):
        """
        Load the YOLO model.
        
        Args:
            model_path: Path to the model file or model name
            
        Returns:
            Loaded YOLO model
        """
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO model: {model_path}")
            model = YOLO(model_path)
            model.to(self.device)
            
            # Configure tracker settings
            logger.info("YOLO model loaded successfully")
            return model
            
        except ImportError:
            raise ImportError(
                "ultralytics package is required. "
                "Install it with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def _warmup_model(self, warmup_iterations: int = 3):
        """
        Warm up the model by running inference on dummy data.
        
        Args:
            warmup_iterations: Number of warmup iterations
        """
        logger.info("Warming up model...")
        
        # Create dummy input
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        
        for i in range(warmup_iterations):
            _ = self.model(dummy_input, verbose=False)
            
        logger.info("Model warmup complete")
    
    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[TrackingResult]:
        """
        Perform detection and segmentation on a single frame.
        
        Args:
            frame: Input RGB image as numpy array
            classes: List of class IDs to detect (None for all)
            
        Returns:
            List of TrackingResult objects
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False
        )
        
        return self._parse_results(results[0])
    
    def track(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None,
        persist: bool = True
    ) -> List[TrackingResult]:
        """
        Perform tracking with segmentation on a frame.
        
        Args:
            frame: Input RGB image as numpy array
            classes: List of class IDs to track (None for all)
            persist: Whether to persist tracks between frames
            
        Returns:
            List of TrackingResult objects with track IDs
        """
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=classes,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        self.tracked_results = self._parse_results(results[0])
        return self.tracked_results
    
    def _parse_results(self, result) -> List[TrackingResult]:
        """
        Parse YOLO results into TrackingResult objects.
        
        Args:
            result: YOLO result object
            
        Returns:
            List of TrackingResult objects
        """
        tracking_results = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return tracking_results
        
        boxes = result.boxes
        masks = result.masks
        names = result.names
        
        for i, box in enumerate(boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Get track ID (if available)
            track_id = int(box.id[0]) if box.id is not None else -1
            
            # Get confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = names[class_id]
            
            # Get segmentation mask (if available)
            mask = None
            if masks is not None and i < len(masks):
                mask = masks[i].data[0].cpu().numpy()
                # Resize mask to original image size
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (result.orig_shape[1], result.orig_shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                mask = (mask > 0.5).astype(np.uint8)
            
            tracking_results.append(TrackingResult(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                mask=mask,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name
            ))
        
        return tracking_results
    
    def initialize_tracking(
        self,
        frame: np.ndarray,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        target_class: Optional[str] = None
    ) -> bool:
        """
        Initialize tracking on a target object.
        
        Args:
            frame: Initial frame
            target_bbox: Bounding box of target (x1, y1, x2, y2)
            target_class: Class name of target object
            
        Returns:
            True if initialization successful, False otherwise
        """
        # Reset tracker state
        self.model.predictor = None  # Reset predictor to clear tracking state
        self.target_track_id = None
        self.target_class_id = None
        
        # Perform initial detection/tracking
        results = self.track(frame, persist=True)
        
        if not results:
            logger.warning("No objects detected for initialization")
            self.state = TrackerState.LOST
            return False
        
        # If target_bbox is provided, find the best matching detection
        if target_bbox is not None:
            best_match = self._find_best_match(results, target_bbox)
            if best_match is not None:
                self.target_track_id = best_match.track_id
                self.target_class_id = best_match.class_id
                self.state = TrackerState.INITIALIZED
                logger.info(f"Tracking initialized on object {self.target_track_id}")
                return True
        
        # If target_class is provided, find first object of that class
        elif target_class is not None:
            for result in results:
                if result.class_name.lower() == target_class.lower():
                    self.target_track_id = result.track_id
                    self.target_class_id = result.class_id
                    self.state = TrackerState.INITIALIZED
                    logger.info(f"Tracking initialized on {target_class}")
                    return True
        
        # If no specific target, track the first detected object
        else:
            self.target_track_id = results[0].track_id
            self.target_class_id = results[0].class_id
            self.state = TrackerState.INITIALIZED
            logger.info(f"Tracking initialized on first object: {results[0].class_name}")
            return True
        
        self.state = TrackerState.LOST
        return False
    
    def _find_best_match(
        self,
        results: List[TrackingResult],
        target_bbox: Tuple[int, int, int, int]
    ) -> Optional[TrackingResult]:
        """
        Find the detection that best matches the target bounding box.
        
        Args:
            results: List of tracking results
            target_bbox: Target bounding box (x1, y1, x2, y2)
            
        Returns:
            Best matching TrackingResult or None
        """
        best_iou = 0.0
        best_match = None
        
        for result in results:
            iou = self._calculate_iou(target_bbox, result.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = result
        
        # Return match only if IoU is above threshold
        if best_iou > 0.3:
            return best_match
        return None
    
    @staticmethod
    def _calculate_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: First bounding box (x1, y1, x2, y2)
            box2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, frame: np.ndarray) -> Optional[TrackingResult]:
        """
        Update tracking on a new frame.
        
        Args:
            frame: New RGB frame
            
        Returns:
            TrackingResult for the target object, or None if lost
        """
        if self.state == TrackerState.UNINITIALIZED:
            logger.warning("Tracker not initialized. Call initialize_tracking first.")
            return None
        
        # Perform tracking
        results = self.track(frame, persist=True)
        
        # Find our target in the results
        for result in results:
            if result.track_id == self.target_track_id:
                self.state = TrackerState.TRACKING
                return result
        
        # Target not found
        self.state = TrackerState.LOST
        logger.warning(f"Target {self.target_track_id} lost")
        return None
    
    def get_target_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Get the segmentation mask for the tracked target.
        
        Args:
            frame: Current frame
            
        Returns:
            Binary mask of the target object or None
        """
        result = self.update(frame)
        if result is not None and result.mask is not None:
            return result.mask
        return None
    
    def get_all_tracks(self) -> List[TrackingResult]:
        """
        Get all current tracking results.
        
        Returns:
            List of all TrackingResult objects from last update
        """
        return self.tracked_results
    
    def reset(self):
        """Reset the tracker state."""
        self.model.predictor = None
        self.target_track_id = None
        self.target_class_id = None
        self.tracked_results = []
        self.state = TrackerState.UNINITIALIZED
        logger.info("Tracker reset")
    
    def get_class_names(self) -> Dict[int, str]:
        """
        Get the mapping of class IDs to names.
        
        Returns:
            Dictionary mapping class ID to class name
        """
        return self.model.names


class TrackingVisualizer:
    """
    Utility class for visualizing tracking results.
    
    Provides methods to draw bounding boxes, masks, and tracking info
    on frames.
    """
    
    # Color palette for different track IDs
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
    ]
    
    @classmethod
    def get_color(cls, track_id: int) -> Tuple[int, int, int]:
        """Get color for a track ID."""
        return cls.COLORS[track_id % len(cls.COLORS)]
    
    @classmethod
    def draw_tracking_result(
        cls,
        frame: np.ndarray,
        result: TrackingResult,
        draw_mask: bool = True,
        draw_bbox: bool = True,
        draw_label: bool = True,
        mask_alpha: float = 0.4,
        highlight: bool = False
    ) -> np.ndarray:
        """
        Draw a single tracking result on the frame.
        
        Args:
            frame: Input frame (will be modified in place)
            result: TrackingResult to draw
            draw_mask: Whether to draw segmentation mask
            draw_bbox: Whether to draw bounding box
            draw_label: Whether to draw label
            mask_alpha: Transparency of mask overlay
            highlight: Whether to highlight this result
            
        Returns:
            Frame with drawings
        """
        color = cls.get_color(result.track_id if result.track_id >= 0 else 0)
        
        if highlight:
            color = (0, 255, 0)  # Green for highlighted
        
        # Draw segmentation mask
        if draw_mask and result.mask is not None:
            mask_colored = np.zeros_like(frame)
            mask_colored[result.mask > 0] = color
            frame = cv2.addWeighted(frame, 1, mask_colored, mask_alpha, 0)
        
        # Draw bounding box
        if draw_bbox:
            x1, y1, x2, y2 = result.bbox
            thickness = 3 if highlight else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if draw_label:
            x1, y1, _, _ = result.bbox
            label = f"ID:{result.track_id} {result.class_name} {result.confidence:.2f}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return frame
    
    @classmethod
    def draw_all_results(
        cls,
        frame: np.ndarray,
        results: List[TrackingResult],
        target_id: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Draw all tracking results on the frame.
        
        Args:
            frame: Input frame
            results: List of TrackingResult objects
            target_id: ID of target to highlight
            **kwargs: Additional arguments for draw_tracking_result
            
        Returns:
            Frame with all drawings
        """
        output = frame.copy()
        
        for result in results:
            highlight = (target_id is not None and result.track_id == target_id)
            output = cls.draw_tracking_result(output, result, highlight=highlight, **kwargs)
        
        return output
    
    @classmethod
    def draw_info_panel(
        cls,
        frame: np.ndarray,
        tracker_state: TrackerState,
        target_id: Optional[int],
        fps: float
    ) -> np.ndarray:
        """
        Draw information panel on the frame.
        
        Args:
            frame: Input frame
            tracker_state: Current tracker state
            target_id: Current target ID
            fps: Current FPS
            
        Returns:
            Frame with info panel
        """
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"State: {tracker_state.value}", (20, 35),
                    font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Target ID: {target_id if target_id else 'None'}", (20, 60),
                    font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 85),
                    font, 0.6, (255, 255, 255), 2)
        
        return frame
    
    @classmethod
    def draw_instructions(cls, frame: np.ndarray) -> np.ndarray:
        """
        Draw control instructions on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with instructions
        """
        h, w = frame.shape[:2]
        
        instructions = [
            "Controls:",
            "SPACE - Initialize tracking on first object",
            "R - Reset tracker",
            "M - Toggle mask display",
            "Q/ESC - Quit"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 130), (400, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = h - 110
        for instruction in instructions:
            cv2.putText(frame, instruction, (20, y_offset),
                        font, 0.5, (255, 255, 255), 1)
            y_offset += 22
        
        return frame


def main():
    """
    Main function demonstrating YOLO tracking with laptop camera.
    
    This function:
    1. Initializes the YOLO tracker
    2. Opens the laptop camera
    3. Provides interactive controls for tracking
    4. Visualizes tracking results with segmentation masks
    """
    import time
    
    print("=" * 60)
    print("YOLO Object Tracking with Segmentation")
    print("=" * 60)
    
    # Initialize tracker
    print("\nInitializing tracker...")
    try:
        tracker = YOLOTracker(
            model_path="yolo11n-seg.pt",  # Use nano model for speed
            confidence_threshold=0.5,
            warmup=True
        )
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        print("Make sure ultralytics is installed: pip install ultralytics")
        return
    
    # Open camera
    print("\nOpening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        print("Make sure a camera is connected and accessible")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Visualization settings
    show_mask = True
    show_all_objects = True
    
    # FPS calculation
    fps = 0.0
    frame_count = 0
    start_time = time.time()
    
    print("\nStarting tracking loop...")
    print("Press SPACE to initialize tracking, R to reset, M to toggle masks, Q to quit")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Track objects
            if tracker.state == TrackerState.UNINITIALIZED:
                # Just detect without persistent tracking
                results = tracker.detect(frame_rgb)
            else:
                # Perform tracking update
                target_result = tracker.update(frame_rgb)
                results = tracker.get_all_tracks()
            
            # Convert back to BGR for display
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw results
            if show_all_objects:
                display_frame = TrackingVisualizer.draw_all_results(
                    display_frame,
                    results,
                    target_id=tracker.target_track_id,
                    draw_mask=show_mask
                )
            elif tracker.state == TrackerState.TRACKING:
                for result in results:
                    if result.track_id == tracker.target_track_id:
                        display_frame = TrackingVisualizer.draw_tracking_result(
                            display_frame,
                            result,
                            draw_mask=show_mask,
                            highlight=True
                        )
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Draw info panel and instructions
            display_frame = TrackingVisualizer.draw_info_panel(
                display_frame,
                tracker.state,
                tracker.target_track_id,
                fps
            )
            display_frame = TrackingVisualizer.draw_instructions(display_frame)
            
            # Show frame
            cv2.imshow("YOLO Tracking", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nQuitting...")
                break
                
            elif key == ord(' '):  # SPACE - Initialize tracking
                print("\nInitializing tracking...")
                success = tracker.initialize_tracking(frame_rgb)
                if success:
                    print(f"Tracking initialized on object ID: {tracker.target_track_id}")
                else:
                    print("Failed to initialize tracking - no objects detected")
                    
            elif key == ord('r'):  # R - Reset
                print("\nResetting tracker...")
                tracker.reset()
                
            elif key == ord('m'):  # M - Toggle mask
                show_mask = not show_mask
                print(f"Mask display: {'ON' if show_mask else 'OFF'}")
                
            elif key == ord('a'):  # A - Toggle all objects
                show_all_objects = not show_all_objects
                print(f"Show all objects: {'ON' if show_all_objects else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
