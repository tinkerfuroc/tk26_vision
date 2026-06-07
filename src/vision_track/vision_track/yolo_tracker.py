#!/usr/bin/env python3
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .core.result_parser import parse_yolo_results
from .core.tracking_pipeline import (
    detect_occlusion,
    periodic_reid_validation,
    register_other_persons,
    save_pre_occlusion_state,
    update_scene_motion,
    update_tracker,
    verify_post_occlusion,
)
from .core.tracking_types import TargetAppearance, TrackerState, TrackingResult
from .core.registry import PersonRegistry
from .core.operator_init import select_operator_detection
from .reid.appearance_manager import update_appearance
from .reid.embedding_cache import FrameEmbeddingCache
from .reid.reid import AppearanceExtractor, ReIDMatcher
from .reid.reid_search import find_best_match_reid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOTracker:
    """
    YOLO-based object tracker with segmentation and re-identification support.
    
    This class provides functionality for:
    - Object detection and segmentation using YOLO
    - Object tracking across frames using ByteTrack
    - Appearance-based re-identification for robust tracking
    - GPU acceleration when available
    
    Attributes:
        model: The YOLO model instance
        device: The device (CPU/GPU) used for inference
        state: Current state of the tracker
        target_track_id: ID of the object being tracked
    """
    
    # Default YOLO model for segmentation
    DEFAULT_MODEL = "yolo11s-seg.pt"
    
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
        warmup: bool = True,
        enable_reid: bool = True,
        inference_size: Optional[int] = None,
        reid_verification_interval: int = 5,
        reid_backbone: str = "osnet_ain_x1_0",
        reid_weights_path: str = "",
        reid_fp16: bool = True,
        reid_gallery_enabled: bool = True,
        reid_gallery_size: int = 6,
        reid_gallery_novelty_max: float = 0.85,
        reid_gallery_score_mode: str = "max",
        yolo_track_conf: float = 0.15,
    ):
        """
        Initialize the YOLO tracker.
        
        Args:
            model_path: Path to the YOLO model or model name
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to use ('cuda', 'cpu', or None for auto)
            warmup: Whether to warm up the model on initialization
            enable_reid: Whether to enable re-identification features
            inference_size: Optional inference size (imgsz). Lower for speed, higher for accuracy.
            reid_verification_interval: Run a full-frame ReID sanity check every N frames while tracking
            reid_backbone: OSNet variant for the deep ReID term ('osnet_ain_x1_0' default)
            reid_weights_path: optional ReID-trained checkpoint overriding the imagenet init
            reid_fp16: run the ReID deep forward in half precision (CUDA only;
                no-op on CPU). Default True for throughput in multi-person scenes.
            reid_gallery_enabled: enable the multi-view reacquisition gallery on
                the target appearance (kill-switch; False restores legacy behavior)
            reid_gallery_size: K diverse views kept in the gallery
            reid_gallery_novelty_max: admit a view only if its cosine to existing
                views is below this threshold
            reid_gallery_score_mode: gallery scoring mode ('max' | 'top2_mean')
            yolo_track_conf: LOW detection conf fed to model.track so ByteTrack's
                two-stage (high/low) association recovery actually runs — kept
                separate from confidence_threshold, which still gates detect()
                calls and downstream consumers
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.inference_size = inference_size
        self.reid_backbone = reid_backbone
        self.reid_weights_path = reid_weights_path
        self.reid_fp16 = reid_fp16
        self.reid_gallery_enabled = reid_gallery_enabled
        self.reid_gallery_size = reid_gallery_size
        self.reid_gallery_novelty_max = reid_gallery_novelty_max
        self.reid_gallery_score_mode = reid_gallery_score_mode
        self.yolo_track_conf = yolo_track_conf
        self.state = TrackerState.UNINITIALIZED
        self.target_track_id: Optional[int] = None
        self.original_track_id: Optional[int] = None
        self.target_class_id: Optional[int] = None
        self.target_class_name: Optional[str] = None
        self.tracked_results: List[TrackingResult] = []

        self._init_reid_settings(enable_reid, reid_verification_interval)
        self._init_motion_tracking()
        self._init_occlusion_state()
        self.person_registry = PersonRegistry()
        self._init_temporal_consistency()

        self.last_results: List[TrackingResult] = []

        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")

        # Resolve the project ByteTrack config (installed share/ path) so the
        # high/low association thresholds + buffer are ours, not stock.
        self.tracker_cfg = self._resolve_tracker_cfg()
        logger.info(f"Using tracker config: {self.tracker_cfg}")

        # Load YOLO model
        self.model = self._load_model(model_path)
        
        # Initialize appearance extractor for re-identification
        if self.enable_reid:
            self.appearance_extractor = AppearanceExtractor(
                self.device,
                reid_backbone=self.reid_backbone,
                reid_weights_path=self.reid_weights_path,
                reid_fp16=self.reid_fp16,
            )
            logger.info("Re-identification enabled")
        else:
            self.appearance_extractor = None
        
        # Warm up the model if requested
        if warmup:
            self._warmup_model()

    def _configure_gallery(self, appearance) -> None:
        """Apply this tracker's gallery ROS params to a TargetAppearance."""
        appearance.configure_gallery(
            enabled=self.reid_gallery_enabled,
            size=self.reid_gallery_size,
            novelty_max=self.reid_gallery_novelty_max,
            score_mode=self.reid_gallery_score_mode,
        )

    def _init_reid_settings(self, enable_reid: bool, reid_verification_interval: int) -> None:
        """Initialize ReID and tracking thresholds."""
        self.enable_reid = enable_reid
        self.target_appearance: Optional[TargetAppearance] = None
        self.reid_threshold = ReIDMatcher.REID_THRESHOLD
        self.frames_lost = 0
        self.max_frames_lost = 600
        self.frame_rate: float = 30.0
        self.frame_count = 0
        self.reid_extraction_interval = 3
        self.fast_tracking_mode = False
        self.reid_verification_interval = max(0, reid_verification_interval)
        self.feature_refresh_interval = 1.5
        # Phase 3: per-frame embedding cache so the four ReID embed call sites
        # within one update() reuse the score-pass feature dict instead of
        # re-embedding the same crop up to 4x/frame. Keyed by (track_id,
        # frame_count); a new frame_count drops the previous frame's entries.
        self.embedding_cache = FrameEmbeddingCache(max_entries=32)

    def _init_motion_tracking(self) -> None:
        """Initialize spatial continuity and motion tracking state."""
        self.last_known_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_known_center: Optional[Tuple[float, float]] = None
        self.scene_center_history: List[Tuple[float, float]] = []
        self.camera_motion_detected: bool = False
        self.last_camera_motion_time: float = 0.0
        self.CAMERA_MOTION_THRESHOLD = 50.0
        self.CAMERA_MOTION_COOLDOWN = 0.5
        self.camera_motion_vector: Tuple[float, float] = (0.0, 0.0)
        self.camera_motion_recent_window = 1.0
        self.spatial_gate_base = 160.0
        self.camera_motion_gate_scale = 0.8
        self.target_velocity: Tuple[float, float] = (0.0, 0.0)
        self.target_velocity_history: List[Tuple[float, float]] = []
        self.last_position_time: float = 0.0
        self.relative_positions: Dict[int, Tuple[float, float]] = {}
        self.candidate_consistency: Dict[int, List[float]] = {}
        self.CONSISTENCY_WINDOW = 5
        self.CONSISTENCY_THRESHOLD = 0.15
        # Phase 2: operator's last known median depth (m), plumbed from the node
        # (only the node owns the depth image). None ⇒ depth gate permissive.
        self.operator_last_depth_m: Optional[float] = None
        self.crosser_depth_jump_m = 0.6
        # Per-frame map: track_id -> candidate median depth (m), set by the node
        # before each tracker.update so the pipeline can gate ReID candidates.
        self.candidate_depths_m: Dict[int, float] = {}

    def _init_occlusion_state(self) -> None:
        """Set up occlusion tracking defaults."""
        self.occlusion_iou_threshold = 0.3
        self.is_occluded = False
        self.occlusion_start_time: Optional[float] = None
        self.pre_occlusion_appearance: Optional[TargetAppearance] = None
        self.frames_since_occlusion_ended = 0
        self.occlusion_recovery_frames = 45

    def _init_temporal_consistency(self) -> None:
        """Initialize temporal consistency tracking variables."""
        self.last_reid_switch_time = 0.0
        self.reid_switch_cooldown = 1.0
        self.consecutive_reid_frames = 0
        self.reid_confirmation_frames = 12
        self.reid_preconfirm_frames = 3
        self.pending_reid_match: Optional[Tuple[int, float]] = None
        self.reid_fit_streak = 0
        self.reid_fit_id: Optional[int] = None
        # Phase 2: asymmetric-hysteresis recovery policy params (defaults;
        # the node overrides from ROS params). Pure FSM lives in core/.
        self.max_recovery_frames = 45
        self.provisional_high_bar = 0.72
        self.provisional_distinct_margin = 0.10
        self.lock_state_machine = None  # set by node after construction
        self.last_lock_decision = None  # latest FSM verdict for the node to read
        self.last_reid_margin = 0.0  # best-vs-second margin from reid_search
        # True when the recovery path (reidentify_target) authoritatively
        # stepped the FSM this frame. The node defers to last_lock_decision
        # then instead of re-stepping present=True (which would defeat the
        # asymmetric hysteresis on a partial-confirm recovery frame).
        self.last_frame_recovery = False

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
            if str(model_path).endswith(".engine"):
                # Optional best-effort TensorRT top-end (Phase 3 Task 4).
                # Ultralytics loads .engine transparently; the engine is
                # resolution/batch-locked, so the runtime imgsz MUST match the
                # export imgsz or detections will be silently wrong.
                logger.warning(
                    "Loading a TensorRT engine — runtime imgsz MUST match the "
                    "export imgsz (resolution/batch-locked)."
                )
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
            
        # Warmup appearance extractor if enabled
        if self.enable_reid and self.appearance_extractor is not None:
            dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = self.appearance_extractor.extract_features(
                dummy_input, (0, 0, 100, 100), None
            )
            
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
        infer_kwargs = dict(
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=classes,
            verbose=False,
        )
        if self.inference_size is not None:
            infer_kwargs["imgsz"] = self.inference_size
        
        results = self.model(
            frame,
            **infer_kwargs
        )
        
        return self._parse_results(results[0])

    def _resolve_tracker_cfg(self) -> str:
        try:
            from ament_index_python.packages import get_package_share_directory
            cfg = os.path.join(get_package_share_directory("vision_track"), "config", "bytetrack.yaml")
            if os.path.exists(cfg):
                return cfg
        except Exception:
            pass
        return "bytetrack.yaml"

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
        track_kwargs = dict(
            conf=self.yolo_track_conf,
            iou=self.iou_threshold,
            classes=classes,
            persist=persist,
            tracker=self.tracker_cfg,
            half=True,
            verbose=False,
        )
        if self.inference_size is not None:
            track_kwargs["imgsz"] = self.inference_size
        
        results = self.model.track(
            frame,
            **track_kwargs
        )
        
        self.tracked_results = self._parse_results(results[0])
        return self.tracked_results
    
    def _parse_results(self, result) -> List[TrackingResult]:
        """Parse YOLO results into TrackingResult objects."""
        return parse_yolo_results(result)
    
    def initialize_tracking(
        self,
        frame: np.ndarray,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        target_class: Optional[str] = None,
        target_id: Optional[int] = None
    ) -> bool:
        """
        Initialize tracking on a target object.
        
        Args:
            frame: Initial frame
            target_bbox: Bounding box of target (x1, y1, x2, y2)
            target_class: Class name of target object (e.g., 'person', 'car')
            target_id: Specific track ID to follow (if already known)
            
        Returns:
            True if initialization successful, False otherwise
        """
        # Reset tracker state
        self.model.predictor = None  # Reset predictor to clear tracking state
        self.target_track_id = None
        self.original_track_id = None  # Reset original ID
        self.target_class_id = None
        self.target_class_name = target_class  # Store target class for filtering
        self.target_appearance = None
        self.frames_lost = 0
        # Reset temporal consistency tracking
        self.last_reid_switch_time = 0.0
        self.consecutive_reid_frames = 0
        self.pending_reid_match = None
        
        # Perform initial detection/tracking
        results = self.track(frame, persist=True)
        
        if not results:
            logger.warning("No objects detected for initialization")
            self.state = TrackerState.LOST
            return False
        
        selected_result = None
        
        # If target_id is provided, find that specific track
        if target_id is not None:
            for result in results:
                if result.track_id == target_id:
                    selected_result = result
                    break
            if selected_result is None:
                logger.warning(f"Target ID {target_id} not found in detections")
        
        # If target_bbox is provided, find the best matching detection
        elif target_bbox is not None:
            # Filter by class first if specified
            candidates = results
            if target_class is not None:
                candidates = [r for r in results if r.class_name.lower() == target_class.lower()]
                if not candidates:
                    candidates = results  # Fallback to all if no class match
            
            best_match = self._find_best_match_iou(candidates, target_bbox)
            if best_match is not None:
                selected_result = best_match
        
        # If target_class is provided, pick the best operator candidate
        # (nearest + most central, conf tie-break) instead of results[0].
        elif target_class is not None:
            img_h, img_w = frame.shape[:2]
            selected_result = select_operator_detection(
                results,
                image_wh=(img_w, img_h),
                # No depth image at init time in this path; centeredness +
                # confidence drive the choice. The node's depth-aware init is a
                # Phase 2 concern — here depth is unavailable.
                depth_lookup=lambda _bbox: None,
                target_class=target_class,
            )

        # If no specific target, track the best central candidate of any class.
        else:
            img_h, img_w = frame.shape[:2]
            selected_result = select_operator_detection(
                results,
                image_wh=(img_w, img_h),
                depth_lookup=lambda _bbox: None,
                target_class=results[0].class_name,
            ) or results[0]
        
        if selected_result is not None:
            self.target_track_id = selected_result.track_id
            self.original_track_id = selected_result.track_id  # Store original ID for consistent display
            self.target_class_id = selected_result.class_id
            self.target_class_name = selected_result.class_name
            self.state = TrackerState.INITIALIZED
            
            # Initialize appearance model for re-identification
            if self.enable_reid:
                self._update_appearance(frame, selected_result)
                # Register this person in the registry
                if self.target_appearance is not None and self.original_track_id is not None:
                    self.person_registry.register_person(self.original_track_id, self.target_appearance)
            
            logger.info(f"Tracking initialized on {self.target_class_name} (ID: {self.original_track_id})")
            return True
        
        self.state = TrackerState.LOST
        return False

    def _apply_reseed(self, selected_result, fresh_reid_feature) -> int:
        """Re-lock onto an externally-confirmed detection, preserving identity.

        Unlike initialize_tracking (which resets appearance), this keeps the
        multi-view gallery + person registry (same operator, self-identified),
        appends the fresh confirmed view, re-locks the ids, clears the lost
        counter, and re-arms the lock FSM. Returns the locked track id, or -1
        if selected_result is None.
        """
        if selected_result is None:
            return -1
        self.target_track_id = selected_result.track_id
        self.original_track_id = selected_result.track_id
        self.target_class_id = selected_result.class_id
        self.target_class_name = selected_result.class_name
        self.frames_lost = 0
        self.state = TrackerState.TRACKING
        # Re-lock onto a confirmed view: clear stale occlusion bookkeeping so a
        # mid-occlusion reseed doesn't carry a pre-occlusion appearance snapshot.
        self.is_occluded = False
        self.pre_occlusion_appearance = None
        if self.target_appearance is not None and fresh_reid_feature is not None:
            self.target_appearance.gallery.maybe_add(fresh_reid_feature)
        if self.lock_state_machine is not None and self.original_track_id is not None:
            self.lock_state_machine.start(self.original_track_id)
        return self.target_track_id

    def reseed_target(self, frame, bbox, target_class: str = "person") -> int:
        """Detect on `frame`, match `bbox`, and re-lock preserving the gallery.

        Returns the locked track id, or -1 if no detection matches the bbox.
        """
        results = self.track(frame, persist=True)
        if not results:
            return -1
        candidates = [r for r in results
                      if target_class is None or r.class_name.lower() == target_class.lower()]
        if not candidates:
            candidates = results
        best = self._find_best_match_iou(candidates, bbox)
        if best is None:
            return -1
        fresh = None
        if self.appearance_extractor is not None:
            feats = self.appearance_extractor.extract_features_batch(
                frame, [best.bbox], [best.mask], [best.class_id])
            if feats and feats[0] and "reid" in feats[0]:
                fresh = feats[0]["reid"]
        # Gallery-additive only: append the fresh view to the multi-view deep
        # gallery (via _apply_reseed) but deliberately do NOT overwrite the
        # color/identity anchors. The reseed match is geometric (IoU) only, so
        # promoting the crop to the anchor would let a wrong-overlap box poison
        # identity -- precision is sacred. The deep gallery's max-over-views
        # still helps recognise the new appearance under drift.
        return self._apply_reseed(best, fresh)

    def _update_appearance(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        similarity: Optional[float] = None
    ):
        """Delegate to the shared appearance updater."""
        update_appearance(self, frame, result, similarity)
    
    def _find_best_match_iou(
        self,
        results: List[TrackingResult],
        target_bbox: Tuple[int, int, int, int]
    ) -> Optional[TrackingResult]:
        """
        Find the detection that best matches the target bounding box using IoU.
        
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
    
    def _update_scene_motion(self, results: List[TrackingResult], frame: Optional[np.ndarray] = None):
        """Delegate motion analysis to the shared pipeline helper."""
        return update_scene_motion(self, results, frame)
    
    def _update_relative_positions(self, target_result: TrackingResult, results: List[TrackingResult]):
        """
        Track positions of other people relative to the target.
        
        These relative positions are invariant to camera motion, so they can help
        identify if the target moved or if the camera just panned.
        
        Args:
            target_result: The target person's detection
            results: All detection results
        """
        target_cx = (target_result.bbox[0] + target_result.bbox[2]) / 2.0
        target_cy = (target_result.bbox[1] + target_result.bbox[3]) / 2.0
        
        for r in results:
            if r.track_id == target_result.track_id or r.class_id != 0:
                continue
            
            cx = (r.bbox[0] + r.bbox[2]) / 2.0
            cy = (r.bbox[1] + r.bbox[3]) / 2.0
            
            # Store relative position (dx, dy) from target
            self.relative_positions[r.track_id] = (cx - target_cx, cy - target_cy)
    
    def _check_relative_position_consistency(
        self, 
        candidate: TrackingResult, 
        results: List[TrackingResult]
    ) -> Tuple[bool, float]:
        """
        Check if relative positions to other people are consistent.
        
        If the candidate is at the same relative position to other people as
        the target was before, it's more likely to be the correct target.
        
        Args:
            candidate: Candidate detection to check
            results: All detections
            
        Returns:
            Tuple of (is_consistent, consistency_score)
        """
        if not self.relative_positions:
            return True, 1.0  # No prior data, can't check
        
        cand_cx = (candidate.bbox[0] + candidate.bbox[2]) / 2.0
        cand_cy = (candidate.bbox[1] + candidate.bbox[3]) / 2.0
        
        position_errors = []
        
        for r in results:
            if r.track_id == candidate.track_id or r.class_id != 0:
                continue
            
            if r.track_id in self.relative_positions:
                expected_dx, expected_dy = self.relative_positions[r.track_id]
                
                # Current relative position
                cx = (r.bbox[0] + r.bbox[2]) / 2.0
                cy = (r.bbox[1] + r.bbox[3]) / 2.0
                actual_dx = cx - cand_cx
                actual_dy = cy - cand_cy
                
                # Error in relative position
                error = ((actual_dx - expected_dx) ** 2 + (actual_dy - expected_dy) ** 2) ** 0.5
                position_errors.append(error)
        
        if not position_errors:
            return True, 1.0  # No common references
        
        # Average error - should be small if this is the right person
        avg_error = sum(position_errors) / len(position_errors)
        
        # Consider consistent if average error < 100 pixels
        RELATIVE_POSITION_THRESHOLD = 100.0
        is_consistent = avg_error < RELATIVE_POSITION_THRESHOLD
        
        # Convert to score (1.0 = perfect, 0.0 = 200px error)
        consistency_score = max(0.0, 1.0 - avg_error / 200.0)
        
        if not is_consistent:
            logger.info(f"Relative position check: candidate ID {candidate.track_id} "
                       f"has avg error {avg_error:.1f}px (threshold={RELATIVE_POSITION_THRESHOLD})")
        
        return is_consistent, consistency_score

    def _passes_spatial_gate(
        self,
        candidate_bbox: Tuple[int, int, int, int],
        use_camera_gate: bool = False
    ) -> Tuple[bool, float, float]:
        """
        Check whether a candidate is spatially plausible relative to last known target position.
        
        During or shortly after camera motion we tighten the gate to avoid snapping
        to nearby people that move in lockstep with the camera pan.
        """
        if self.last_known_center is None:
            return True, 0.0, float("inf")
        
        # Prefer velocity + camera motion compensated prediction when available
        predicted = self._predict_target_position()
        if predicted is None:
            predicted = self.last_known_center
        
        cand_cx = (candidate_bbox[0] + candidate_bbox[2]) * 0.5
        cand_cy = (candidate_bbox[1] + candidate_bbox[3]) * 0.5
        
        dx = cand_cx - predicted[0]
        dy = cand_cy - predicted[1]
        dist = (dx * dx + dy * dy) ** 0.5
        
        gate = self.spatial_gate_base
        if self.last_known_bbox is not None:
            w = self.last_known_bbox[2] - self.last_known_bbox[0]
            h = self.last_known_bbox[3] - self.last_known_bbox[1]
            gate = max(gate, max(w, h) * 1.5)
        
        if use_camera_gate:
            gate = max(80.0, gate * self.camera_motion_gate_scale)
        
        return dist <= gate, dist, gate
    
    def _update_candidate_consistency(self, track_id: int, similarity: float):
        """
        Track appearance similarity consistency for a candidate over time.
        
        Legitimate targets should have consistent similarity scores.
        Wrong candidates often show erratic scores.
        
        Args:
            track_id: Track ID of the candidate
            similarity: Current similarity score
        """
        if track_id not in self.candidate_consistency:
            self.candidate_consistency[track_id] = []
        
        self.candidate_consistency[track_id].append(similarity)
        
        # Keep only recent history
        if len(self.candidate_consistency[track_id]) > self.CONSISTENCY_WINDOW:
            self.candidate_consistency[track_id].pop(0)
    
    def _get_candidate_consistency_score(self, track_id: int) -> float:
        """
        Get consistency score for a candidate based on historical similarity variance.
        
        Args:
            track_id: Track ID to check
            
        Returns:
            Consistency score (1.0 = perfectly consistent, 0.0 = very inconsistent)
        """
        if track_id not in self.candidate_consistency:
            return 0.5  # No history, neutral score
        
        history = self.candidate_consistency[track_id]
        if len(history) < 2:
            return 0.5  # Not enough history
        
        # Compute variance
        variance = np.var(history)
        
        # Convert variance to score (lower variance = higher score)
        # variance of 0.01 = score ~0.9, variance of 0.04 = score ~0.6
        score = max(0.0, 1.0 - variance / self.CONSISTENCY_THRESHOLD)
        
        return score

    def _periodic_reid_validation(
        self,
        frame: np.ndarray,
        results: List[TrackingResult],
        current_result: TrackingResult,
        current_similarity: Optional[float] = None
    ) -> Tuple[bool, Optional[TrackingResult]]:
        """Delegate periodic validation to shared pipeline helper."""
        return periodic_reid_validation(self, frame, results, current_result, current_similarity)
    
    def _predict_target_position(self, dt: float = 0.033) -> Optional[Tuple[float, float]]:
        """
        Predict target position based on velocity and camera motion compensation.
        
        Args:
            dt: Time delta since last frame (default ~30fps)
            
        Returns:
            Predicted (cx, cy) or None if can't predict
        """
        if self.last_known_center is None:
            return None
        
        # Start with last known position
        pred_x, pred_y = self.last_known_center
        
        # Add velocity-based prediction
        vx, vy = self.target_velocity
        pred_x += vx * dt
        pred_y += vy * dt
        
        # Compensate for camera motion (camera moved, so target appears to move opposite)
        # This is critical during camera shake
        cam_dx, cam_dy = self.camera_motion_vector
        pred_x += cam_dx  # Add camera motion to expected position
        pred_y += cam_dy
        
        return (pred_x, pred_y)
    
    def _update_target_velocity(self, current_center: Tuple[float, float], dt: Optional[float] = None):
        """Update target velocity estimate with smoothing.

        Args:
            current_center: Current target center (cx, cy).
            dt: Scene-time delta (s) between this and the previous center,
                derived from frame stamps. When None, falls back to wall-clock
                (legacy behavior) so non-stamped callers keep working.
        """
        if dt is None:
            current_time = time.time()
            dt = (current_time - self.last_position_time) if self.last_position_time > 0 else 0.0
            self.last_position_time = current_time

        if self.last_known_center is not None and dt > 0.001:
            vx = (current_center[0] - self.last_known_center[0]) / dt
            vy = (current_center[1] - self.last_known_center[1]) / dt
            alpha = 0.3
            old_vx, old_vy = self.target_velocity
            self.target_velocity = (
                alpha * vx + (1 - alpha) * old_vx,
                alpha * vy + (1 - alpha) * old_vy,
            )
    
    def _register_other_persons(self, frame: np.ndarray, results: List[TrackingResult]):
        """Register other persons via the shared pipeline helper."""
        return register_other_persons(self, frame, results)
    
    def _find_best_match_reid(
        self,
        frame: np.ndarray,
        results: List[TrackingResult]
    ) -> Optional[Tuple[TrackingResult, float]]:
        """Delegate re-identification scoring to shared helper."""
        return find_best_match_reid(self, frame, results)
    
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
    
    def _detect_occlusion(
        self,
        target_result: TrackingResult,
        all_results: List[TrackingResult]
    ) -> Tuple[bool, Optional[TrackingResult]]:
        """Detect potential occlusion using the shared helper."""
        return detect_occlusion(self, target_result, all_results)
    
    def _save_pre_occlusion_state(self):
        """Save pre-occlusion state via pipeline helper."""
        return save_pre_occlusion_state(self)
    
    def _verify_post_occlusion(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        current_time: float
    ) -> bool:
        """Verify post-occlusion identity via pipeline helper."""
        return verify_post_occlusion(self, frame, result, current_time)
    
    def update(
        self,
        frame: np.ndarray,
        target_id: Optional[int] = None
    ) -> Optional[TrackingResult]:
        """Run the tracking pipeline."""
        return update_tracker(self, frame, target_id)
    
    def _with_original_id(self, result: TrackingResult) -> TrackingResult:
        """
        Return a copy of the result with the original (stable) track ID.
        
        This ensures users always see the same track ID for the target,
        even if YOLO assigns a new ID after re-identification.
        Also updates last known position for spatial continuity.
        """
        # Update last known position for spatial continuity in ReID
        self.last_known_bbox = result.bbox
        x1, y1, x2, y2 = result.bbox
        self.last_known_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        
        return TrackingResult(
            track_id=self.original_track_id,
            bbox=result.bbox,
            mask=result.mask,
            confidence=result.confidence,
            class_id=result.class_id,
            class_name=result.class_name
        )
    
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
        self.original_track_id = None
        self.target_class_id = None
        self.target_class_name = None
        self.target_appearance = None
        self.tracked_results = []
        self.frames_lost = 0
        self.state = TrackerState.UNINITIALIZED
        # Reset temporal consistency tracking
        self.last_reid_switch_time = 0.0
        self.consecutive_reid_frames = 0
        self.pending_reid_match = None
        self.reid_fit_streak = 0
        self.reid_fit_id = None
        # dead: original_track_id already None here (cleared above), so this
        # start() never fires on reset(). The FSM is (re)started on the next
        # initialize_tracking() commit instead; kept for symmetry/intent.
        if self.lock_state_machine is not None and self.original_track_id is not None:
            self.lock_state_machine.start(self.original_track_id)
        self.last_lock_decision = None
        self.last_reid_margin = 0.0
        self.last_frame_recovery = False
        # Clear the person registry
        self.person_registry.clear()
        # Reset camera motion detection
        self.scene_center_history.clear()
        self.camera_motion_detected = False
        self.last_camera_motion_time = 0.0
        self.camera_motion_vector = (0.0, 0.0)
        # Reset spatial tracking
        self.last_known_bbox = None
        self.last_known_center = None
        # Reset velocity and relative position tracking
        self.target_velocity = (0.0, 0.0)
        self.target_velocity_history.clear() if hasattr(self, 'target_velocity_history') and self.target_velocity_history else None
        self.last_position_time = 0.0
        self.relative_positions.clear()
        # Reset candidate consistency tracking
        self.candidate_consistency.clear()
        # Phase 2: clear depth-gate state (operator depth + per-candidate depths)
        self.operator_last_depth_m = None
        self.candidate_depths_m = {}
        # Reset frame counter
        self.frame_count = 0
        self.fast_tracking_mode = False
        # Phase 3: drop any stale per-frame embeddings.
        self.embedding_cache.clear()
        logger.info("Tracker reset")
    
    def get_class_names(self) -> Dict[int, str]:
        """
        Get the mapping of class IDs to names.
        
        Returns:
            Dictionary mapping class ID to class name
        """
        return self.model.names
    
    def set_reid_threshold(self, threshold: float):
        """
        Set the re-identification similarity threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        self.reid_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"ReID threshold set to {self.reid_threshold}")
    
    def get_tracker_info(self) -> Dict[str, Any]:
        """
        Get current tracker state information.
        
        Returns:
            Dictionary with tracker state info
        """
        return {
            "state": self.state.value,
            "target_track_id": self.original_track_id,  # Report stable ID to user
            "internal_yolo_id": self.target_track_id,   # Internal YOLO ID (may change)
            "target_class_id": self.target_class_id,
            "target_class_name": self.target_class_name,
            "frames_lost": self.frames_lost,
            "reid_enabled": self.enable_reid,
            "has_appearance_model": self.target_appearance is not None
        }
