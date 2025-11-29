#!/usr/bin/env python3
"""
YOLO-based Object Tracking with Segmentation and Re-Identification

This module provides a YOLOTracker class that uses YOLO models for object tracking
with segmentation capabilities. It supports GPU acceleration when available and
includes appearance-based re-identification for robust tracking through occlusions.

Author: TinkerFuroc
Date: 2025
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import time

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


@dataclass
class TargetAppearance:
    """
    Stores appearance features for re-identification.
    
    Maintains a history of feature embeddings and visual characteristics
    to enable robust re-identification after occlusion or off-screen events.
    """
    # Feature embedding history (for CNN-based matching)
    feature_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    
    # Color histogram history (HSV color distribution)
    color_hist_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    
    # Body part color history (for person ReID - upper/middle/lower body colors)
    body_color_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    
    # Size and aspect ratio history
    size_history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=30))
    
    # Last known position and velocity for motion prediction
    position_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    # Timestamp of last successful track
    last_seen_time: float = 0.0
    
    # Class information
    class_id: int = -1
    class_name: str = ""
    
    def get_average_feature(self) -> Optional[np.ndarray]:
        """Get averaged feature embedding from history."""
        if not self.feature_history:
            return None
        
        # Filter to only include features of the same dimension (handles mixed features)
        try:
            # Get the most common feature dimension
            dims = [f.shape[0] for f in self.feature_history]
            if not dims:
                return None
            
            # Use the dimension of the most recent feature as reference
            target_dim = dims[-1]
            
            # Only average features with matching dimensions
            matching_features = [f for f in self.feature_history if f.shape[0] == target_dim]
            
            if not matching_features:
                return None
            
            return np.mean(np.array(matching_features), axis=0)
        except Exception:
            # Fallback: return the most recent feature
            return self.feature_history[-1] if self.feature_history else None
    
    def get_average_color_hist(self) -> Optional[np.ndarray]:
        """Get averaged color histogram from history."""
        if not self.color_hist_history:
            return None
        try:
            return np.mean(np.array(list(self.color_hist_history)), axis=0)
        except Exception:
            return self.color_hist_history[-1] if self.color_hist_history else None
    
    def get_body_color(self) -> Optional[np.ndarray]:
        """Get averaged body part color histogram from history."""
        if not self.body_color_history:
            return None
        try:
            return np.mean(np.array(list(self.body_color_history)), axis=0)
        except Exception:
            return self.body_color_history[-1] if self.body_color_history else None
    
    def get_average_size(self) -> Optional[Tuple[float, float]]:
        """Get average size from history."""
        if not self.size_history:
            return None
        try:
            sizes = np.array(list(self.size_history))
            return (float(np.mean(sizes[:, 0])), float(np.mean(sizes[:, 1])))
        except Exception:
            return self.size_history[-1] if self.size_history else None
    
    def predict_position(self, dt: float = 1.0) -> Optional[Tuple[float, float]]:
        """Predict next position based on velocity."""
        if not self.position_history:
            return None
        last_pos = self.position_history[-1]
        return (
            last_pos[0] + self.velocity[0] * dt,
            last_pos[1] + self.velocity[1] * dt
        )


class PersonRegistry:
    """
    Registry of known persons with their distinctive features.
    
    This prevents wrong ID assignment by:
    1. Storing features of all tracked persons (not just the target)
    2. Checking that a candidate doesn't better match another known person
    3. Requiring candidates to be distinctively closer to the target than to others
    """
    
    def __init__(self):
        """Initialize the person registry."""
        # Map from display_id to TargetAppearance
        self.known_persons: Dict[int, TargetAppearance] = {}
        # Minimum distinctiveness - target must be this much more similar than any other known person
        self.distinctiveness_threshold = 0.15
    
    def register_person(self, display_id: int, appearance: TargetAppearance):
        """
        Register a person with their appearance features.
        
        Args:
            display_id: The stable display ID for this person
            appearance: Their appearance features
        """
        self.known_persons[display_id] = appearance
        logger.debug(f"Registered person ID {display_id} in registry (total: {len(self.known_persons)})")
    
    def update_person(self, display_id: int, appearance: TargetAppearance):
        """Update a person's appearance features."""
        self.known_persons[display_id] = appearance
    
    def get_person(self, display_id: int) -> Optional[TargetAppearance]:
        """Get a person's appearance by their display ID."""
        return self.known_persons.get(display_id)
    
    def remove_person(self, display_id: int):
        """Remove a person from the registry."""
        if display_id in self.known_persons:
            del self.known_persons[display_id]
            logger.debug(f"Removed person ID {display_id} from registry")
    
    def clear(self):
        """Clear all registered persons."""
        self.known_persons.clear()
        logger.debug("Cleared person registry")
    
    def check_distinctiveness(
        self,
        target_id: int,
        candidate_features: Dict[str, np.ndarray],
        target_similarity: float,
        similarity_func: callable
    ) -> bool:
        """
        Check if a candidate is distinctively the target and not another known person.
        
        Args:
            target_id: The ID of the person we're trying to match
            candidate_features: Features extracted from the candidate
            target_similarity: Similarity score to the target
            similarity_func: Function to compute similarity between appearance and features
            
        Returns:
            True if candidate is distinctively the target, False if it might be someone else
        """
        if len(self.known_persons) <= 1:
            # Only one person registered, no need to check distinctiveness
            return True
        
        # Check similarity to all other known persons
        for person_id, appearance in self.known_persons.items():
            if person_id == target_id:
                continue
            
            # Compute similarity to this other person
            other_similarity = similarity_func(appearance, candidate_features)
            
            # If candidate is too similar to another known person, reject
            margin = target_similarity - other_similarity
            if margin < self.distinctiveness_threshold:
                logger.debug(f"Candidate rejected: similarity to target {target_id} ({target_similarity:.3f}) "
                           f"not distinctive from person {person_id} ({other_similarity:.3f}), margin={margin:.3f}")
                return False
        
        return True
    
    def find_best_match(
        self,
        candidate_features: Dict[str, np.ndarray],
        similarity_func: callable,
        threshold: float = 0.5
    ) -> Optional[Tuple[int, float]]:
        """
        Find which registered person best matches the candidate.
        
        Args:
            candidate_features: Features extracted from the candidate
            similarity_func: Function to compute similarity
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (person_id, similarity) or None if no match
        """
        best_id = None
        best_similarity = threshold
        
        for person_id, appearance in self.known_persons.items():
            similarity = similarity_func(appearance, candidate_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = person_id
        
        if best_id is not None:
            return (best_id, best_similarity)
        return None


class PersonReIDModel:
    """
    Specialized Person Re-Identification model.
    
    Uses a combination of techniques optimized for person re-identification:
    1. Body part-based feature extraction (upper body, lower body, full body)
    2. Color histogram in multiple color spaces
    3. Deep features from a model fine-tuned for person ReID
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the Person ReID model.
        
        Args:
            device: Device for computation
        """
        self.device = device
        self.feature_dim = 512
        self.model = None
        self.use_deep_features = False
        
        self._load_reid_model()
    
    def _load_reid_model(self):
        """Load a ReID-optimized model."""
        try:
            # Try to use ResNet18 with better pooling for ReID
            from torchvision.models import resnet18, ResNet18_Weights
            
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            
            # Create a ReID-style model with global + part-based pooling
            self.model = torch.nn.Sequential(
                # Feature backbone (remove avgpool and fc)
                *list(base_model.children())[:-2],
            )
            
            # Global average pooling
            self.gap = torch.nn.AdaptiveAvgPool2d(1)
            
            # Horizontal strip pooling for part-based features
            self.strip_pool = torch.nn.AdaptiveAvgPool2d((3, 1))  # 3 horizontal parts
            
            self.model.to(self.device)
            self.gap.to(self.device)
            self.strip_pool.to(self.device)
            self.model.eval()
            
            self.use_deep_features = True
            self.feature_dim = 512 + 512 * 3  # global + 3 parts
            logger.info("Loaded ResNet18-based Person ReID model")
            
        except Exception as e:
            logger.warning(f"Could not load ReID model: {e}")
            self.use_deep_features = False
    
    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract ReID features from a person crop.
        
        Args:
            crop: Person crop (RGB), should be the full person bounding box
            
        Returns:
            Feature vector optimized for person matching
        """
        if not self.use_deep_features or self.model is None:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Resize to standard ReID size (256x128 is common for person ReID)
        crop_resized = cv2.resize(crop, (128, 256))
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        crop_normalized = (crop_resized / 255.0 - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(crop_normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            # Get feature maps
            features = self.model(tensor)  # [1, 512, H, W]
            
            # Global features
            global_feat = self.gap(features).flatten(1)  # [1, 512]
            
            # Part-based features (3 horizontal strips)
            part_feat = self.strip_pool(features)  # [1, 512, 3, 1]
            part_feat = part_feat.flatten(1)  # [1, 512*3]
            
            # Concatenate
            combined = torch.cat([global_feat, part_feat], dim=1)
            
            # L2 normalize
            combined = torch.nn.functional.normalize(combined, p=2, dim=1)
        
        return combined.cpu().numpy().flatten()


class AppearanceExtractor:
    """
    Extracts appearance features for re-identification.
    
    Uses a combination of:
    1. Specialized Person ReID features (when tracking persons)
    2. Body part-based color histograms (upper/lower body)
    3. General CNN features for non-person objects
    """
    
    # COCO class ID for person
    PERSON_CLASS_ID = 0
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the appearance extractor.
        
        Args:
            device: Device to use for computation
        """
        self.device = device
        
        # Person-specific ReID model
        self.person_reid = PersonReIDModel(device)
        
        # General feature extractor for non-person objects
        self._load_general_feature_extractor()
    
    def _load_general_feature_extractor(self):
        """Load a general CNN for non-person feature extraction."""
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            
            self.general_model = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
            self.general_model = torch.nn.Sequential(
                *list(self.general_model.children())[:-1],
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten()
            )
            self.general_model.to(self.device)
            self.general_model.eval()
            self.use_general_cnn = True
            logger.info("Loaded MobileNetV3 for general feature extraction")
            
        except Exception as e:
            logger.warning(f"Could not load general feature extractor: {e}")
            self.general_model = None
            self.use_general_cnn = False
    
    def extract_features(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        mask: Optional[np.ndarray] = None,
        class_id: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        Extract appearance features from a detection.
        
        Args:
            frame: Full frame (RGB)
            bbox: Bounding box (x1, y1, x2, y2)
            mask: Optional segmentation mask
            class_id: Class ID of the detection (0 for person in COCO)
            
        Returns:
            Dictionary with different feature types
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure valid coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return {}
        
        # Crop the region
        crop = frame[y1:y2, x1:x2].copy()
        
        # Apply mask if available
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            masked_crop = crop.copy()
            masked_crop[mask_crop == 0] = 0
        else:
            masked_crop = crop
            mask_crop = None
        
        features = {}
        
        # Use specialized features for persons
        if class_id == self.PERSON_CLASS_ID:
            # Person ReID features
            features['reid'] = self.person_reid.extract_features(crop)
            
            # Body part color histograms
            features['body_color'] = self._extract_body_part_colors(crop, mask_crop)
        else:
            # General CNN features for non-person objects
            if self.use_general_cnn and self.general_model is not None:
                features['cnn'] = self._extract_general_features(crop)
        
        # Color histogram (for all objects)
        features['color_hist'] = self._extract_color_histogram(masked_crop, mask_crop)
        
        # Size features
        features['size'] = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        
        return features
    
    def _extract_body_part_colors(
        self,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract color histograms for different body parts.
        
        Divides the person into upper body (top 40%), middle (40%), and lower body (bottom 20%).
        This helps distinguish people by clothing colors.
        
        Args:
            crop: Person crop (RGB)
            mask: Optional mask
            
        Returns:
            Concatenated color histograms for body parts
        """
        h, w = crop.shape[:2]
        
        # Define body part regions (approximate for standing person)
        # Upper body: top 40% (head + torso)
        # Middle: 40% (torso + upper legs)  
        # Lower: bottom 20% (legs/feet)
        upper_end = int(h * 0.4)
        middle_end = int(h * 0.8)
        
        parts = [
            crop[:upper_end, :],      # Upper body
            crop[upper_end:middle_end, :],  # Middle
            crop[middle_end:, :]      # Lower body
        ]
        
        if mask is not None:
            masks = [
                mask[:upper_end, :],
                mask[upper_end:middle_end, :],
                mask[middle_end:, :]
            ]
        else:
            masks = [None, None, None]
        
        # Extract histogram for each part
        histograms = []
        for part, part_mask in zip(parts, masks):
            if part.size == 0:
                histograms.append(np.zeros(32, dtype=np.float32))
            else:
                hist = self._extract_compact_color_histogram(part, part_mask)
                histograms.append(hist)
        
        return np.concatenate(histograms)
    
    def _extract_compact_color_histogram(
        self,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract a compact color histogram.
        
        Args:
            crop: Image crop (RGB)
            mask: Optional mask
            
        Returns:
            Normalized histogram
        """
        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        
        # Use fewer bins for compact representation
        h_bins, s_bins = 16, 16
        
        if mask is not None and mask.size > 0:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = None
        
        # 2D histogram of H and S (ignore V for illumination invariance)
        hist = cv2.calcHist([hsv], [0, 1], mask_uint8, [h_bins, s_bins], [0, 180, 0, 256])
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-6)
        
        return hist.astype(np.float32)
    
    def _extract_color_histogram(
        self,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract color histogram in HSV space.
        
        Args:
            crop: Cropped image region (RGB)
            mask: Optional mask for the region
            
        Returns:
            Normalized color histogram
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        
        h_bins, s_bins, v_bins = 16, 8, 8
        h_range = [0, 180]
        s_range = [0, 256]
        v_range = [0, 256]
        
        if mask is not None and mask.size > 0:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = None
        
        hist_h = cv2.calcHist([hsv], [0], mask_uint8, [h_bins], h_range)
        hist_s = cv2.calcHist([hsv], [1], mask_uint8, [s_bins], s_range)
        hist_v = cv2.calcHist([hsv], [2], mask_uint8, [v_bins], v_range)
        
        hist = np.concatenate([
            hist_h.flatten(),
            hist_s.flatten(),
            hist_v.flatten()
        ])
        
        hist = hist / (np.sum(hist) + 1e-6)
        return hist.astype(np.float32)
    
    def _extract_general_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract general CNN features.
        
        Args:
            crop: Image crop (RGB)
            
        Returns:
            Feature vector
        """
        crop_resized = cv2.resize(crop, (224, 224))
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        crop_normalized = (crop_resized / 255.0 - mean) / std
        
        tensor = torch.from_numpy(crop_normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            features = self.general_model(tensor)
        
        return features.cpu().numpy().flatten()


class ReIDMatcher:
    """
    Handles re-identification matching between stored appearance and candidates.
    
    Uses a weighted combination of multiple similarity metrics:
    1. Person ReID features (cosine similarity) - primary for persons
    2. Body part color histogram correlation
    3. Size similarity
    
    Motion prediction is disabled as it's unreliable when camera moves.
    """
    
    # Weights for PERSON re-identification - heavily favor deep features
    WEIGHT_REID = 0.75       # Person ReID deep features (most important)
    WEIGHT_BODY_COLOR = 0.20  # Body part colors (clothing)
    WEIGHT_COLOR = 0.03      # General color histogram
    WEIGHT_SIZE = 0.02       # Size (least reliable)
    
    # Weights for NON-PERSON objects
    WEIGHT_CNN_GENERAL = 0.50
    WEIGHT_COLOR_GENERAL = 0.40
    WEIGHT_SIZE_GENERAL = 0.10
    
    # Thresholds - strict for multi-person scenarios
    REID_THRESHOLD = 0.60     # Minimum similarity for re-identification
    REID_MARGIN = 0.10        # Best match must be this much better than second-best
    TIME_DECAY_FACTOR = 0.998  # Very slow decay
    MAX_REID_TIME = 120.0     # 2 minutes max search time
    
    # Person class ID (COCO)
    PERSON_CLASS_ID = 0
    
    @classmethod
    def compute_similarity(
        cls,
        target: TargetAppearance,
        candidate_features: Dict[str, np.ndarray],
        candidate_bbox: Tuple[int, int, int, int],
        current_time: float,
        is_person: bool = False
    ) -> float:
        """
        Compute similarity between stored target and a candidate detection.
        
        Args:
            target: Stored target appearance
            candidate_features: Features extracted from candidate
            candidate_bbox: Bounding box of candidate
            current_time: Current timestamp
            is_person: Whether the target is a person (uses specialized matching)
            
        Returns:
            Similarity score between 0 and 1
        """
        scores = []
        weights = []
        
        # Time decay - reduce confidence for older appearances
        time_since_seen = current_time - target.last_seen_time
        if time_since_seen > cls.MAX_REID_TIME:
            return 0.0
        time_decay = cls.TIME_DECAY_FACTOR ** time_since_seen
        
        if is_person:
            # Use person-specific ReID matching
            return cls._compute_person_similarity(target, candidate_features, time_decay)
        else:
            # Use general object matching
            return cls._compute_general_similarity(target, candidate_features, time_decay)
    
    @classmethod
    def _compute_person_similarity(
        cls,
        target: TargetAppearance,
        candidate_features: Dict[str, np.ndarray],
        time_decay: float
    ) -> float:
        """Compute similarity for person re-identification."""
        scores = []
        weights = []
        
        # 1. Person ReID features (most important)
        if 'reid' in candidate_features:
            target_reid = target.get_average_feature()
            if target_reid is not None:
                candidate_reid = candidate_features['reid']
                # Check dimension compatibility
                if target_reid.shape[0] == candidate_reid.shape[0]:
                    reid_sim = cls._cosine_similarity(target_reid, candidate_reid)
                    # Use a more discriminative transformation
                    # Cosine similarity of 0.8+ should be a strong match
                    # Below 0.5 should be rejected
                    # Apply sigmoid-like transformation centered around 0.6
                    reid_sim_transformed = 1.0 / (1.0 + np.exp(-10 * (reid_sim - 0.6)))
                    scores.append(reid_sim_transformed)
                    weights.append(cls.WEIGHT_REID)
                else:
                    # Dimension mismatch - skip ReID features, rely on other features
                    logger.debug(f"ReID dimension mismatch: {target_reid.shape[0]} vs {candidate_reid.shape[0]}")
        
        # 2. Body part color similarity
        if 'body_color' in candidate_features:
            target_body = target.get_body_color()
            if target_body is not None:
                body_sim = cls._histogram_similarity(target_body, candidate_features['body_color'])
                # Body color should also be discriminative
                body_sim_transformed = body_sim ** 2  # Square to penalize low matches
                scores.append(body_sim_transformed)
                weights.append(cls.WEIGHT_BODY_COLOR)
        
        # 3. General color histogram
        if 'color_hist' in candidate_features:
            target_hist = target.get_average_color_hist()
            if target_hist is not None:
                color_sim = cls._histogram_similarity(target_hist, candidate_features['color_hist'])
                scores.append(color_sim)
                weights.append(cls.WEIGHT_COLOR)
        
        # 4. Size similarity
        if 'size' in candidate_features:
            target_size = target.get_average_size()
            if target_size is not None:
                size_sim = cls._size_similarity(target_size, tuple(candidate_features['size']))
                scores.append(size_sim)
                weights.append(cls.WEIGHT_SIZE)
        
        if not scores:
            return 0.0
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        similarity = np.sum(np.array(scores) * weights) * time_decay
        
        return float(similarity)
    
    @classmethod
    def _compute_general_similarity(
        cls,
        target: TargetAppearance,
        candidate_features: Dict[str, np.ndarray],
        time_decay: float
    ) -> float:
        """Compute similarity for general (non-person) objects."""
        scores = []
        weights = []
        
        # 1. CNN features
        if 'cnn' in candidate_features:
            target_cnn = target.get_average_feature()
            if target_cnn is not None:
                cnn_sim = cls._cosine_similarity(target_cnn, candidate_features['cnn'])
                cnn_sim = (cnn_sim + 1.0) / 2.0
                scores.append(cnn_sim)
                weights.append(cls.WEIGHT_CNN_GENERAL)
        
        # 2. Color histogram
        if 'color_hist' in candidate_features:
            target_hist = target.get_average_color_hist()
            if target_hist is not None:
                color_sim = cls._histogram_similarity(target_hist, candidate_features['color_hist'])
                scores.append(color_sim)
                weights.append(cls.WEIGHT_COLOR_GENERAL)
        
        # 3. Size
        if 'size' in candidate_features:
            target_size = target.get_average_size()
            if target_size is not None:
                size_sim = cls._size_similarity(target_size, tuple(candidate_features['size']))
                scores.append(size_sim)
                weights.append(cls.WEIGHT_SIZE_GENERAL)
        
        if not scores:
            return 0.0
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        similarity = np.sum(np.array(scores) * weights) * time_decay
        
        return float(similarity)
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b + 1e-6))
    
    @staticmethod
    def _histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compute histogram similarity using Bhattacharyya coefficient."""
        # Ensure same length
        min_len = min(len(hist1), len(hist2))
        h1 = hist1[:min_len]
        h2 = hist2[:min_len]
        bc = np.sum(np.sqrt(h1 * h2 + 1e-10))
        return float(np.clip(bc, 0.0, 1.0))
    
    @staticmethod
    def _size_similarity(size1: Tuple[float, float], size2: Tuple[float, float]) -> float:
        """Compute size similarity based on relative difference."""
        w1, h1 = size1
        w2, h2 = size2
        
        w_diff = abs(w1 - w2) / max(w1, w2, 1)
        h_diff = abs(h1 - h2) / max(h1, h2, 1)
        
        similarity = 1.0 - (w_diff + h_diff) / 2
        return max(0.0, similarity)


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
        warmup: bool = True,
        enable_reid: bool = True
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
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.state = TrackerState.UNINITIALIZED
        self.target_track_id: Optional[int] = None  # Current YOLO track ID (changes on ReID)
        self.original_track_id: Optional[int] = None  # Original ID (stable, for user display)
        self.target_class_id: Optional[int] = None
        self.target_class_name: Optional[str] = None  # Store target class name for filtering
        self.tracked_results: List[TrackingResult] = []
        
        # Re-identification settings
        self.enable_reid = enable_reid
        self.target_appearance: Optional[TargetAppearance] = None
        self.reid_threshold = ReIDMatcher.REID_THRESHOLD
        self.frames_lost = 0
        self.max_frames_lost = 150  # ~5 seconds at 30fps before giving up
        
        # Person registry - stores features of all known persons to prevent wrong ID assignment
        self.person_registry = PersonRegistry()
        
        # Temporal consistency tracking - prevent rapid ID switching
        self.last_reid_switch_time: float = 0.0
        self.reid_switch_cooldown: float = 0.5  # Minimum seconds between YOLO ID switches
        self.consecutive_reid_frames: int = 0   # Counter for consecutive ReID frames
        self.reid_confirmation_frames: int = 3  # Require this many frames before switching
        self.pending_reid_match: Optional[Tuple[int, float]] = None  # (track_id, first_seen_time)
        
        # Determine device
        self.device = self._get_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = self._load_model(model_path)
        
        # Initialize appearance extractor for re-identification
        if self.enable_reid:
            self.appearance_extractor = AppearanceExtractor(self.device)
            logger.info("Re-identification enabled")
        else:
            self.appearance_extractor = None
        
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
        
        # If target_class is provided, find first object of that class
        elif target_class is not None:
            for result in results:
                if result.class_name.lower() == target_class.lower():
                    selected_result = result
                    break
        
        # If no specific target, track the first detected object
        else:
            selected_result = results[0]
        
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
    
    def _update_appearance(self, frame: np.ndarray, result: TrackingResult):
        """
        Update the target appearance model.
        
        Args:
            frame: Current frame
            result: Current tracking result
        """
        if self.appearance_extractor is None:
            return
        
        # Extract features - pass class_id for person-specific ReID
        features = self.appearance_extractor.extract_features(
            frame, result.bbox, result.mask, class_id=result.class_id
        )
        
        if not features:
            return
        
        current_time = time.time()
        
        # Initialize or update appearance
        if self.target_appearance is None:
            self.target_appearance = TargetAppearance(
                class_id=result.class_id,
                class_name=result.class_name
            )
        
        # Determine the feature key and get the feature
        feature_key = 'reid' if 'reid' in features else 'cnn' if 'cnn' in features else None
        
        if feature_key:
            new_feature = features[feature_key]
            
            # Check if feature dimension changed (shouldn't happen normally, but handle it)
            if self.target_appearance.feature_history:
                last_dim = self.target_appearance.feature_history[-1].shape[0]
                new_dim = new_feature.shape[0]
                
                if last_dim != new_dim:
                    # Dimension mismatch - clear old features to avoid mixing
                    logger.debug(f"Feature dimension changed from {last_dim} to {new_dim}, clearing history")
                    self.target_appearance.feature_history.clear()
            
            self.target_appearance.feature_history.append(new_feature)
        
        if 'color_hist' in features:
            self.target_appearance.color_hist_history.append(features['color_hist'])
        
        # Update body color for persons
        if 'body_color' in features:
            self.target_appearance.body_color_history.append(features['body_color'])
        
        if 'size' in features:
            self.target_appearance.size_history.append(tuple(features['size']))
        
        # Update position and velocity
        center_x = (result.bbox[0] + result.bbox[2]) / 2
        center_y = (result.bbox[1] + result.bbox[3]) / 2
        
        if self.target_appearance.position_history:
            last_pos = self.target_appearance.position_history[-1]
            dt = current_time - self.target_appearance.last_seen_time
            if dt > 0:
                vx = (center_x - last_pos[0]) / dt
                vy = (center_y - last_pos[1]) / dt
                # Smooth velocity update
                alpha = 0.3
                old_vx, old_vy = self.target_appearance.velocity
                self.target_appearance.velocity = (
                    alpha * vx + (1 - alpha) * old_vx,
                    alpha * vy + (1 - alpha) * old_vy
                )
        
        self.target_appearance.position_history.append((center_x, center_y))
        self.target_appearance.last_seen_time = current_time
        
        # Update the person in the registry
        if self.original_track_id is not None and result.class_id == 0:  # Person class
            self.person_registry.update_person(self.original_track_id, self.target_appearance)
    
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
    
    def _register_other_persons(self, frame: np.ndarray, results: List[TrackingResult]):
        """
        Register other persons seen in the frame to help with distinctiveness checking.
        
        This helps prevent wrong ID assignment by knowing what other people look like.
        
        Args:
            frame: Current frame
            results: All detection results
        """
        if self.appearance_extractor is None:
            return
        
        for result in results:
            # Only register persons
            if result.class_id != 0:  # Not a person
                continue
            
            # Skip the current target
            if result.track_id == self.target_track_id:
                continue
            
            # Skip if already registered with a different display ID
            # (We use negative YOLO IDs as temporary display IDs for non-target persons)
            temp_display_id = -result.track_id if result.track_id > 0 else result.track_id - 1000
            
            if self.person_registry.get_person(temp_display_id) is not None:
                continue  # Already registered
            
            # Extract features
            features = self.appearance_extractor.extract_features(
                frame, result.bbox, result.mask, class_id=result.class_id
            )
            
            if not features:
                continue
            
            # Create an appearance for this person
            other_appearance = TargetAppearance(class_id=result.class_id, class_name=result.class_name)
            
            if 'reid' in features:
                other_appearance.feature_history.append(features['reid'])
            elif 'cnn' in features:
                other_appearance.feature_history.append(features['cnn'])
            
            if 'color_hist' in features:
                other_appearance.color_hist_history.append(features['color_hist'])
            
            if 'body_color' in features:
                other_appearance.body_color_history.append(features['body_color'])
            
            if 'size' in features:
                other_appearance.size_history.append(tuple(features['size']))
            
            other_appearance.last_seen_time = time.time()
            
            # Register with temporary ID
            self.person_registry.register_person(temp_display_id, other_appearance)
    
    def _find_best_match_reid(
        self,
        frame: np.ndarray,
        results: List[TrackingResult]
    ) -> Optional[TrackingResult]:
        """
        Find the detection that best matches the stored appearance using re-identification.
        
        Uses strict matching criteria to avoid confusing similar-looking people:
        1. Similarity must exceed threshold
        2. Best match must have sufficient margin over second-best (when multiple candidates)
        3. Prefers continuing with the same YOLO ID when scores are close
        
        Args:
            frame: Current frame
            results: List of tracking results
            
        Returns:
            Best matching TrackingResult or None
        """
        if self.target_appearance is None or self.appearance_extractor is None:
            return None
        
        current_time = time.time()
        
        # STRICT class filtering - only match same class as target
        target_class_id = self.target_appearance.class_id
        candidates = [r for r in results if r.class_id == target_class_id]
        
        # If no same-class candidates, don't try to match different classes
        if not candidates:
            logger.debug(f"No candidates of class {self.target_class_name} found for re-identification")
            return None
        
        # Check if target is a person (for person-specific ReID)
        is_person = (target_class_id == 0)  # COCO person class ID
        
        # Get target features for raw similarity logging
        target_reid = self.target_appearance.get_average_feature()
        
        # Compute similarity for all candidates
        candidate_scores = []
        
        for result in candidates:
            # Extract features for candidate - pass class_id for person-specific extraction
            features = self.appearance_extractor.extract_features(
                frame, result.bbox, result.mask, class_id=result.class_id
            )
            
            if not features:
                continue
            
            # Log raw cosine similarity for debugging (before transformation)
            # Only compare if dimensions match
            if is_person and 'reid' in features and target_reid is not None:
                if target_reid.shape[0] == features['reid'].shape[0]:
                    raw_cosine = ReIDMatcher._cosine_similarity(target_reid, features['reid'])
                    logger.debug(f"ID {result.track_id}: raw cosine={raw_cosine:.3f}")
                else:
                    logger.debug(f"ID {result.track_id}: feature dim mismatch ({target_reid.shape[0]} vs {features['reid'].shape[0]})")
            
            # Compute similarity with person-specific matching
            similarity = ReIDMatcher.compute_similarity(
                self.target_appearance,
                features,
                result.bbox,
                current_time,
                is_person=is_person
            )
            
            candidate_scores.append((result, similarity))
        
        if not candidate_scores:
            return None
        
        # Sort by similarity (highest first)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Log all candidates for debugging
        logger.debug(f"ReID candidates: {[(r.track_id, f'{s:.3f}') for r, s in candidate_scores]}")
        
        best_match, best_similarity = candidate_scores[0]
        
        # Check if best similarity exceeds threshold
        if best_similarity <= self.reid_threshold:
            logger.debug(f"Best similarity {best_similarity:.3f} below threshold {self.reid_threshold}")
            return None
        
        # If multiple candidates, check margin between best and second-best
        if len(candidate_scores) > 1:
            second_best_match, second_best_similarity = candidate_scores[1]
            margin = best_similarity - second_best_similarity
            
            # Log the comparison
            logger.debug(f"Best: ID {best_match.track_id} ({best_similarity:.3f}), "
                        f"Second: ID {second_best_match.track_id} ({second_best_similarity:.3f}), "
                        f"Margin: {margin:.3f}")
            
            # If margin is too small, be more conservative
            if margin < ReIDMatcher.REID_MARGIN:
                # Check if we should prefer the previous YOLO ID for temporal consistency
                prev_yolo_id = self.target_track_id
                
                # See if any of the top candidates matches our previous YOLO ID
                for result, score in candidate_scores:
                    if result.track_id == prev_yolo_id and score > self.reid_threshold:
                        logger.debug(f"Preferring previous YOLO ID {prev_yolo_id} for temporal consistency "
                                    f"(score: {score:.3f}, margin too small: {margin:.3f})")
                        return result
                
                # If neither matches previous ID and margin is too small, reject
                logger.debug(f"Rejecting match: margin {margin:.3f} < {ReIDMatcher.REID_MARGIN}, "
                            f"ambiguous between ID {best_match.track_id} and ID {second_best_match.track_id}")
                return None
        
        # Check distinctiveness against other known persons in the registry
        if is_person and self.original_track_id is not None:
            # Extract features for the best match for distinctiveness check
            best_features = self.appearance_extractor.extract_features(
                frame, best_match.bbox, best_match.mask, class_id=best_match.class_id
            )
            
            if best_features:
                def similarity_func(appearance: TargetAppearance, features: Dict[str, np.ndarray]) -> float:
                    return ReIDMatcher.compute_similarity(
                        appearance, features, best_match.bbox, current_time, is_person=True
                    )
                
                is_distinctive = self.person_registry.check_distinctiveness(
                    self.original_track_id,
                    best_features,
                    best_similarity,
                    similarity_func
                )
                
                if not is_distinctive:
                    logger.debug(f"Rejecting match: candidate not distinctive enough from other known persons")
                    return None
        
        logger.info(f"Re-identified target as ID {best_match.track_id} with similarity {best_similarity:.3f}")
        return best_match
    
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
    
    def update(
        self,
        frame: np.ndarray,
        target_id: Optional[int] = None
    ) -> Optional[TrackingResult]:
        """
        Update tracking on a new frame.
        
        Uses a multi-stage matching strategy:
        1. First try to match by track ID (ByteTrack)
        2. If track ID lost, use re-identification to find the target
        
        Args:
            frame: New RGB frame
            target_id: Optional specific track ID to follow. If provided, will switch
                      to tracking this ID instead of the current target.
            
        Returns:
            TrackingResult for the target object, or None if lost
        """
        if self.state == TrackerState.UNINITIALIZED:
            logger.warning("Tracker not initialized. Call initialize_tracking first.")
            return None
        
        # If a specific target_id is requested, switch to it
        if target_id is not None and target_id != self.target_track_id:
            logger.info(f"Switching target from {self.target_track_id} to {target_id}")
            self.target_track_id = target_id
            self.original_track_id = target_id  # Update original ID when explicitly switching targets
            self.frames_lost = 0
            # Reset appearance model since we're tracking a new target
            self.target_appearance = None
        
        # Perform tracking
        results = self.track(frame, persist=True)
        
        # Register other visible persons for distinctiveness checking
        if self.enable_reid and len(results) > 1:
            self._register_other_persons(frame, results)
        
        # Stage 1: Try to find target by track ID
        for result in results:
            if result.track_id == self.target_track_id:
                self.state = TrackerState.TRACKING
                self.frames_lost = 0
                
                # Reset pending ReID match since we found our target by ID
                self.pending_reid_match = None
                self.consecutive_reid_frames = 0
                
                # Update class info if needed (in case we switched targets)
                if self.target_class_id != result.class_id:
                    self.target_class_id = result.class_id
                    self.target_class_name = result.class_name
                
                # Update appearance model
                if self.enable_reid:
                    self._update_appearance(frame, result)
                
                # Return result with consistent original track ID
                return self._with_original_id(result)
        
        # Stage 2: Track ID not found, try re-identification
        # Note: frames_lost is incremented AFTER we try ReID, so that a successful
        # pending match doesn't cause "lost" to be reported
        
        if self.enable_reid and self.frames_lost <= self.max_frames_lost:
            self.state = TrackerState.REIDENTIFYING
            
            # Try to re-identify using appearance
            reid_match = self._find_best_match_reid(frame, results)
            
            if reid_match is not None:
                current_time = time.time()
                new_yolo_id = reid_match.track_id
                
                # Check if this is the same ID as pending match
                if self.pending_reid_match is not None:
                    pending_id, pending_start_time = self.pending_reid_match
                    
                    if pending_id == new_yolo_id:
                        # Same ID confirmed again
                        self.consecutive_reid_frames += 1
                        
                        # Check if we've confirmed for enough frames
                        if self.consecutive_reid_frames >= self.reid_confirmation_frames:
                            # Also check cooldown from last switch
                            time_since_last_switch = current_time - self.last_reid_switch_time
                            
                            if time_since_last_switch >= self.reid_switch_cooldown:
                                # Confirmed! Accept the new ID
                                old_yolo_id = self.target_track_id
                                self.target_track_id = new_yolo_id
                                self.state = TrackerState.TRACKING
                                self.frames_lost = 0
                                self.last_reid_switch_time = current_time
                                self.pending_reid_match = None
                                self.consecutive_reid_frames = 0
                                
                                # Update appearance
                                self._update_appearance(frame, reid_match)
                                
                                logger.info(f"Re-identified target (YOLO ID: {old_yolo_id} -> {self.target_track_id}, "
                                           f"display ID: {self.original_track_id}, confirmed over {self.reid_confirmation_frames} frames)")
                                return self._with_original_id(reid_match)
                            else:
                                logger.debug(f"ReID cooldown: {time_since_last_switch:.2f}s < {self.reid_switch_cooldown}s")
                        else:
                            logger.debug(f"ReID pending confirmation: {self.consecutive_reid_frames}/{self.reid_confirmation_frames} frames")
                    else:
                        # Different ID - reset pending
                        logger.debug(f"ReID candidate changed from {pending_id} to {new_yolo_id}, resetting confirmation")
                        self.pending_reid_match = (new_yolo_id, current_time)
                        self.consecutive_reid_frames = 1
                else:
                    # Start new pending match
                    self.pending_reid_match = (new_yolo_id, current_time)
                    self.consecutive_reid_frames = 1
                    logger.debug(f"ReID pending: ID {new_yolo_id}, need {self.reid_confirmation_frames} frames to confirm")
                
                # While confirming, we have a valid match - don't increment frames_lost
                # Return the match for visualization
                return self._with_original_id(reid_match)
            else:
                # No match found - increment frames lost and reset pending
                self.frames_lost += 1
                self.pending_reid_match = None
                self.consecutive_reid_frames = 0
        else:
            # ReID disabled or exceeded max frames
            self.frames_lost += 1
        
        # Target not found
        if self.frames_lost > self.max_frames_lost:
            self.state = TrackerState.LOST
            logger.warning(f"Target lost after {self.frames_lost} frames")
        
        return None
    
    def _with_original_id(self, result: TrackingResult) -> TrackingResult:
        """
        Return a copy of the result with the original (stable) track ID.
        
        This ensures users always see the same track ID for the target,
        even if YOLO assigns a new ID after re-identification.
        """
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
        # Clear the person registry
        self.person_registry.clear()
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
        internal_target_id: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Draw all tracking results on the frame.
        
        Args:
            frame: Input frame
            results: List of TrackingResult objects
            target_id: Display ID of target to highlight (stable ID shown to user)
            internal_target_id: Internal YOLO ID used to find the target in results
            **kwargs: Additional arguments for draw_tracking_result
            
        Returns:
            Frame with all drawings
        """
        output = frame.copy()
        
        # Use internal_target_id for matching if provided, otherwise use target_id
        match_id = internal_target_id if internal_target_id is not None else target_id
        
        # Track if we've already found and drawn the target (to prevent duplicates)
        target_drawn = False
        
        for result in results:
            # Only match if match_id is valid (not None and not -1)
            # and result.track_id is also valid (not -1)
            is_target = (
                match_id is not None and 
                match_id != -1 and 
                result.track_id != -1 and
                result.track_id == match_id and
                not target_drawn  # Only match once
            )
            
            if is_target and target_id is not None:
                # For the target, create a copy with the stable display ID
                display_result = TrackingResult(
                    track_id=target_id,  # Use stable display ID
                    bbox=result.bbox,
                    mask=result.mask,
                    confidence=result.confidence,
                    class_id=result.class_id,
                    class_name=result.class_name
                )
                output = cls.draw_tracking_result(output, display_result, highlight=True, **kwargs)
                target_drawn = True
            else:
                # Draw with original YOLO ID
                output = cls.draw_tracking_result(output, result, highlight=False, **kwargs)
        
        return output
    
    @classmethod
    def draw_info_panel(
        cls,
        frame: np.ndarray,
        tracker_state: TrackerState,
        target_id: Optional[int],
        fps: float,
        frames_lost: int = 0,
        target_class: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw information panel on the frame.
        
        Args:
            frame: Input frame
            tracker_state: Current tracker state
            target_id: Current target ID
            fps: Current FPS
            frames_lost: Number of frames target has been lost
            target_class: Class name of the target
            
        Returns:
            Frame with info panel
        """
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 155), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # State color coding
        state_colors = {
            TrackerState.UNINITIALIZED: (128, 128, 128),  # Gray
            TrackerState.INITIALIZED: (255, 255, 0),      # Cyan
            TrackerState.TRACKING: (0, 255, 0),           # Green
            TrackerState.LOST: (0, 0, 255),               # Red
            TrackerState.REIDENTIFYING: (0, 165, 255),    # Orange
        }
        state_color = state_colors.get(tracker_state, (255, 255, 255))
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"State: {tracker_state.value}", (20, 35),
                    font, 0.6, state_color, 2)
        
        target_str = f"ID: {target_id}" if target_id else "None"
        if target_class:
            target_str += f" ({target_class})"
        cv2.putText(frame, f"Target: {target_str}", (20, 60),
                    font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 85),
                    font, 0.6, (255, 255, 255), 2)
        
        if frames_lost > 0:
            cv2.putText(frame, f"Lost frames: {frames_lost}", (20, 110),
                        font, 0.6, (0, 165, 255), 2)
            cv2.putText(frame, "Searching with ReID...", (20, 135),
                        font, 0.5, (0, 165, 255), 1)
        else:
            cv2.putText(frame, "ReID: Enabled", (20, 110),
                        font, 0.6, (0, 255, 0), 2)
        
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
            "SPACE - Initialize tracking (person class)",
            "P - Track first person detected",
            "R - Reset tracker",
            "M - Toggle mask display",
            "A - Toggle show all objects",
            "Q/ESC - Quit"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 175), (420, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = h - 130
        for instruction in instructions:
            cv2.putText(frame, instruction, (20, y_offset),
                        font, 0.5, (255, 255, 255), 1)
            y_offset += 22
        
        return frame


def main():
    """
    Main function demonstrating YOLO tracking with laptop camera.
    
    This function:
    1. Initializes the YOLO tracker with re-identification
    2. Opens the laptop camera
    3. Provides interactive controls for tracking
    4. Visualizes tracking results with segmentation masks
    5. Demonstrates re-identification after occlusion
    """
    print("=" * 60)
    print("YOLO Object Tracking with Segmentation and Re-ID")
    print("=" * 60)
    
    # Initialize tracker
    print("\nInitializing tracker...")
    try:
        tracker = YOLOTracker(
            model_path="yolo11n-seg.pt",  # Use nano model for speed
            confidence_threshold=0.5,
            warmup=True,
            enable_reid=True  # Enable re-identification
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
    print("\nTIP: The tracker can re-identify objects even after occlusion or going off-screen!")
    
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
                # Perform tracking update (includes re-identification)
                target_result = tracker.update(frame_rgb)
                results = tracker.get_all_tracks()
            
            # Convert back to BGR for display
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw results - use original_track_id for display, internal ID for matching
            if show_all_objects:
                display_frame = TrackingVisualizer.draw_all_results(
                    display_frame,
                    results,
                    target_id=tracker.original_track_id,
                    internal_target_id=tracker.target_track_id,  # Internal YOLO ID for matching
                    draw_mask=show_mask
                )
            elif tracker.state in [TrackerState.TRACKING, TrackerState.REIDENTIFYING]:
                # Get the result returned by update() which already has original_track_id
                if target_result is not None:
                    display_frame = TrackingVisualizer.draw_tracking_result(
                        display_frame,
                        target_result,
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
            
            # Draw info panel and instructions - use original_track_id for display
            display_frame = TrackingVisualizer.draw_info_panel(
                display_frame,
                tracker.state,
                tracker.original_track_id,
                fps,
                tracker.frames_lost,
                tracker.target_class_name
            )
            display_frame = TrackingVisualizer.draw_instructions(display_frame)
            
            # Show frame
            cv2.imshow("YOLO Tracking with Re-ID", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nQuitting...")
                break
                
            elif key == ord(' '):  # SPACE - Initialize tracking on first object
                print("\nInitializing tracking on first detected object...")
                success = tracker.initialize_tracking(frame_rgb)
                if success:
                    print(f"Tracking initialized: {tracker.target_class_name} (ID: {tracker.original_track_id})")
                    print("Try occluding the object or moving it off-screen - it will be re-identified!")
                else:
                    print("Failed to initialize tracking - no objects detected")
            
            elif key == ord('p'):  # P - Track first person
                print("\nInitializing tracking on first person...")
                success = tracker.initialize_tracking(frame_rgb, target_class="person")
                if success:
                    print(f"Tracking person (ID: {tracker.original_track_id})")
                    print("Try moving out of frame and back - ReID will find you!")
                else:
                    print("Failed to find a person to track")
                    
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
