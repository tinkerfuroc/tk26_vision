#!/usr/bin/env python3
"""
YOLO-based Object Tracking with Segmentation and Re-Identification

This module provides a YOLOTracker class that uses YOLO models for object tracking
with segmentation capabilities. It supports GPU acceleration when available and
includes appearance-based re-identification for robust tracking through occlusions.

Author: TinkerFuroc
Date: 2025
"""

import copy
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
        # Lowered from 0.15 to 0.03 because the "other person" registry can get contaminated
        # with target features during ID switches
        self.distinctiveness_threshold = 0.03
    
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
    
    def clear_temporary_ids(self):
        """
        Clear persons registered with temporary IDs (ID <= -1000).
        These are often contaminated with target features during ID bounces.
        """
        temp_ids = [pid for pid in self.known_persons.keys() if pid <= -1000]
        for pid in temp_ids:
            del self.known_persons[pid]
            logger.debug(f"Removed temporary person ID {pid} from registry")
        if temp_ids:
            logger.info(f"Cleared {len(temp_ids)} temporary person IDs from registry")
    
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
            logger.info(f"Distinctiveness check: target {target_id} sim={target_similarity:.3f}, "
                       f"other person {person_id} sim={other_similarity:.3f}, margin={margin:.3f} "
                       f"(required: {self.distinctiveness_threshold})")
            if margin < self.distinctiveness_threshold:
                logger.info(f"Candidate rejected: similarity to target {target_id} ({target_similarity:.3f}) "
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
    Enhanced Person Re-Identification model.
    
    Uses multiple complementary feature types for robust person matching:
    1. Deep CNN features with part-based pooling (global + horizontal parts)
    2. Channel attention for emphasizing discriminative features
    3. Multiple spatial scales for better robustness
    
    Key insight: Generic ImageNet features are NOT discriminative enough for person ReID.
    This model applies transformations to make features more person-specific.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the Person ReID model.
        
        Args:
            device: Device for computation
        """
        self.device = device
        self.feature_dim = 512  # Output feature dimension
        self.model = None
        self.use_deep_features = False
        
        self._load_reid_model()
    
    def _load_reid_model(self):
        """Load an enhanced ReID model with attention mechanisms."""
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            
            # Use ResNet50 for better feature extraction (deeper = more discriminative)
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # Feature backbone (remove avgpool and fc) - outputs 2048 channels
            self.backbone = torch.nn.Sequential(
                *list(base_model.children())[:-2],
            )
            
            # Channel attention module - helps focus on discriminative channels
            self.channel_attention = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(2048, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(512, 2048),
                torch.nn.Sigmoid()
            )
            
            # Bottleneck to reduce to 512 dimensions (standard ReID size)
            self.bottleneck = torch.nn.Sequential(
                torch.nn.Linear(2048, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(inplace=True)
            )
            
            # Part-based bottlenecks (for 4 horizontal parts)
            self.part_bottlenecks = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(2048, 128),
                    torch.nn.BatchNorm1d(128),
                    torch.nn.ReLU(inplace=True)
                ) for _ in range(4)
            ])
            
            # Global average pooling
            self.gap = torch.nn.AdaptiveAvgPool2d(1)
            
            # Part pooling - 4 horizontal strips (head, upper body, lower body, legs)
            self.part_pool = torch.nn.AdaptiveAvgPool2d((4, 1))
            
            # Move to device
            self.backbone.to(self.device)
            self.channel_attention.to(self.device)
            self.bottleneck.to(self.device)
            self.part_bottlenecks.to(self.device)
            self.gap.to(self.device)
            self.part_pool.to(self.device)
            
            # Set to eval mode
            self.backbone.eval()
            self.channel_attention.eval()
            self.bottleneck.eval()
            self.part_bottlenecks.eval()
            
            self.use_deep_features = True
            self.feature_dim = 512 + 128 * 4  # global (512) + 4 parts (128 each)
            logger.info("Loaded ResNet50-based Person ReID model with attention")
            
        except Exception as e:
            logger.warning(f"Could not load enhanced ReID model: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Fallback to simpler model if enhanced fails."""
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = torch.nn.Sequential(
                *list(base_model.children())[:-2],
            )
            self.gap = torch.nn.AdaptiveAvgPool2d(1)
            self.backbone.to(self.device)
            self.gap.to(self.device)
            self.backbone.eval()
            
            self.channel_attention = None
            self.bottleneck = None
            self.part_bottlenecks = None
            self.part_pool = None
            
            self.use_deep_features = True
            self.feature_dim = 512
            logger.info("Loaded fallback ResNet18 ReID model")
            
        except Exception as e:
            logger.warning(f"Could not load fallback model: {e}")
            self.use_deep_features = False
    
    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract discriminative ReID features from a person crop.
        
        Args:
            crop: Person crop (RGB), should be the full person bounding box
            
        Returns:
            L2-normalized feature vector optimized for person matching
        """
        if not self.use_deep_features or self.backbone is None:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Resize to standard ReID size (256x128 is standard for person ReID)
        # Height > Width because people are taller than wide
        crop_resized = cv2.resize(crop, (128, 256))
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        crop_normalized = (crop_resized / 255.0 - mean) / std
        
        # Convert to tensor [1, 3, 256, 128]
        tensor = torch.from_numpy(crop_normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            # Get feature maps [1, 2048, 8, 4] or [1, 512, 8, 4] for ResNet18
            features = self.backbone(tensor)
            
            if self.channel_attention is not None:
                # Apply channel attention
                attn_weights = self.channel_attention(features)
                attn_weights = attn_weights.view(-1, features.shape[1], 1, 1)
                features = features * attn_weights
                
                # Global features
                global_feat = self.gap(features).flatten(1)  # [1, 2048]
                global_feat = self.bottleneck(global_feat)  # [1, 512]
                
                # Part-based features (4 horizontal strips)
                part_features = self.part_pool(features)  # [1, 2048, 4, 1]
                part_feats = []
                for i in range(4):
                    part_i = part_features[:, :, i, :].flatten(1)  # [1, 2048]
                    part_i = self.part_bottlenecks[i](part_i)  # [1, 128]
                    part_feats.append(part_i)
                part_feat = torch.cat(part_feats, dim=1)  # [1, 512]
                
                # Concatenate global + parts
                combined = torch.cat([global_feat, part_feat], dim=1)  # [1, 1024]
            else:
                # Simple fallback
                combined = self.gap(features).flatten(1)  # [1, 512]
            
            # L2 normalize - CRITICAL for cosine similarity to work properly
            combined = torch.nn.functional.normalize(combined, p=2, dim=1)
        
        return combined.cpu().numpy().flatten()
        
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
        Extract discriminative color features for different body parts.
        
        Uses LAB color space (perceptually uniform) and multiple body regions
        for better person discrimination. Different people wear different colored
        clothes on different body parts.
        
        Body parts:
        - Head region (top 15%): Usually skin/hair color
        - Upper torso (15-35%): Shirt/jacket upper
        - Lower torso (35-55%): Shirt/jacket lower  
        - Upper legs (55-75%): Pants upper
        - Lower legs (75-100%): Pants lower / shoes
        
        Args:
            crop: Person crop (RGB)
            mask: Optional segmentation mask
            
        Returns:
            Concatenated multi-scale color features for body parts
        """
        h, w = crop.shape[:2]
        
        if h < 10 or w < 5:
            return np.zeros(160, dtype=np.float32)  # 5 parts * 32 bins
        
        # Define 5 body part regions for finer discrimination
        part_boundaries = [
            (0.0, 0.15),    # Head
            (0.15, 0.35),   # Upper torso
            (0.35, 0.55),   # Lower torso
            (0.55, 0.75),   # Upper legs
            (0.75, 1.0),    # Lower legs
        ]
        
        parts = []
        masks_list = []
        
        for start_ratio, end_ratio in part_boundaries:
            y_start = int(h * start_ratio)
            y_end = int(h * end_ratio)
            if y_end > y_start:
                parts.append(crop[y_start:y_end, :])
                if mask is not None:
                    masks_list.append(mask[y_start:y_end, :])
                else:
                    masks_list.append(None)
            else:
                parts.append(None)
                masks_list.append(None)
        
        # Extract histogram for each part using LAB color space
        histograms = []
        for part, part_mask in zip(parts, masks_list):
            if part is None or part.size == 0:
                histograms.append(np.zeros(32, dtype=np.float32))
            else:
                hist = self._extract_lab_color_histogram(part, part_mask)
                histograms.append(hist)
        
        return np.concatenate(histograms)
    
    def _extract_lab_color_histogram(
        self,
        crop: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Extract color histogram in LAB color space.
        
        LAB is perceptually uniform - equal distances in LAB correspond to
        equal perceived color differences. This makes it better for distinguishing
        similar colors (like different shades of blue shirts).
        
        Args:
            crop: Image crop (RGB)
            mask: Optional mask
            
        Returns:
            Normalized LAB color histogram (32 bins)
        """
        if crop.size == 0:
            return np.zeros(32, dtype=np.float32)
        
        # Convert RGB to LAB
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
        except:
            return np.zeros(32, dtype=np.float32)
        
        # Use a, b channels (color information, ignore L for illumination invariance)
        # But also include some L information for brightness-based discrimination
        a_bins, b_bins = 12, 12
        l_bins = 8
        
        if mask is not None and mask.size > 0:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = None
        
        # Histogram of a and b channels (color)
        # a: green-red, b: blue-yellow
        hist_ab = cv2.calcHist([lab], [1, 2], mask_uint8, [a_bins, b_bins], [0, 256, 0, 256])
        hist_ab = hist_ab.flatten()
        
        # Histogram of L channel (brightness) - helps distinguish light vs dark clothing
        hist_l = cv2.calcHist([lab], [0], mask_uint8, [l_bins], [0, 256])
        hist_l = hist_l.flatten()
        
        # Combine and normalize
        hist = np.concatenate([hist_ab, hist_l])  # 144 + 8 = 152, but we want 32
        
        # Reduce to 32 dimensions using stride sampling
        # Actually, let's use a simpler histogram
        h_bins, s_bins = 16, 16  # 256 total
        
        # Use HSV for the main histogram (proven to work well)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask_uint8, [h_bins, s_bins], [0, 180, 0, 256])
        hist = hist.flatten()  # 256 values
        
        # Add LAB a-b for extra discrimination
        lab_hist = cv2.calcHist([lab], [1, 2], mask_uint8, [8, 8], [0, 256, 0, 256])
        lab_hist = lab_hist.flatten()  # 64 values
        
        # Combine: dominant HSV + LAB
        # Downsample HSV to 192 bins, LAB to 64 = 256 total, then to 32
        combined = np.concatenate([hist, lab_hist])  # 320 values
        
        # Normalize
        combined = combined / (np.sum(combined) + 1e-6)
        
        # Reduce to 32 dimensions by averaging groups
        n_groups = 32
        group_size = len(combined) // n_groups
        reduced = np.array([
            np.sum(combined[i*group_size:(i+1)*group_size]) 
            for i in range(n_groups)
        ], dtype=np.float32)
        
        # Re-normalize
        reduced = reduced / (np.sum(reduced) + 1e-6)
        
        return reduced
    
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
    1. Person ReID deep features (ResNet50 with attention)
    2. Body part color histograms (5-part LAB+HSV features)
    3. General color histogram
    4. Size consistency
    
    Key insight: For robust ReID, we need multiple complementary features.
    Deep features capture body shape/pose, color features capture clothing.
    """
    
    # Weights for PERSON re-identification
    # Deep features are most discriminative when trained properly
    # Body color is critical as a backup and for clothing-based discrimination
    WEIGHT_REID = 0.50        # Person ReID deep features (ResNet50)
    WEIGHT_BODY_COLOR = 0.35  # Body part colors (clothing) - CRITICAL
    WEIGHT_COLOR = 0.10       # General color histogram
    WEIGHT_SIZE = 0.05        # Size (least reliable but helps)
    
    # Weights for NON-PERSON objects
    WEIGHT_CNN_GENERAL = 0.50
    WEIGHT_COLOR_GENERAL = 0.40
    WEIGHT_SIZE_GENERAL = 0.10
    
    # Thresholds - balanced for accuracy vs pose variation tolerance
    # Analysis from logs shows:
    #   - Same person continuous tracking: reid ~0.85-0.98, body ~0.80-0.97
    #   - Same person with pose change: reid ~0.70-0.80, body ~0.70-0.85
    #   - Different person (WRONG match): reid ~0.55-0.70, body ~0.60-0.75
    # Setting thresholds to accept pose variation but reject different people
    REID_THRESHOLD = 0.60     # Minimum COMBINED similarity for re-identification
    REID_MARGIN = 0.08        # Best match must be clearly better than second-best
    TIME_DECAY_FACTOR = 1.0   # NO time decay - features should be stable over time
    MAX_REID_TIME = 600.0     # 10 minutes max search time
    
    # CRITICAL: Minimum RAW reid similarity
    # Same person with pose change: ~0.70-0.80
    # Different person: ~0.55-0.70
    # Set floor at 0.55 to allow pose variation while rejecting most wrong matches
    MIN_REID_SIMILARITY_RAW = 0.55  # Raw cosine similarity floor
    
    # Minimum body color similarity - more tolerant for lighting variation
    # Same person: ~0.70-0.95, Different person: ~0.60-0.75  
    MIN_BODY_COLOR_SIMILARITY = 0.50  # Reject if body colors are too different
    
    # Legacy threshold for backward compatibility
    MIN_REID_SIMILARITY = 0.40  # Transformed similarity minimum
    
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
        """
        Compute similarity for person re-identification.
        
        Uses a strict multi-stage verification approach:
        1. Deep ReID features (most discriminative)
        2. Body part color features (clothing appearance)
        3. General color distribution
        4. Size consistency
        
        Key insight: For ReID to work, we need features where:
        - Same person across time: similarity > 0.7
        - Different people: similarity < 0.4
        
        If features don't show this separation, they're not discriminative enough.
        """
        reid_sim_raw = None
        body_color_sim = None
        color_sim = None
        size_sim = None
        
        # 1. Person ReID features (most important) - use raw cosine similarity
        if 'reid' in candidate_features:
            target_reid = target.get_average_feature()
            if target_reid is not None:
                candidate_reid = candidate_features['reid']
                # Check dimension compatibility
                if target_reid.shape[0] == candidate_reid.shape[0]:
                    # Raw cosine similarity for L2-normalized vectors is in [-1, 1]
                    # From logs: Same person ~0.70-0.98 (pose dependent), Different person ~0.55-0.70
                    reid_sim_raw = cls._cosine_similarity(target_reid, candidate_reid)
                    
                    # Hard rejection: RAW reid similarity must be above floor
                    # Set conservatively to allow pose variation
                    if reid_sim_raw < cls.MIN_REID_SIMILARITY_RAW:
                        logger.info(f"HARD REJECT: ReID raw similarity {reid_sim_raw:.3f} < {cls.MIN_REID_SIMILARITY_RAW}")
                        return 0.0
                else:
                    logger.debug(f"ReID dimension mismatch: {target_reid.shape[0]} vs {candidate_reid.shape[0]}")
        
        # 2. Body part color similarity - use histogram intersection (stricter than Bhattacharyya)
        if 'body_color' in candidate_features:
            target_body = target.get_body_color()
            if target_body is not None:
                # Use histogram intersection instead of Bhattacharyya
                # Intersection gives 1.0 only for identical histograms
                body_color_sim = cls._histogram_intersection(target_body, candidate_features['body_color'])
                
                # Hard rejection: if clothing colors are too different
                if body_color_sim < cls.MIN_BODY_COLOR_SIMILARITY:
                    logger.info(f"Rejecting candidate: body color similarity {body_color_sim:.3f} < {cls.MIN_BODY_COLOR_SIMILARITY}")
                    return 0.0
        
        # 3. General color histogram
        if 'color_hist' in candidate_features:
            target_hist = target.get_average_color_hist()
            if target_hist is not None:
                color_sim = cls._histogram_intersection(target_hist, candidate_features['color_hist'])
        
        # 4. Size similarity
        if 'size' in candidate_features:
            target_size = target.get_average_size()
            if target_size is not None:
                size_sim = cls._size_similarity(target_size, tuple(candidate_features['size']))
        
        # Combine scores with proper weighting
        # The key is that ReID features should dominate when available
        scores = []
        weights = []
        
        if reid_sim_raw is not None:
            # Transform cosine similarity: [-1, 1] -> [0, 1]
            # Then apply a steeper transformation to penalize low similarities
            reid_normalized = (reid_sim_raw + 1.0) / 2.0  # [0, 1]
            # Apply power to make it steeper (penalize low similarities more)
            reid_transformed = reid_normalized ** 1.5
            scores.append(reid_transformed)
            weights.append(cls.WEIGHT_REID)
        
        if body_color_sim is not None:
            # Body color is already in [0, 1], apply mild steepening
            body_transformed = body_color_sim ** 1.3
            scores.append(body_transformed)
            weights.append(cls.WEIGHT_BODY_COLOR)
        
        if color_sim is not None:
            scores.append(color_sim)
            weights.append(cls.WEIGHT_COLOR)
        
        if size_sim is not None:
            scores.append(size_sim)
            weights.append(cls.WEIGHT_SIZE)
        
        if not scores:
            return 0.0
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        raw_similarity = np.sum(np.array(scores) * weights)
        similarity = raw_similarity * time_decay
        
        # Detailed logging with component breakdown
        components = []
        if reid_sim_raw is not None:
            components.append(f"reid={reid_sim_raw:.3f}")
        if body_color_sim is not None:
            components.append(f"body={body_color_sim:.3f}")
        if color_sim is not None:
            components.append(f"color={color_sim:.3f}")
        if size_sim is not None:
            components.append(f"size={size_sim:.3f}")
        
        logger.info(f"Person similarity: final={similarity:.3f} ({', '.join(components)})")
        
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
    def _histogram_intersection(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compute histogram similarity using histogram intersection.
        
        Intersection is stricter than Bhattacharyya - it only counts overlapping parts.
        For two identical histograms, intersection = 1.0
        For completely different histograms, intersection = 0.0
        
        This is better for distinguishing different clothing colors because
        it doesn't give partial credit for "similar" colors.
        """
        # Ensure same length
        min_len = min(len(hist1), len(hist2))
        h1 = hist1[:min_len]
        h2 = hist2[:min_len]
        
        # Normalize histograms to sum to 1
        h1_sum = np.sum(h1)
        h2_sum = np.sum(h2)
        
        if h1_sum < 1e-6 or h2_sum < 1e-6:
            return 0.0
        
        h1_norm = h1 / h1_sum
        h2_norm = h2 / h2_sum
        
        # Histogram intersection: sum of min at each bin
        intersection = np.sum(np.minimum(h1_norm, h2_norm))
        
        return float(np.clip(intersection, 0.0, 1.0))
    
    @staticmethod
    def _chi_square_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compute chi-square distance between histograms.
        
        Chi-square is very sensitive to differences - good for ReID.
        Returns similarity (1 - normalized distance).
        """
        min_len = min(len(hist1), len(hist2))
        h1 = hist1[:min_len]
        h2 = hist2[:min_len]
        
        # Chi-square distance
        denominator = h1 + h2 + 1e-10
        chi_sq = np.sum((h1 - h2) ** 2 / denominator)
        
        # Convert to similarity (0 distance = 1 similarity)
        # Chi-square can be large, so we use exponential decay
        similarity = np.exp(-chi_sq * 0.5)
        
        return float(np.clip(similarity, 0.0, 1.0))
    
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
        self.max_frames_lost = 600  # ~20 seconds at 30fps before giving up (increased for long-term)
        
        # Spatial continuity tracking - last known position for re-id preference
        self.last_known_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_known_center: Optional[Tuple[float, float]] = None
        
        # Occlusion detection
        self.occlusion_iou_threshold: float = 0.3  # IoU threshold to consider occlusion
        self.is_occluded: bool = False
        self.occlusion_start_time: Optional[float] = None
        self.pre_occlusion_appearance: Optional[TargetAppearance] = None  # Saved appearance before occlusion
        self.frames_since_occlusion_ended: int = 0
        self.occlusion_recovery_frames: int = 20  # Frames to wait after occlusion before trusting track
        
        # Person registry - stores features of all known persons to prevent wrong ID assignment
        self.person_registry = PersonRegistry()
        
        # Temporal consistency tracking - prevent rapid ID switching
        self.last_reid_switch_time: float = 0.0
        self.reid_switch_cooldown: float = 1.0  # Minimum seconds between YOLO ID switches (increased)
        self.consecutive_reid_frames: int = 0   # Counter for consecutive ReID frames
        self.reid_confirmation_frames: int = 8  # Require this many frames before switching (increased for stability)
        self.pending_reid_match: Optional[Tuple[int, float]] = None  # (track_id, first_seen_time)
        
        # Store last results for debug visualization
        self.last_results: List[TrackingResult] = []
        
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
            # Get bounding box coordinates
            x1, y1, x2, y2 = xyxy_np[i].astype(int)
            
            # Get track ID (if available)
            track_id = -1
            if id_np is not None:
                track_id = int(id_np[i])
            
            # Get confidence and class
            confidence = float(conf_np[i])
            class_id = int(cls_np[i])
            class_name = names[class_id]
            
            # Get segmentation mask (if available) - handle both tensor and numpy
            mask = None
            if masks is not None and i < len(masks):
                # Handle different ways masks can be stored in ultralytics
                try:
                    mask_obj = masks[i]
                    if hasattr(mask_obj, 'data'):
                        mask_data = mask_obj.data
                        # data could be tensor or already accessed
                        if hasattr(mask_data, '__getitem__'):
                            mask_data = mask_data[0]
                        if hasattr(mask_data, 'cpu'):
                            mask = mask_data.cpu().numpy()
                        elif isinstance(mask_data, np.ndarray):
                            mask = mask_data
                        else:
                            mask = np.asarray(mask_data)
                    elif hasattr(mask_obj, 'xy'):
                        # Polygon format - skip for now
                        mask = None
                    else:
                        mask = None
                except Exception as e:
                    logger.debug(f"Failed to extract mask: {e}")
                    mask = None
                
                if mask is not None:
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
        Only registers new persons that haven't been seen before, limited to max 3 per frame.
        
        Args:
            frame: Current frame
            results: All detection results
        """
        if self.appearance_extractor is None:
            return
        
        registered_count = 0
        max_register_per_frame = 2  # Limit to avoid performance impact
        
        for result in results:
            if registered_count >= max_register_per_frame:
                break
            
            # Only register persons
            if result.class_id != 0:  # Not a person
                continue
            
            # Skip persons with invalid track IDs (ID -1 or less)
            # These are often the target with a temporary ID, and registering them
            # would contaminate the registry with target features
            if result.track_id < 0:
                continue
            
            # Skip the current target
            if result.track_id == self.target_track_id:
                continue
            
            # Also skip if this might be the pending ReID match
            if self.pending_reid_match is not None and result.track_id == self.pending_reid_match[0]:
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
            registered_count += 1
    
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
        target_class_name = self.target_appearance.class_name
        
        # For person tracking, we MUST ensure we only match person class (class_id=0)
        # Also filter out invalid track IDs (-1)
        if target_class_id == 0:  # Person
            candidates = [r for r in results if r.class_id == 0 and r.class_name.lower() == 'person' and r.track_id >= 0]
        else:
            candidates = [r for r in results if r.class_id == target_class_id and r.track_id >= 0]
        
        # If no same-class candidates, don't try to match different classes
        if not candidates:
            logger.info(f"ReID: No valid person candidates (track_id >= 0) for re-identification. "
                       f"Total results: {len(results)}, persons with invalid IDs: {sum(1 for r in results if r.class_id == 0 and r.track_id < 0)}")
            return None
        
        logger.debug(f"ReID: {len(candidates)} candidates of class '{target_class_name}' from {len(results)} total detections")
        
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
                logger.debug(f"ID {result.track_id}: No features extracted")
                continue
            
            # Log detailed feature comparison for debugging
            if is_person and 'reid' in features and target_reid is not None:
                if target_reid.shape[0] == features['reid'].shape[0]:
                    raw_cosine = ReIDMatcher._cosine_similarity(target_reid, features['reid'])
                    logger.debug(f"ID {result.track_id}: raw ReID cosine={raw_cosine:.3f}")
                else:
                    logger.debug(f"ID {result.track_id}: feature dim mismatch ({target_reid.shape[0]} vs {features['reid'].shape[0]})")
            
            # Log body color comparison
            if is_person and 'body_color' in features:
                target_body = self.target_appearance.get_body_color()
                if target_body is not None:
                    body_sim = ReIDMatcher._histogram_similarity(target_body, features['body_color'])
                    logger.debug(f"ID {result.track_id}: body color similarity={body_sim:.3f}")
            
            # Compute similarity with person-specific matching
            similarity = ReIDMatcher.compute_similarity(
                self.target_appearance,
                features,
                result.bbox,
                current_time,
                is_person=is_person
            )
            
            candidate_scores.append((result, similarity, features))
        
        if not candidate_scores:
            logger.debug("No candidates with valid features")
            return None
        
        # Sort by similarity (highest first)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Log all candidates for debugging - CHANGE TO INFO FOR TROUBLESHOOTING
        logger.info(f"ReID candidates (threshold={self.reid_threshold}): {[(r.track_id, f'{s:.3f}') for r, s, _ in candidate_scores]}")
        
        best_match, best_similarity, best_features = candidate_scores[0]
        
        # Check if best similarity exceeds threshold
        if best_similarity <= self.reid_threshold:
            logger.info(f"ReID FAILED: Best similarity {best_similarity:.3f} <= threshold {self.reid_threshold} "
                       f"(best candidate: ID {best_match.track_id})")
            return None
        
        # If multiple candidates, check margin between best and second-best
        if len(candidate_scores) > 1:
            second_best_match, second_best_similarity, _ = candidate_scores[1]
            margin = best_similarity - second_best_similarity
            
            # Log the comparison
            logger.info(f"ReID margin check: Best ID {best_match.track_id} ({best_similarity:.3f}), "
                        f"Second ID {second_best_match.track_id} ({second_best_similarity:.3f}), "
                        f"Margin: {margin:.3f} (required: {ReIDMatcher.REID_MARGIN})")
            
            # If margin is too small, use spatial continuity as tiebreaker
            if margin < ReIDMatcher.REID_MARGIN:
                # Try spatial continuity - prefer the candidate closer to last known position
                if self.last_known_center is not None:
                    def get_distance_to_last(result: TrackingResult) -> float:
                        x1, y1, x2, y2 = result.bbox
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        lx, ly = self.last_known_center
                        return ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                    
                    dist_best = get_distance_to_last(best_match)
                    dist_second = get_distance_to_last(second_best_match)
                    
                    logger.info(f"Using spatial continuity: Best ID {best_match.track_id} dist={dist_best:.1f}px, "
                               f"Second ID {second_best_match.track_id} dist={dist_second:.1f}px")
                    
                    # If second-best is significantly closer (>50px advantage), prefer it
                    # unless best has appearance advantage
                    spatial_threshold = 100.0  # pixels
                    if dist_second < dist_best - spatial_threshold and second_best_similarity > self.reid_threshold:
                        # Second candidate is much closer to last position - might be an occlusion case
                        logger.info(f"Spatial tiebreaker: preferring closer ID {second_best_match.track_id} "
                                   f"(dist={dist_second:.1f}px vs {dist_best:.1f}px)")
                        best_match = second_best_match
                        best_similarity = second_best_similarity
                        best_features = candidate_scores[1][2]
                    elif dist_best < dist_second - spatial_threshold:
                        # Best candidate is much closer - trust it despite small margin
                        logger.info(f"Spatial confirmation: ID {best_match.track_id} is closer "
                                   f"(dist={dist_best:.1f}px vs {dist_second:.1f}px), accepting despite small margin")
                    else:
                        # Both are similar distance - truly ambiguous
                        logger.info(f"ReID FAILED: margin {margin:.3f} < {ReIDMatcher.REID_MARGIN}, "
                                    f"and spatial positions similar (ambiguous between ID {best_match.track_id} and ID {second_best_match.track_id})")
                        return None
                else:
                    # No spatial info available - reject ambiguous match
                    logger.info(f"ReID FAILED: margin {margin:.3f} < {ReIDMatcher.REID_MARGIN}, "
                                f"ambiguous between ID {best_match.track_id} and ID {second_best_match.track_id}")
                    return None
        
        # Check distinctiveness against other known persons in the registry
        # BUT ONLY if there are multiple candidates in the current frame
        # If there's only one person visible, we should trust the similarity score alone
        # because the registry can get contaminated with target features during ID bounces
        if is_person and self.original_track_id is not None and len(candidate_scores) > 1:
            # Use the features we already extracted
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
                    logger.info(f"ReID FAILED: candidate ID {best_match.track_id} not distinctive enough from other known persons")
                    return None
        elif is_person and len(candidate_scores) == 1:
            # When only one person is visible, be cautious but allow pose variation
            # From logs: Same person with pose change ~0.75-0.82, different person ~0.65-0.77
            # Setting threshold at 0.72 to allow pose variation while filtering most wrong matches
            SINGLE_PERSON_THRESHOLD = 0.72  # Allow pose variation
            if best_similarity < SINGLE_PERSON_THRESHOLD:
                logger.info(f"ReID FAILED: only one person visible but similarity {best_similarity:.3f} < {SINGLE_PERSON_THRESHOLD} "
                           f"(requires high confidence when no comparison is possible)")
                return None
            logger.info(f"Single person mode: similarity {best_similarity:.3f} >= {SINGLE_PERSON_THRESHOLD}, accepting")
        
        # Final sanity check - make sure we're returning a person when tracking person
        if target_class_id == 0 and best_match.class_id != 0:
            logger.warning(f"Rejecting match: expected person but got class {best_match.class_name} (id={best_match.class_id})")
            return None
        
        logger.info(f"Re-identified target (class='{best_match.class_name}') as ID {best_match.track_id} with similarity {best_similarity:.3f}")
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
    
    def _detect_occlusion(
        self,
        target_result: TrackingResult,
        all_results: List[TrackingResult]
    ) -> Tuple[bool, Optional[TrackingResult]]:
        """
        Detect if the target is being occluded by another person.
        
        Args:
            target_result: The current tracking result for our target
            all_results: All detection results in the current frame
            
        Returns:
            Tuple of (is_occluded, occluder_result)
        """
        if target_result is None:
            return False, None
        
        target_bbox = target_result.bbox
        target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
        
        for result in all_results:
            # Skip self
            if result.track_id == target_result.track_id:
                continue
            
            # Only consider person detections as potential occluders
            if result.class_id != 0:
                continue
            
            # Calculate IoU
            iou = self._calculate_iou(target_bbox, result.bbox)
            
            if iou > self.occlusion_iou_threshold:
                # Check if the occluder is in front (larger bounding box or closer to camera)
                # Heuristic: person with larger bbox is likely closer to camera
                occluder_area = (result.bbox[2] - result.bbox[0]) * (result.bbox[3] - result.bbox[1])
                
                # Also check overlap ratio relative to target's area
                x1 = max(target_bbox[0], result.bbox[0])
                y1 = max(target_bbox[1], result.bbox[1])
                x2 = min(target_bbox[2], result.bbox[2])
                y2 = min(target_bbox[3], result.bbox[3])
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                overlap_ratio = intersection / target_area if target_area > 0 else 0
                
                # If significant overlap (>40% of target covered), consider it occlusion
                if overlap_ratio > 0.4 or (occluder_area > target_area * 1.2 and iou > 0.35):
                    logger.info(f"Occlusion detected: Person ID {result.track_id} overlapping target "
                               f"(IoU={iou:.2f}, overlap_ratio={overlap_ratio:.2f}, "
                               f"occluder_area={occluder_area}, target_area={target_area})")
                    return True, result
        
        return False, None
    
    def _save_pre_occlusion_state(self):
        """Save the target appearance before occlusion for later verification."""
        if self.target_appearance is not None:
            # Deep copy the appearance
            self.pre_occlusion_appearance = copy.deepcopy(self.target_appearance)
            logger.info("Saved pre-occlusion appearance for later verification")
    
    def _verify_post_occlusion(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        current_time: float
    ) -> bool:
        """
        Verify that the tracked person after occlusion is still the same target.
        
        Args:
            frame: Current frame
            result: The detection result to verify
            current_time: Current timestamp
            
        Returns:
            True if the person matches pre-occlusion appearance, False otherwise
        """
        if self.pre_occlusion_appearance is None:
            return True  # No pre-occlusion appearance saved, assume OK
        
        if self.appearance_extractor is None:
            return True
        
        features = self.appearance_extractor.extract_features(frame, result.bbox, result.mask, class_id=0)
        if not features:
            return True  # Can't extract features, assume OK
        
        # Compare with pre-occlusion appearance
        similarity = ReIDMatcher.compute_similarity(
            self.pre_occlusion_appearance, features, result.bbox, current_time, is_person=True
        )
        
        # Use stricter threshold for post-occlusion verification
        POST_OCCLUSION_VERIFY_THRESHOLD = 0.50
        
        if similarity < POST_OCCLUSION_VERIFY_THRESHOLD:
            logger.warning(f"Post-occlusion verification FAILED: similarity={similarity:.3f} < {POST_OCCLUSION_VERIFY_THRESHOLD}. "
                          f"Track ID {result.track_id} may have switched to occluder!")
            return False
        
        logger.info(f"Post-occlusion verification PASSED: similarity={similarity:.3f}")
        return True
    
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
        
        # Store results for debug visualization
        self.last_results = results if results else []
        
        # Log detection summary for debugging
        if results:
            person_results = [r for r in results if r.class_id == 0]
            person_ids = [r.track_id for r in person_results]
            # Log more frequently to debug ID bouncing
            logger.debug(f"Frame detections: {len(person_results)} persons with IDs {person_ids}. Looking for target YOLO ID: {self.target_track_id}")
        
        # Stage 1: Try to find target by track ID
        # IMPORTANT: We verify appearance to catch ID switches (when ByteTrack assigns
        # our target's ID to a different person, e.g., when someone walks in front)
        # Also check for pending ReID match ID to handle confirmation period
        target_ids_to_check = [self.target_track_id]
        if self.pending_reid_match is not None:
            pending_id, _ = self.pending_reid_match
            if pending_id not in target_ids_to_check:
                target_ids_to_check.append(pending_id)
        
        for result in results:
            if result.track_id in target_ids_to_check:
                # Verify class hasn't changed (prevent tracking a non-person object)
                if self.target_class_id is not None and result.class_id != self.target_class_id:
                    logger.warning(f"Track ID {result.track_id} class changed from {self.target_class_name} to {result.class_name}, ignoring")
                    continue
                
                # CRITICAL: Check for occlusion by another person
                # If another person significantly overlaps our target, we need to be extra careful
                is_occluded, occluder = self._detect_occlusion(result, results)
                
                if is_occluded and self.enable_reid and result.class_id == 0:
                    current_time = time.time()
                    
                    if not self.is_occluded:
                        # Occlusion just started - save pre-occlusion appearance
                        self._save_pre_occlusion_state()
                        self.is_occluded = True
                        self.occlusion_start_time = current_time
                        logger.warning(f"Occlusion started! Target ID {result.track_id} being occluded by ID {occluder.track_id}")
                    
                    # During occlusion, be VERY strict about appearance matching
                    # The track ID might have been "stolen" by the occluder
                    features = self.appearance_extractor.extract_features(frame, result.bbox, result.mask, class_id=0)
                    if features:
                        similarity = ReIDMatcher.compute_similarity(
                            self.target_appearance, features, result.bbox, current_time, is_person=True
                        )
                        
                        # Very strict threshold during occlusion
                        OCCLUSION_SIMILARITY_THRESHOLD = 0.60
                        
                        if similarity < OCCLUSION_SIMILARITY_THRESHOLD:
                            # Track likely transferred to occluder - don't follow!
                            logger.warning(f"Track ID {result.track_id} appearance degraded during occlusion "
                                          f"(similarity={similarity:.3f} < {OCCLUSION_SIMILARITY_THRESHOLD}). "
                                          f"Treating as lost - will re-identify after occlusion.")
                            # Skip this match, fall through to lost/ReID handling
                            continue
                        else:
                            # Still looks like target despite occlusion
                            logger.info(f"Target appears occluded but still identifiable (similarity={similarity:.3f})")
                            # Don't update appearance during occlusion!
                            self.state = TrackerState.TRACKING
                            self.frames_lost = 0
                            return self._with_original_id(result)
                
                # If we were occluded but now we're not, verify this is still the target
                if self.is_occluded and not is_occluded:
                    current_time = time.time()
                    self.frames_since_occlusion_ended += 1
                    
                    if self.frames_since_occlusion_ended <= self.occlusion_recovery_frames:
                        # Still in recovery period - verify with pre-occlusion appearance
                        if not self._verify_post_occlusion(frame, result, current_time):
                            # Failed verification - track was stolen by occluder!
                            logger.warning(f"Post-occlusion verification failed! Track ID {result.track_id} "
                                          f"likely switched to occluder. Will re-identify.")
                            # Force ReID by not accepting this match
                            continue
                        else:
                            # Passed verification - but keep checking for a few more frames
                            logger.info(f"Post-occlusion frame {self.frames_since_occlusion_ended}/{self.occlusion_recovery_frames}")
                    else:
                        # Recovery period over - clear occlusion state
                        logger.info("Occlusion recovery complete - resuming normal tracking")
                        self.is_occluded = False
                        self.occlusion_start_time = None
                        self.pre_occlusion_appearance = None
                        self.frames_since_occlusion_ended = 0
                
                # CRITICAL: Verify appearance to detect ID switches
                # ByteTrack can assign our target's ID to a different person during occlusion
                if self.enable_reid and self.target_appearance is not None and result.class_id == 0:
                    features = self.appearance_extractor.extract_features(frame, result.bbox, result.mask, class_id=0)
                    if features:
                        current_time = time.time()
                        similarity = ReIDMatcher.compute_similarity(
                            self.target_appearance, features, result.bbox, current_time, is_person=True
                        )
                        
                        # Threshold for detecting ID switch (person changed completely)
                        ID_SWITCH_DETECTION_THRESHOLD = 0.35
                        # Threshold for updating appearance (only update when very confident)
                        APPEARANCE_UPDATE_THRESHOLD = 0.55
                        
                        if similarity < ID_SWITCH_DETECTION_THRESHOLD:
                            # ID switch detected! This track ID now belongs to a different person
                            logger.warning(f"ID switch detected! Track ID {result.track_id} appearance changed "
                                          f"(similarity={similarity:.3f} < {ID_SWITCH_DETECTION_THRESHOLD}). "
                                          f"Likely a different person. Will try ReID.")
                            # Don't accept this match - fall through to ReID stage
                            continue
                        
                        # Check if this is the pending ReID ID (continue confirmation)
                        is_pending_id = (self.pending_reid_match is not None and 
                                        result.track_id == self.pending_reid_match[0])
                        is_confirmed_target = (result.track_id == self.target_track_id)
                        
                        if is_pending_id and not is_confirmed_target:
                            # Continue confirmation for pending ID
                            self.consecutive_reid_frames += 1
                            logger.debug(f"Pending ID {result.track_id} confirmed in Stage 1 "
                                        f"({self.consecutive_reid_frames}/{self.reid_confirmation_frames} frames)")
                            
                            if self.consecutive_reid_frames >= self.reid_confirmation_frames:
                                # Fully confirmed! Update target track ID
                                old_id = self.target_track_id
                                self.target_track_id = result.track_id
                                self.pending_reid_match = None
                                self.consecutive_reid_frames = 0
                                
                                # Clear other persons registry to prevent contamination buildup
                                self.person_registry.clear()
                                if self.original_track_id is not None:
                                    self.person_registry.register_person(self.original_track_id, self.target_appearance)
                                
                                logger.info(f"ReID confirmed via Stage 1: YOLO ID {old_id} -> {self.target_track_id}")
                            
                            # Return result (with original display ID)
                            self.state = TrackerState.TRACKING
                            self.frames_lost = 0
                            return self._with_original_id(result)
                        
                        # Track is the confirmed target - accept it
                        self.state = TrackerState.TRACKING
                        self.frames_lost = 0
                        self.pending_reid_match = None
                        self.consecutive_reid_frames = 0
                        
                        # Reset occlusion state if we're cleanly tracking
                        if not is_occluded and self.is_occluded:
                            logger.info("Occlusion cleared - resuming normal tracking")
                            self.is_occluded = False
                            self.occlusion_start_time = None
                            self.pre_occlusion_appearance = None
                            self.frames_since_occlusion_ended = 0
                        
                        # Update class info if needed
                        if self.target_class_id != result.class_id:
                            self.target_class_id = result.class_id
                            self.target_class_name = result.class_name
                        
                        # ONLY update appearance when similarity is high enough
                        # This prevents contamination during occlusion
                        if similarity >= APPEARANCE_UPDATE_THRESHOLD:
                            self._update_appearance(frame, result)
                            logger.debug(f"Track ID {result.track_id} appearance updated (similarity={similarity:.3f})")
                        else:
                            logger.debug(f"Track ID {result.track_id} verified but appearance NOT updated "
                                        f"(similarity={similarity:.3f} < {APPEARANCE_UPDATE_THRESHOLD})")
                        
                        return self._with_original_id(result)
                
                # Non-person or no ReID - just accept track ID match
                self.state = TrackerState.TRACKING
                self.frames_lost = 0
                
                # Reset pending ReID match since we found our target by ID
                self.pending_reid_match = None
                self.consecutive_reid_frames = 0
                
                # Update class info if needed (in case we switched targets)
                if self.target_class_id != result.class_id:
                    self.target_class_id = result.class_id
                    self.target_class_name = result.class_name
                
                # Update appearance model for non-persons
                if self.enable_reid and result.class_id != 0:
                    self._update_appearance(frame, result)
                
                # Return result with consistent original track ID
                return self._with_original_id(result)
        
        # Stage 1 didn't find target - log why
        if results:
            person_results = [r for r in results if r.class_id == 0]
            person_ids = [r.track_id for r in person_results]
            valid_person_ids = [r.track_id for r in person_results if r.track_id >= 0]
            
            if self.target_track_id in person_ids:
                logger.info(f"Stage 1: Target ID {self.target_track_id} found but REJECTED (appearance mismatch or class change)")
            else:
                logger.info(f"Stage 1: Target ID {self.target_track_id} NOT in person detections. "
                           f"Found {len(person_results)} persons with IDs: {person_ids} (valid: {valid_person_ids})")
        else:
            logger.info(f"Stage 1: No detections at all")
        
        # Stage 2: Track ID not found, try re-identification
        # Note: frames_lost is incremented AFTER we try ReID, so that a successful
        # pending match doesn't cause "lost" to be reported
        
        if self.enable_reid and self.frames_lost <= self.max_frames_lost:
            self.state = TrackerState.REIDENTIFYING
            
            # Try to re-identify using appearance
            # Register other visible persons for distinctiveness checking (only during ReID)
            # This is expensive so only do it when we need it
            if len(results) > 1:
                self._register_other_persons(frame, results)
            
            reid_match = self._find_best_match_reid(frame, results)
            
            # Log ReID result
            if reid_match is None:
                logger.info(f"Stage 2 ReID: No match found (frames_lost={self.frames_lost})")
            else:
                logger.info(f"Stage 2 ReID: Found match ID {reid_match.track_id}")
            
            if reid_match is not None:
                current_time = time.time()
                new_yolo_id = reid_match.track_id
                
                # Safety check: never accept invalid track IDs
                if new_yolo_id < 0:
                    logger.warning(f"ReID returned invalid track ID {new_yolo_id}, ignoring")
                    reid_match = None
            
            if reid_match is not None:
                # Get the similarity score from the match
                features = self.appearance_extractor.extract_features(
                    frame, reid_match.bbox, reid_match.mask, class_id=reid_match.class_id
                )
                if features:
                    match_similarity = ReIDMatcher.compute_similarity(
                        self.target_appearance, features, reid_match.bbox, current_time, is_person=True
                    )
                else:
                    match_similarity = 0.5  # Default if features not available
                
                # ALWAYS require confirmation frames to prevent rapid ID bouncing
                # This is critical when multiple persons have similar appearances
                
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
                                
                                # Clear other persons registry to prevent contamination buildup
                                # Other person appearances may have been contaminated with target features
                                self.person_registry.clear()
                                # Re-register the target
                                if self.original_track_id is not None:
                                    self.person_registry.register_person(self.original_track_id, self.target_appearance)
                                
                                logger.info(f"Confirmed ReID: YOLO ID {old_yolo_id} -> {self.target_track_id} "
                                           f"(confirmed over {self.reid_confirmation_frames} frames)")
                                return self._with_original_id(reid_match)
                            else:
                                logger.debug(f"ReID cooldown: {time_since_last_switch:.2f}s < {self.reid_switch_cooldown}s")
                        else:
                            logger.debug(f"ReID pending confirmation: {self.consecutive_reid_frames}/{self.reid_confirmation_frames} frames")
                    else:
                        # Different ID - but don't reset if similarity is decent
                        # This handles YOLO ID bouncing
                        logger.debug(f"ReID candidate changed from {pending_id} to {new_yolo_id}")
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
                # No match found - increment frames lost
                # But DON'T reset pending_reid_match immediately - the target might just be
                # temporarily undetected for one frame
                self.frames_lost += 1
                
                # Only reset pending match after several consecutive failures
                # This prevents losing progress due to momentary detection failures
                if self.frames_lost > 3:
                    if self.pending_reid_match is not None:
                        logger.debug(f"Resetting pending ReID match after {self.frames_lost} failed frames")
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
