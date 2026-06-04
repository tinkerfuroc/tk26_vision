import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from ..core.tracking_types import TargetAppearance

logger = logging.getLogger(__name__)

class PersonReIDModel:
    """
    Person Re-Identification model wrapping a pluggable deep backbone.

    The deep term is a genuinely pretrained OSNet (via torchreid), exposed behind
    a stable ``extract_features(crop) -> L2-normalized np.ndarray`` interface so
    reid_search / appearance_manager are untouched.

    This replaces the legacy random-head path (a ResNet50 with untrained
    channel_attention / bottleneck / part_bottlenecks modules — a random
    projection of ImageNet features with no ReID checkpoint), which was the #1
    root cause of wrong identity locks.

    Weight strategy: the default backbone (osnet_ain_x1_0) is imagenet-init;
    loading a Market/MSMT-trained checkpoint via ``reid_weights_path`` is the
    recommended upgrade for maximal lookalike discrimination (config-only).
    """

    def __init__(
        self,
        device: str = "cpu",
        backbone_name: str = "osnet_ain_x1_0",
        reid_weights_path: str = "",
    ):
        """
        Initialize the Person ReID model.

        Args:
            device: Device for computation.
            backbone_name: OSNet variant ('osnet_ain_x1_0' default,
                'osnet_x0_25' alt).
            reid_weights_path: optional ReID-trained checkpoint overriding the
                imagenet init; empty ⇒ keep imagenet.
        """
        self.device = device
        self.backbone_name = backbone_name
        from .reid_backbone import build_reid_backbone
        self.backbone = build_reid_backbone(
            backbone_name, device=device, reid_weights_path=reid_weights_path
        )
        self.feature_dim = self.backbone.feature_dim
        # True iff a real deep backbone is available. Lets call sites (and the
        # batch path) cheaply short-circuit to zero vectors when it isn't.
        self.use_deep_features = self.backbone is not None

    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract an L2-normalized ReID embedding from a person crop (RGB)."""
        return self.backbone.extract_features(crop)

    @staticmethod
    def _stack_crops(crops: list) -> "torch.Tensor":
        """Resize + ImageNet-normalize K crops into one [K,3,256,128] CPU tensor.

        Numerically identical preprocessing to the OSNet backbone's per-crop
        ``extract_features`` (resize to (W,H)=(128,256), /255, ImageNet
        normalize, permute to CHW). Empty list -> a real [0,3,256,128] tensor so
        downstream stacking/forward is well-defined.
        """
        from .reid_backbone import (
            _IMAGENET_MEAN,
            _IMAGENET_STD,
            _REID_H,
            _REID_W,
        )
        if not crops:
            return torch.zeros((0, 3, _REID_H, _REID_W), dtype=torch.float32)
        batch = np.empty((len(crops), _REID_H, _REID_W, 3), dtype=np.float32)
        for i, crop in enumerate(crops):
            resized = cv2.resize(crop, (_REID_W, _REID_H))
            batch[i] = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        return tensor

    def extract_features_batch(self, crops: list) -> np.ndarray:
        """Embed K crops in ONE forward pass. Row i == extract_features(crops[i]).

        Returns [K, feature_dim] float32, each row L2-normalized. Empty -> [0, dim].

        Degenerate crops (None / empty / <2px on a side) are handled exactly as
        the per-crop path: their row is a zero vector and they do NOT consume a
        deep forward slot, so the batched output stays row-equivalent to looping
        ``extract_features``.
        """
        if not self.use_deep_features or self.backbone is None:
            return np.zeros((len(crops), self.feature_dim), dtype=np.float32)
        if not crops:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        out = np.zeros((len(crops), self.feature_dim), dtype=np.float32)
        valid_idx = []
        valid_crops = []
        for i, crop in enumerate(crops):
            if (
                crop is None
                or crop.size == 0
                or crop.shape[0] < 2
                or crop.shape[1] < 2
            ):
                continue  # leave zero row (mirrors OSNetBackbone.extract_features)
            valid_idx.append(i)
            valid_crops.append(crop)

        if not valid_crops:
            return out

        tensor = self._stack_crops(valid_crops).to(self.device)
        with torch.no_grad():
            feats = self.backbone.model(tensor)            # [P, feature_dim]
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        feats = feats.cpu().numpy().astype(np.float32)
        for slot, i in enumerate(valid_idx):
            out[i] = feats[slot]
        return out


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
    
    def __init__(
        self,
        device: str = "cpu",
        reid_backbone: str = "osnet_ain_x1_0",
        reid_weights_path: str = "",
    ):
        """
        Initialize the appearance extractor.

        Args:
            device: Device to use for computation.
            reid_backbone: OSNet variant for the person ReID deep term.
            reid_weights_path: optional ReID-trained checkpoint overriding the
                imagenet init.
        """
        self.device = device

        # Person-specific ReID model
        self.person_reid = PersonReIDModel(
            device, backbone_name=reid_backbone, reid_weights_path=reid_weights_path
        )
        
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
            
            # Upper/lower color signatures (coarse but stable for clothing)
            h_mid = y1 + (y2 - y1) // 2
            upper_crop = frame[y1:h_mid, x1:x2]
            lower_crop = frame[h_mid:y2, x1:x2]
            upper_mask = mask[y1:h_mid, x1:x2] if mask is not None else None
            lower_mask = mask[h_mid:y2, x1:x2] if mask is not None else None
            features['upper_color'] = self._extract_color_histogram(upper_crop, upper_mask) if upper_crop.size > 0 else np.zeros(32, dtype=np.float32)
            features['lower_color'] = self._extract_color_histogram(lower_crop, lower_mask) if lower_crop.size > 0 else np.zeros(32, dtype=np.float32)
        else:
            # General CNN features for non-person objects
            if self.use_general_cnn and self.general_model is not None:
                features['cnn'] = self._extract_general_features(crop)
        
        # Color histogram (for all objects)
        features['color_hist'] = self._extract_color_histogram(masked_crop, mask_crop)
        
        # Size features
        features['size'] = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        # Shape cues: aspect ratio and mask coverage help separate similar-stature people
        aspect_ratio = (x2 - x1) / max((y2 - y1), 1e-6)
        features['aspect_ratio'] = np.array([aspect_ratio], dtype=np.float32)
        if mask_crop is not None and mask_crop.size > 0:
            area_ratio = float(np.sum(mask_crop) / max(mask_crop.size, 1))
            features['mask_coverage'] = np.array([area_ratio], dtype=np.float32)

        return features

    def extract_features_batch(self, frame, bboxes, masks, class_ids) -> list:
        """Vectorize the deep ReID forward across N detections.

        Returns list[dict] aligned to ``bboxes``; each dict == ``extract_features``
        for that detection. The per-detection color/size dicts are built exactly
        as the single-crop path (byte-identical), and ONLY the ``'reid'`` deep
        vector is recomputed via one batched forward — collapsing K deep passes
        into one while preserving full row-equivalence. Non-person / invalid-bbox
        entries are left as the per-crop path produced them (no batch slot used).
        """
        n = len(bboxes)
        masks = masks if masks is not None else [None] * n
        class_ids = class_ids if class_ids is not None else [-1] * n

        # Build per-detection dicts (incl. color/size) with the existing per-crop
        # path, and collect the person crops that need a deep embedding.
        out = [None] * n
        person_idx = []
        person_crops = []
        for i in range(n):
            d = self.extract_features(frame, bboxes[i], masks[i], class_ids[i])
            out[i] = d
            if class_ids[i] == self.PERSON_CLASS_ID and d:
                # Recompute the same clamped crop used inside extract_features
                # (cheap numpy slice) so the batch forward sees identical pixels.
                x1, y1, x2, y2 = bboxes[i]
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    person_idx.append(i)
                    person_crops.append(frame[y1:y2, x1:x2].copy())

        if person_crops:
            deep = self.person_reid.extract_features_batch(person_crops)  # [P, dim]
            for slot, i in enumerate(person_idx):
                if out[i] is not None and "reid" in out[i]:
                    out[i]["reid"] = deep[slot]
        return out

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
    
    # Weights for PERSON re-identification.
    # Phase 1: the deep term is now a genuinely pretrained OSNet (not the old
    # untrained random head), so it dominates; color is demoted to a backup cue.
    WEIGHT_REID = 0.75        # Person ReID deep features (trained OSNet)
    WEIGHT_BODY_COLOR = 0.13  # Body part colors (clothing backup)
    WEIGHT_COLOR = 0.05       # General color histogram
    WEIGHT_UPPER = 0.04       # Upper-body color signature
    WEIGHT_LOWER = 0.03       # Lower-body color signature
    WEIGHT_SIZE = 0.0         # De-emphasize size/shape (unreliable across people)
    
    # Weights for NON-PERSON objects
    WEIGHT_CNN_GENERAL = 0.50
    WEIGHT_COLOR_GENERAL = 0.40
    WEIGHT_SIZE_GENERAL = 0.10
    
    # Thresholds — recalibrated for the trained OSNet operating point.
    # OSNet's combined-score distribution shifts vs the old random head, so the
    # combined floor is lowered. Starting points; final values via the offline
    # Occluded-REID ROC (Step 1.5/Step 5, an informing knob, not a CI gate) or
    # arena tuning. See person-tracker-benchmark-strategy.
    REID_THRESHOLD = 0.55     # Minimum COMBINED similarity for re-identification
    REID_MARGIN = 0.15        # Best match must be clearly better than second-best
    TIME_DECAY_FACTOR = 1.0   # NO time decay - features should be stable over time
    MAX_REID_TIME = 600.0     # 10 minutes max search time

    # Minimum RAW reid cosine similarity floor.
    # Trained OSNet's same/different cosine gap is wide, so the raw floor is
    # lowered from the legacy 0.60 to its same/different operating point
    # (retune in Step 5).
    MIN_REID_SIMILARITY_RAW = 0.40  # Raw cosine similarity floor (OSNet)

    # Color hard floors — now a BACKUP cue, not a gate. Relaxed so they no
    # longer hard-reject a true match on lighting/clothing variation; the
    # trained deep term carries the discrimination.
    MIN_BODY_COLOR_SIMILARITY = 0.40
    MIN_UPPER_SIMILARITY = 0.40
    MIN_LOWER_SIMILARITY = 0.40
    
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
        reid_anchor_sim = None
        body_color_sim = None
        body_color_anchor_sim = None
        color_sim = None
        color_anchor_sim = None
        upper_sim = None
        lower_sim = None
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
                    
                    # Anchor feature comparison (helps with drift against similar stature people)
                    if target.anchor_feature is not None and target.anchor_feature.shape[0] == candidate_reid.shape[0]:
                        reid_anchor_sim = cls._cosine_similarity(target.anchor_feature, candidate_reid)
                        reid_sim_raw = max(reid_sim_raw, reid_anchor_sim)
                    
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
                if target.anchor_body_color is not None:
                    body_color_anchor_sim = cls._histogram_intersection(target.anchor_body_color, candidate_features['body_color'])
                    body_color_sim = max(body_color_sim, body_color_anchor_sim)
                
                # Hard rejection: if clothing colors are too different
                if body_color_sim < cls.MIN_BODY_COLOR_SIMILARITY:
                    logger.info(f"Rejecting candidate: body color similarity {body_color_sim:.3f} < {cls.MIN_BODY_COLOR_SIMILARITY}")
                    return 0.0
        
        # Upper/lower specific clothing color to combat outfit ambiguity
        if 'upper_color' in candidate_features:
            target_upper = None
            if target.anchor_upper_color is not None:
                target_upper = target.anchor_upper_color
            elif target.upper_color_history:
                target_upper = target.upper_color_history[-1]
            if target_upper is not None:
                upper_sim = cls._histogram_intersection(target_upper, candidate_features['upper_color'])
                if upper_sim < cls.MIN_UPPER_SIMILARITY:
                    logger.info(f"Rejecting candidate: upper color similarity {upper_sim:.3f} < {cls.MIN_UPPER_SIMILARITY}")
                    return 0.0
        
        if 'lower_color' in candidate_features:
            target_lower = None
            if target.anchor_lower_color is not None:
                target_lower = target.anchor_lower_color
            elif target.lower_color_history:
                target_lower = target.lower_color_history[-1]
            if target_lower is not None:
                lower_sim = cls._histogram_intersection(target_lower, candidate_features['lower_color'])
                if lower_sim < cls.MIN_LOWER_SIMILARITY:
                    logger.info(f"Rejecting candidate: lower color similarity {lower_sim:.3f} < {cls.MIN_LOWER_SIMILARITY}")
                    return 0.0
        
        # 3. General color histogram
        if 'color_hist' in candidate_features:
            target_hist = target.get_average_color_hist()
            if target_hist is not None:
                color_sim = cls._histogram_intersection(target_hist, candidate_features['color_hist'])
                if target.anchor_color_hist is not None:
                    color_anchor_sim = cls._histogram_intersection(target.anchor_color_hist, candidate_features['color_hist'])
                    color_sim = max(color_sim, color_anchor_sim)
        
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
        
        if upper_sim is not None:
            scores.append(upper_sim ** 1.1)
            weights.append(cls.WEIGHT_UPPER)
        
        if lower_sim is not None:
            scores.append(lower_sim ** 1.1)
            weights.append(cls.WEIGHT_LOWER)
        
        if color_sim is not None:
            scores.append(color_sim)
            weights.append(cls.WEIGHT_COLOR)
        
        if size_sim is not None and cls.WEIGHT_SIZE > 0:
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
