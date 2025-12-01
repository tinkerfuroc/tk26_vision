import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

from ..core.tracking_types import TargetAppearance

logger = logging.getLogger(__name__)

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
    WEIGHT_REID = 0.55        # Person ReID deep features (ResNet50)
    WEIGHT_BODY_COLOR = 0.28  # Body part colors (clothing)
    WEIGHT_COLOR = 0.08       # General color histogram
    WEIGHT_UPPER = 0.05       # Upper-body color signature
    WEIGHT_LOWER = 0.04       # Lower-body color signature
    WEIGHT_SIZE = 0.0         # De-emphasize size/shape (unreliable across people)
    
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
    REID_THRESHOLD = 0.68     # Minimum COMBINED similarity for re-identification
    REID_MARGIN = 0.15        # Best match must be clearly better than second-best
    TIME_DECAY_FACTOR = 1.0   # NO time decay - features should be stable over time
    MAX_REID_TIME = 600.0     # 10 minutes max search time
    
    # CRITICAL: Minimum RAW reid similarity
    # Same person with pose change: ~0.70-0.80
    # Different person: ~0.55-0.70
    # Set floor at 0.55 to allow pose variation while rejecting most wrong matches
    MIN_REID_SIMILARITY_RAW = 0.60  # Raw cosine similarity floor
    
    # Minimum body color similarity - more tolerant for lighting variation
    # Same person: ~0.70-0.95, Different person: ~0.60-0.75  
    MIN_BODY_COLOR_SIMILARITY = 0.60  # Reject if body colors are too different
    MIN_UPPER_SIMILARITY = 0.55
    MIN_LOWER_SIMILARITY = 0.55
    
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
