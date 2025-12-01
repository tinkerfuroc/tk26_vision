import cv2
import numpy as np
from typing import List, Optional, Tuple

from ..core.tracking_types import TrackerState, TrackingResult

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
