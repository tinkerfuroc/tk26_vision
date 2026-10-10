"""Persist RGB observations through the existing shared VisionLogger."""
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid


def log_scene(logger, image, observations, *, task, capture_header, status,
              error='', model_info=None, annotated_image=None):
    """Return the JSON path; report missing images/writes as logging failures.

    Unique per-request branches avoid millisecond filename collisions.
    No-image errors have JSON only; inference errors also preserve the input.
    """
    branch = task + '_' + uuid.uuid4().hex
    metadata = {
        'task': task, 'status': int(status), 'error_msg': error,
        'detected': bool(observations), 'capture_header': capture_header,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'model': model_info or {}, 'bbox_format': 'pixel_xyxy_exclusive',
    }
    detections = [dict(bbox=list(o.bbox), labels=list(o.labels),
                       cls_name=','.join(o.labels), description=o.description)
                  for o in observations]
    if image is None:
        path = Path(logger.aux_path('no_image', 'req', 'json', branch=branch))
        path.write_text(json.dumps(dict(metadata, detections=[]), indent=2),
                        encoding='utf-8')
        return str(path.resolve())
    stamp = logger.write(image, detections, branch=branch, extras=metadata)
    if stamp is None:
        raise OSError('VisionLogger did not write artifacts')
    paths = [Path(logger.aux_path(stamp, kind, ext, branch=branch))
             for kind, ext in [('orig', 'jpg'), ('overlay', 'jpg'), ('req', 'json')]]
    if annotated_image is not None:
        import cv2
        # Persist the same annotation pixels as the service returns (JPEG on disk).
        if not cv2.imwrite(str(paths[1]), annotated_image):
            raise OSError('Could not save returned annotated image')
    if any(not p.is_file() or p.stat().st_size == 0 for p in paths):
        raise OSError('Missing/empty vision_log artifact')
    return str(paths[-1].resolve())
