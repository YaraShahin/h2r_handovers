"""Grasp selection policies."""

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

BASE_FRAME = 'panda_link0'


def _log(logger, message, warn=False):
    """Log via the node's ROS logger or print fallback."""
    if logger is None:
        print(message)
    elif warn:
        logger.warning(message)
    else:
        logger.info(message)


def highest_score(poses, frame_id, tf_buffer=None, logger=None):
    """Returns index 0, as GraspNet candidates are pre-sorted by score in descending order."""
    _log(logger, f'highest_score: selected candidate index 0 of {len(poses)} (top score).')
    return 0


def top_down(poses, frame_id, tf_buffer=None, max_tilt_deg=45.0, strict=True,
             logger=None):
    """Selects the highest-scored grasp whose approach axis (Z) is within max_tilt_deg of straight down.
    If strict=False, falls back to the most downward-pointing grasp if none qualify."""
    if tf_buffer is None:
        _log(logger, 'No TF buffer provided to top_down — '
             + ('rejecting all candidates!' if strict else 'falling back to top score.'),
             warn=True)
        return None if strict else 0
    try:
        t = tf_buffer.lookup_transform(
            BASE_FRAME, frame_id, rclpy.time.Time(),
            rclpy.duration.Duration(seconds=1.0))
    except Exception as e:
        _log(logger, f'TF lookup failed in top_down policy: {e} — '
             + ('rejecting all candidates.' if strict else 'falling back to top score.'),
             warn=True)
        return None if strict else 0
    q = t.transform.rotation
    rot_to_base = Rotation.from_quat([q.x, q.y, q.z, q.w])

    quats = np.array([
        [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        for p in poses])
    approaches = Rotation.from_quat(quats).as_matrix()[:, :, 2]  # pose Z axes
    downwardness = -rot_to_base.apply(approaches)[:, 2]  # 1.0 = straight down

    threshold = np.cos(np.radians(max_tilt_deg))
    for i, d in enumerate(downwardness):
        if d >= threshold:
            _log(logger, f'top_down: selected candidate index {i} of {len(poses)} '
                 f'(tilt {np.degrees(np.arccos(np.clip(d, -1, 1))):.0f} deg).')
            return i
    best = int(np.argmax(downwardness))
    tilt = np.degrees(np.arccos(np.clip(downwardness[best], -1, 1)))
    if strict:
        _log(logger, f'top_down: no grasp within {max_tilt_deg:.0f} deg of vertical — '
             f'rejecting all {len(poses)} candidates '
             f'(closest: candidate {best}, tilt {tilt:.0f} deg).', warn=True)
        return None
    _log(logger, f'top_down: no grasp within {max_tilt_deg:.0f} deg of vertical — '
         f'falling back to candidate {best} (tilt {tilt:.0f} deg).', warn=True)
    return best


def nearest_depth(poses, frame_id, tf_buffer=None, logger=None):
    """Pick grasp closest to the camera (smallest Z). Closer grasps are often more reachable/accurate."""
    idx = min(range(len(poses)), key=lambda i: poses[i].position.z)
    _log(logger, f'nearest_depth: selected candidate index {idx} of {len(poses)} '
         f'(depth {poses[idx].position.z:.3f} m).')
    return idx


def top_k_aligned(poses, frame_id, tf_buffer=None, k=50, logger=None):
    """Pick grasp from top K that requires the least wrist rotation from current hand orientation."""
    limit = min(k, len(poses))
    
    if tf_buffer is None:
        _log(logger, 'No TF buffer provided to top_k_aligned!', warn=True)
        return 0

    try:
        # Get camera's rotation in hand frame
        transform = tf_buffer.lookup_transform(
            'panda_hand', frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)
        )
        q_cam = transform.transform.rotation
    except Exception as e:
        _log(logger, f'TF lookup failed in policy: {e}', warn=True)
        return 0
    
    def alignment_score(pose):
        q_g = pose.orientation
        # Cosine of half-angle between quaternions via w component of product with inverted q_cam.
        w = q_cam.w * q_g.w - q_cam.x * q_g.x - q_cam.y * q_g.y - q_cam.z * q_g.z
        return abs(w)
        
    best_idx = max(range(limit), key=lambda i: alignment_score(poses[i]))
    _log(logger, f'top_k_aligned: selected candidate index {best_idx} of {len(poses)} '
         f'(best alignment among top {limit}).')
    return best_idx


def ergonomic(poses, frame_id, tf_buffer=None, hand_center=None, logger=None):
    """Select grasp that maximizes distance from hand centroid for human comfort/safety."""
    if hand_center is None:
        stri = 'ergonomic: no hand centre for this inference, falling back to highest_score.'
        return highest_score(poses, frame_id, tf_buffer, logger=logger), stri

    positions = np.array([[p.position.x, p.position.y, p.position.z] for p in poses])
    clearances = np.linalg.norm(positions - np.asarray(hand_center), axis=1)

    score_idx = 0
    clearance_idx = int(np.argmax(clearances))
    separation = float(np.linalg.norm(positions[score_idx] - positions[clearance_idx]))

    stri = (
        f'ergonomic: highest-score grasp index {score_idx} (hand clearance {clearances[score_idx]:.3f} m), '
        f'max-clearance grasp index {clearance_idx} of {len(poses)} (hand clearance {clearances[clearance_idx]:.3f} m), '
        f'distance between the two grasps: {separation:.3f} m.'
    )
    return clearance_idx, stri


POLICIES = {
    'highest_score': highest_score,
    'top_down': top_down,
    'nearest_depth': nearest_depth,
    'top_k_aligned': top_k_aligned,
    'ergonomic': ergonomic,
}
