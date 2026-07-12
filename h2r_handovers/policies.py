"""Grasp selection policies.

Each policy is a callable: (poses: list[Pose], frame_id: str) -> int | None
It receives the list of candidate grasp poses (already sorted by GraspNet
score descending) and returns the *index* of the chosen grasp, or None to
reject every candidate (nothing is published for this capture).

To add a new policy, define a function and register it in POLICIES.
"""


import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

BASE_FRAME = 'panda_link0'


def _log(logger, message, warn=False):
    """Log via the node's ROS logger when provided (visible under launch and
    in /rosout); falls back to print for bare/offline use.

    info and warn go through separate calls: rclpy caches the severity per
    source line, so one shared logger call raises 'Logger severity cannot
    be changed between calls' as soon as both severities are used.
    """
    if logger is None:
        print(message)
    elif warn:
        logger.error(message)
    else:
        logger.error(message)


def highest_score(poses, frame_id, tf_buffer=None, logger=None):
    """Pick the top-scored grasp (index 0 — GraspNet sorts by score)."""
    _log(logger, f'highest_score: selected candidate index 0 of {len(poses)} (top score).')
    return 0


def top_down(poses, frame_id, tf_buffer=None, max_tilt_deg=45.0, strict=True,
             logger=None):
    """Pick the best-scored grasp that approaches from above.

    Walks the candidates in GraspNet-score order and returns the first whose
    approach axis (pose Z) is within *max_tilt_deg* of straight down in the
    robot base frame.

    If none qualifies (or TF is unavailable, so tilt can't be checked):
    - strict=True: all candidates are REJECTED — returns None, no grasp is
      published and the orchestrator's capture simply times out.
    - strict=False: falls back to the most downward-pointing candidate
      (top-scored one on TF failure), so a sideways-only presentation still
      yields a grasp.
    """
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
    """Pick the grasp closest to the camera (smallest Z in camera frame).

    Closer grasps tend to be more reachable and have better depth accuracy.
    Only meaningful when poses are still in the camera optical frame.
    """
    idx = min(range(len(poses)), key=lambda i: poses[i].position.z)
    _log(logger, f'nearest_depth: selected candidate index {idx} of {len(poses)} '
         f'(depth {poses[idx].position.z:.3f} m).')
    return idx


def top_k_aligned(poses, frame_id, tf_buffer=None, k=50, logger=None):
    """Pick the grasp that requires the least wrist rotation.
    
    Looks at the top K grasps (which have the highest GraspNet scores)
    and selects the one whose orientation requires the smallest rotational 
    movement from the robot's current hand orientation.
    """
    limit = min(k, len(poses))
    
    if tf_buffer is None:
        _log(logger, 'No TF buffer provided to top_k_aligned!', warn=True)
        return 0

    try:
        # Lookup transform from hand to camera to get the camera's rotation in hand frame
        # Use a timeout to prevent LookupException if the tree is slightly delayed
        transform = tf_buffer.lookup_transform(
            'panda_hand', frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)
        )
        q_cam = transform.transform.rotation
    except Exception as e:
        _log(logger, f'TF lookup failed in policy: {e}', warn=True)
        return 0
    
    def alignment_score(pose):
        q_g = pose.orientation
        # Compute the scalar component (w) of the quaternion multiplication: 
        # (q_cam * q_g) where q_cam is inverted. 
        # This gives the cos(theta/2) of the angle between them.
        w = q_cam.w * q_g.w - q_cam.x * q_g.x - q_cam.y * q_g.y - q_cam.z * q_g.z
        return abs(w)
        
    best_idx = max(range(limit), key=lambda i: alignment_score(poses[i]))
    _log(logger, f'top_k_aligned: selected candidate index {best_idx} of {len(poses)} '
         f'(best alignment among top {limit}).')
    return best_idx


def ergonomic(poses, frame_id, tf_buffer=None, hand_center=None, logger=None):
    """Hybrid score / hand-clearance policy.

    Selects two candidates:
    1. the highest-scored grasp (index 0 — GraspNet sorts by score), and
    2. the grasp with the largest clearance from the human hand centre
       (*hand_center*, the 3-D hand centroid the GraspNet driver publishes
       on 'hand_center' in the same camera frame as the candidates),
    prints both together with the distance between them, and returns the
    max-clearance grasp so the robot grips the end away from the human.

    Falls back to highest_score when no hand centre is available for this
    inference (hand not visible, or driver without the hand_center topic).
    """
    if hand_center is None:
        stri = 'ergonomic: no hand centre for this inference — ' + 'falling back to highest_score.'
        return highest_score(poses, frame_id, tf_buffer, logger=logger), stri

    positions = np.array([[p.position.x, p.position.y, p.position.z] for p in poses])
    clearances = np.linalg.norm(positions - np.asarray(hand_center), axis=1)

    score_idx = 0
    clearance_idx = int(np.argmax(clearances))
    separation = float(np.linalg.norm(positions[score_idx] - positions[clearance_idx]))

    stri = f'ergonomic: highest-score grasp index {score_idx} ' + f'(hand clearance {clearances[score_idx]:.3f} m), ' + f'max-clearance grasp index {clearance_idx} of {len(poses)} ' +  f'(hand clearance {clearances[clearance_idx]:.3f} m), ' +  f'distance between the two grasps: {separation:.3f} m.'
    return clearance_idx, stri


# Map parameter strings to the corresponding policy functions
POLICIES = {
    'highest_score': highest_score,
    'top_down': top_down,
    'nearest_depth': nearest_depth,
    'top_k_aligned': top_k_aligned,
    'ergonomic': ergonomic,
}
