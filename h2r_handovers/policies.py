"""Grasp selection policies.

Each policy is a callable: (poses: list[Pose], frame_id: str) -> int
It receives the list of candidate grasp poses (already sorted by GraspNet
score descending) and returns the *index* of the chosen grasp.

To add a new policy, define a function and register it in POLICIES.
"""


import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

BASE_FRAME = 'panda_link0'


def highest_score(poses, frame_id, tf_buffer=None):
    """Pick the top-scored grasp (index 0 — GraspNet sorts by score)."""
    return 0


def top_down(poses, frame_id, tf_buffer=None, max_tilt_deg=45.0):
    """Prefer grasps that approach from above, keeping the robot upright.

    Walks the candidates in GraspNet-score order and returns the first whose
    approach axis (pose Z) is within *max_tilt_deg* of straight down in the
    robot base frame. If none qualifies, falls back to the most downward-
    pointing candidate, so a sideways-only presentation still yields a grasp.
    """
    if tf_buffer is None:
        print('No TF buffer provided to top_down!')
        return 0
    try:
        t = tf_buffer.lookup_transform(
            BASE_FRAME, frame_id, rclpy.time.Time(),
            rclpy.duration.Duration(seconds=1.0))
    except Exception as e:
        print(f'TF lookup failed in top_down policy: {e}')
        return 0
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
            return i
    best = int(np.argmax(downwardness))
    print(f'top_down: no grasp within {max_tilt_deg:.0f} deg of vertical — '
          f'falling back to candidate {best} '
          f'(tilt {np.degrees(np.arccos(np.clip(downwardness[best], -1, 1))):.0f} deg).')
    return best


def nearest_depth(poses, frame_id, tf_buffer=None):
    """Pick the grasp closest to the camera (smallest Z in camera frame).

    Closer grasps tend to be more reachable and have better depth accuracy.
    Only meaningful when poses are still in the camera optical frame.
    """
    return min(range(len(poses)), key=lambda i: poses[i].position.z)


def top_k_aligned(poses, frame_id, tf_buffer=None, k=50):
    """Pick the grasp that requires the least wrist rotation.
    
    Looks at the top K grasps (which have the highest GraspNet scores)
    and selects the one whose orientation requires the smallest rotational 
    movement from the robot's current hand orientation.
    """
    limit = min(k, len(poses))
    
    if tf_buffer is None:
        print("No TF buffer provided to top_k_aligned!")
        return 0
        
    try:
        # Lookup transform from hand to camera to get the camera's rotation in hand frame
        # Use a timeout to prevent LookupException if the tree is slightly delayed
        transform = tf_buffer.lookup_transform(
            'panda_hand', frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)
        )
        q_cam = transform.transform.rotation
    except Exception as e:
        print(f"TF lookup failed in policy: {e}")
        return 0
    
    def alignment_score(pose):
        q_g = pose.orientation
        # Compute the scalar component (w) of the quaternion multiplication: 
        # (q_cam * q_g) where q_cam is inverted. 
        # This gives the cos(theta/2) of the angle between them.
        w = q_cam.w * q_g.w - q_cam.x * q_g.x - q_cam.y * q_g.y - q_cam.z * q_g.z
        return abs(w)
        
    best_idx = max(range(limit), key=lambda i: alignment_score(poses[i]))
    return best_idx


def ergonomic(poses, frame_id, tf_buffer=None):
    """Pick the grasp that is most comfortable for the human.

    TODO — planned implementation:
    Use the EgoHOS segmentation mask + depth to determine *where* the human
    is gripping the object (the hand-contact region).  Then select the grasp
    candidate whose approach direction is furthest from that region, so the
    robot grips the "free" end and the human can release naturally.

    Inputs needed (will be added as a second subscriber or shared state):
    - segmentation_mask (to locate the hand-contact region on the object)
    - aligned depth (to get 3-D positions of contact pixels)

    For now, falls back to highest_score.
    """
    return highest_score(poses, frame_id, tf_buffer)


# Map parameter strings to the corresponding policy functions
POLICIES = {
    'highest_score': highest_score,
    'top_down': top_down,
    'nearest_depth': nearest_depth,
    'top_k_aligned': top_k_aligned,
    'ergonomic': ergonomic,
}
