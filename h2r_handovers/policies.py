"""Grasp selection policies.

Each policy is a callable: (poses: list[Pose], frame_id: str) -> int
It receives the list of candidate grasp poses (already sorted by GraspNet
score descending) and returns the *index* of the chosen grasp.

To add a new policy, define a function and register it in POLICIES.
"""


import rclpy

def highest_score(poses, frame_id, tf_buffer=None):
    """Pick the top-scored grasp (index 0 — GraspNet sorts by score)."""
    return 0


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
    'nearest_depth': nearest_depth,
    'top_k_aligned': top_k_aligned,
    'ergonomic': ergonomic,
}
