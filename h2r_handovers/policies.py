"""Grasp selection policies.

Each policy is a callable: (poses: list[Pose], frame_id: str) -> int
It receives the list of candidate grasp poses (already sorted by GraspNet
score descending) and returns the *index* of the chosen grasp.

To add a new policy, define a function and register it in POLICIES.
"""


def highest_score(poses, frame_id):
    """Pick the top-scored grasp (index 0 — GraspNet sorts by score)."""
    return 0


def nearest_depth(poses, frame_id):
    """Pick the grasp closest to the camera (smallest Z in camera frame).

    Closer grasps tend to be more reachable and have better depth accuracy.
    Only meaningful when poses are still in the camera optical frame.
    """
    return min(range(len(poses)), key=lambda i: poses[i].position.z)


def ergonomic(poses, frame_id):
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
    return highest_score(poses, frame_id)


# Map parameter strings to the corresponding policy functions
POLICIES = {
    'highest_score': highest_score,
    'nearest_depth': nearest_depth,
    'ergonomic': ergonomic,
}
