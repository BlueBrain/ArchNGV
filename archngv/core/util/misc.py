def VOI(points, bb):
    """
    Volume of Interest defined by the bounding box
    """
    mask = (points[:, 0] >= bb[0, 0]) & (points[:, 0] <= bb[1, 0]) & \
           (points[:, 1] >= bb[0, 1]) & (points[:, 1] <= bb[1, 1]) & \
           (points[:, 2] >= bb[0, 2]) & (points[:, 2] <= bb[1, 2])

    return points[mask, :]

