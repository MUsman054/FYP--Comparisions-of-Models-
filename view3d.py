"""
view3d.py — live 3D view of the lifted pose with X / Y / Z axes.

Display frame (origin = pelvis):
    X = right (as the webcam sees it)   red
    Y = up (calibrated vertical)        green
    Z = toward the webcam               blue
Azimuth rotates around Y, elevation tilts the view from above.
Presets: "Front" = what the webcam sees, "Side" = lean is visible, "Top" = depth from above.
"""

import cv2
import numpy as np

# Human3.6M 17-joint skeleton (same order as the lifters output)
EDGES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
LEFT, RIGHT = {4, 5, 6, 11, 12, 13}, {1, 2, 3, 14, 15, 16}

PRESETS = {"3/4": (35, 20), "Front": (0, 0), "Side": (90, 0), "Top": (0, 89)}

# BGR
BG, GRID, TXT, DIM = (28, 28, 28), (65, 65, 65), (235, 235, 235), (150, 150, 150)
C_X, C_Y, C_Z = (70, 70, 235), (80, 210, 80), (240, 150, 50)
C_L, C_R, C_C = (230, 170, 40), (40, 150, 255), (220, 220, 220)


def _rot_to(a, b):
    """Rotation matrix turning unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def _view_matrix(az, el):
    a, e = np.radians(az), np.radians(el)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return rx @ ry


def to_display(p3d, up=None):
    """Camera coords (x right, y down, z away) -> display coords, pelvis at origin,
    tilted so the calibrated vertical points straight up."""
    flip = np.diag([1.0, -1.0, -1.0])
    p = (np.asarray(p3d, float) - p3d[0]) @ flip.T
    if up is not None:
        u = flip @ np.asarray(up, float)
        n = np.linalg.norm(u)
        if n > 1e-6:
            u = u / n
            if u[1] < 0:                      # accept either sign convention for `up`
                u = -u
            p = p @ _rot_to(u, np.array([0.0, 1.0, 0.0])).T
    return p


class View3D:
    def __init__(self, size=(480, 480)):
        self.w, self.h = size
        self.trunk = None                      # smoothed trunk length, keeps the scale steady

    def render(self, p3d, up=None, az=35, el=20):
        """Returns an RGB image."""
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        if p3d is None:
            cv2.putText(img, "no 3D pose yet - pick a 3D lifter", (20, self.h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, DIM, 1, cv2.LINE_AA)
            return img[:, :, ::-1]

        p = to_display(p3d, up)
        t = max(np.linalg.norm(p[8] - p[0]), 1e-6)
        self.trunk = t if self.trunk is None else 0.9 * self.trunk + 0.1 * t
        T = self.trunk
        s = self.h * 0.2 / T
        cx, cy = self.w / 2, self.h * 0.55
        R = _view_matrix(az, el)

        def proj(v):
            q = R @ v
            return (int(cx + s * q[0]), int(cy - s * q[1])), q[2]

        def line(a, b, col, th=1):
            cv2.line(img, proj(a)[0], proj(b)[0], col, th, cv2.LINE_AA)

        # floor grid (XZ plane at the lowest joint)
        fy, g = p[:, 1].min(), 1.5 * T
        for k in np.linspace(-g, g, 7):
            line(np.array([k, fy, -g]), np.array([k, fy, g]), GRID)
            line(np.array([-g, fy, k]), np.array([g, fy, k]), GRID)

        # axes from the pelvis
        L = 0.7 * T
        for vec, col, name in [((L, 0, 0), C_X, "X"), ((0, L, 0), C_Y, "Y"), ((0, 0, L), C_Z, "Z")]:
            end = np.array(vec, float)
            cv2.arrowedLine(img, proj(np.zeros(3))[0], proj(end)[0], col, 2, cv2.LINE_AA, tipLength=0.15)
            x, y = proj(end * 1.15)[0]
            cv2.putText(img, name, (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

        # webcam marker, in front of the person at head height
        cam, _ = proj(np.array([0.0, p[10, 1], 2.0 * T]))
        cv2.rectangle(img, (cam[0] - 7, cam[1] - 5), (cam[0] + 7, cam[1] + 5), C_Z, -1)
        tx = cam[0] + 10 if cam[0] + 70 < self.w else cam[0] - 62     # keep the label on screen
        cv2.putText(img, "webcam", (tx, cam[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_Z, 1, cv2.LINE_AA)

        # bones, far ones first so near ones draw on top
        bones = []
        for a, b in EDGES:
            (pa, za), (pb, zb) = proj(p[a]), proj(p[b])
            col = C_L if b in LEFT else C_R if b in RIGHT else C_C
            bones.append(((za + zb) / 2, pa, pb, col))
        for _, pa, pb, col in sorted(bones, key=lambda x: x[0]):
            cv2.line(img, pa, pb, col, 3, cv2.LINE_AA)
            cv2.circle(img, pb, 4, col, -1, cv2.LINE_AA)
        cv2.circle(img, proj(p[0])[0], 4, C_C, -1, cv2.LINE_AA)

        cv2.putText(img, f"az {az:.0f}  el {el:.0f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)
        cv2.putText(img, "blue = left   orange = right", (10, self.h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1, cv2.LINE_AA)
        return img[:, :, ::-1]


if __name__ == "__main__":                      # quick self-test: seated person leaning 30°
    th = np.radians(30)
    j = np.zeros((17, 3))
    j[1], j[4] = [-0.12, 0, 0], [0.12, 0, 0]                     # hips (camera coords, y down)
    for hip, kn, an in [(1, 2, 3), (4, 5, 6)]:
        j[kn] = j[hip] + [0, 0, -0.45]                           # thighs toward camera
        j[an] = j[kn] + [0, 0.45, 0]
    j[7] = [0, -0.25 * np.cos(th), -0.25 * np.sin(th)]
    j[8] = [0, -0.5 * np.cos(th), -0.5 * np.sin(th)]
    j[9], j[10] = j[8] + [0, -0.1, -0.05], j[8] + [0, -0.22, -0.08]
    j[14], j[11] = j[8] + [0.2, 0, 0], j[8] + [-0.2, 0, 0]
    j[15], j[12] = j[14] + [0, 0.25, -0.12], j[11] + [0, 0.25, -0.12]
    j[16], j[13] = j[15] + [0, 0, -0.25], j[12] + [0, 0, -0.25]
    v = View3D()
    tiles = [v.render(j, (0, -1, 0), *PRESETS[k])[:, :, ::-1] for k in PRESETS]
    cv2.imwrite("view3d_test.png", np.hstack(tiles))
    print("saved view3d_test.png")
