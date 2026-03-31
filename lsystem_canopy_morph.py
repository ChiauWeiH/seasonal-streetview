"""
L-System Canopy Morphing
========================
Uses L-System branch structures to drive TPS (Thin Plate Spline) warping
of tree canopy images, simulating seasonal structural changes.

Usage:
    python lsystem_canopy_morph.py --input canopy.png --output result.png

Author: Chiau-Wei Huang
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import regionprops, label
from scipy.interpolate import Rbf


# =========================================================
# L-System
# =========================================================

def generate_l_system(axiom, rules, iterations):
    """Recursively expand L-System string."""
    current_string = axiom
    for _ in range(iterations):
        current_string = ''.join([rules.get(c, c) for c in current_string])
    return current_string


def draw_l_system(instructions, angle, length, canvas_width, canvas_height):
    """
    Convert L-System string into a list of line segments.
    Returns list of (x1, y1, x2, y2) tuples.
    """
    stack, lines = [], []
    x, y = canvas_width // 2, canvas_height - 10
    current_angle = -90  # Start pointing upward

    for cmd in instructions:
        if cmd == 'F':
            new_x = x + length * math.cos(math.radians(current_angle))
            new_y = y + length * math.sin(math.radians(current_angle))
            lines.append([x, y, new_x, new_y])
            x, y = new_x, new_y
        elif cmd == '+':
            current_angle += angle
        elif cmd == '-':
            current_angle -= angle
        elif cmd == '[':
            stack.append((x, y, current_angle))
        elif cmd == ']':
            x, y, current_angle = stack.pop()

    return lines


# =========================================================
# Control Point Matching
# =========================================================

def closest_line_with_index(point, lines):
    """Find the index of the closest line segment to a given point."""
    min_dist, closest_idx = float('inf'), None
    for idx, (x1, y1, x2, y2) in enumerate(lines):
        px, py = point
        dx, dy = x2 - x1, y2 - y1
        denom = dx**2 + dy**2
        if denom == 0:
            continue
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / denom))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dist = np.hypot(px - proj_x, py - proj_y)
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx
    return closest_idx


def compute_control_points(src_points, lines_initial, lines_extreme, max_dist=25):
    """
    For each source point, find its closest branch segment and compute
    where it should move based on the structural change between
    the initial and extreme L-System states.

    Args:
        src_points:     Dense grid of points inside the canopy mask
        lines_initial:  L-System branch segments (before deformation)
        lines_extreme:  L-System branch segments (after deformation)
        max_dist:       Maximum allowed displacement per point (pixels)

    Returns:
        dst_points: List of displaced destination points
    """
    dst_points = []

    for pt in src_points:
        idx = closest_line_with_index(pt, lines_initial)
        x1, y1, x2, y2 = lines_initial[idx]
        x1n, y1n, x2n, y2n = lines_extreme[idx]

        v_init = np.array([x2 - x1, y2 - y1])
        v_new = np.array([x2n - x1n, y2n - y1n])

        norm_new = np.linalg.norm(v_new)
        if norm_new == 0:
            dst_points.append(pt)
            continue

        # Compute rotation and scale between initial and new branch vectors
        scale = np.linalg.norm(v_init) / norm_new
        angle_init = math.atan2(v_new[1], v_new[0])
        angle_new = math.atan2(v_init[1], v_init[0])
        rotation = angle_new - angle_init

        relative = np.array([pt[0] - x1, pt[1] - y1])
        rotation_matrix = np.array([
            [math.cos(rotation), -math.sin(rotation)],
            [math.sin(rotation),  math.cos(rotation)]
        ]) * scale

        new_relative = rotation_matrix @ relative
        new_pt = np.array([x1n, y1n]) + new_relative

        # Clamp displacement to max_dist
        move_vec = new_pt - pt
        move_len = np.linalg.norm(move_vec)
        if move_len > max_dist:
            move_vec = move_vec / move_len * max_dist
            new_pt = pt + move_vec

        dst_points.append(new_pt)

    return dst_points


# =========================================================
# TPS Warp
# =========================================================

def tps_warp(canopy_img, src_points, dst_points, map_margin=85):
    """
    Apply Thin Plate Spline warping to the canopy image.

    Args:
        canopy_img:  BGRA canopy image (numpy array)
        src_points:  Source control points (Nx2)
        dst_points:  Destination control points (Nx2)
        map_margin:  Extra boundary pixels to avoid clipping

    Returns:
        Warped BGRA image as uint8
    """
    canvas_height, canvas_width = canopy_img.shape[:2]

    src_x = np.array(src_points)[:, 0]
    src_y = np.array(src_points)[:, 1]
    dst_x = np.array(dst_points)[:, 0]
    dst_y = np.array(dst_points)[:, 1]

    rbf_x = Rbf(src_x, src_y, dst_x, function='thin_plate')
    rbf_y = Rbf(src_x, src_y, dst_y, function='thin_plate')

    # Estimate output size
    grid_y_temp, grid_x_temp = np.mgrid[0:canvas_height, 0:canvas_width]
    map_x_temp = rbf_x(grid_x_temp.flatten(), grid_y_temp.flatten())
    map_y_temp = rbf_y(grid_x_temp.flatten(), grid_y_temp.flatten())

    output_width = max(canvas_width, int(np.max(map_x_temp)) + map_margin)
    output_height = max(canvas_height, int(np.max(map_y_temp)) + map_margin)

    # Build final warp maps
    grid_y, grid_x = np.mgrid[0:output_height, 0:output_width]
    map_x = rbf_x(grid_x.flatten(), grid_y.flatten()).reshape((output_height, output_width)).astype(np.float32)
    map_y = rbf_y(grid_x.flatten(), grid_y.flatten()).reshape((output_height, output_width)).astype(np.float32)

    # Warp RGB and alpha separately
    warped_rgb = cv2.remap(canopy_img[:, :, :3], map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
    warped_alpha = cv2.remap(canopy_img[:, :, 3], map_x, map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

    return np.dstack([warped_rgb, warped_alpha]).astype(np.uint8)


# =========================================================
# Main
# =========================================================

def main(input_path, output_path, show=False):
    # --- Load canopy image ---
    canopy_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if canopy_img is None:
        raise FileNotFoundError(f"Cannot open image: {input_path}")
    if canopy_img.shape[2] == 3:
        alpha = np.full(canopy_img.shape[:2], 255, dtype=np.uint8)
        canopy_img = np.dstack([canopy_img, alpha])

    canvas_height, canvas_width = canopy_img.shape[:2]

    # --- L-System parameters ---
    # Adjust these to control the degree of seasonal deformation
    axiom = "F"
    rules = {"F": "F[+F][-F]"}
    iterations = 2

    angle_initial  = 10   # branch angle for initial state (narrow = winter-like)
    length_initial = 5    # branch length for initial state

    angle_extreme  = 10   # branch angle for target state
    length_extreme = 8    # branch length for target state (longer = summer-like)

    inst = generate_l_system(axiom, rules, iterations)
    lines_initial = draw_l_system(inst, angle_initial, length_initial, canvas_width, canvas_height)
    lines_extreme = draw_l_system(inst, angle_extreme, length_extreme, canvas_width, canvas_height)

    # --- Build dense control points from canopy mask ---
    alpha_mask = canopy_img[:, :, 3] > 0
    bbox = regionprops(label(alpha_mask))[0].bbox
    min_r, min_c, max_r, max_c = bbox
    grid_y, grid_x = np.mgrid[min_r:max_r:2, min_c:max_c:2]
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    src_points = np.array([(x, y) for x, y in grid_points if alpha_mask[int(y), int(x)]])

    # --- Compute displaced control points ---
    dst_points = compute_control_points(src_points, lines_initial, lines_extreme, max_dist=25)

    # --- Apply TPS warp ---
    warped = tps_warp(canopy_img, src_points, dst_points)

    # --- Save output ---
    cv2.imwrite(output_path, warped)
    print(f"Saved: {output_path}")

    # --- Optional visualization ---
    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(canopy_img, cv2.COLOR_BGRA2RGBA))
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA))
        plt.title("Warped (L-System Morphing)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L-System driven canopy morphing via TPS warp")
    parser.add_argument("--input",  required=True, help="Path to input canopy image (PNG with alpha)")
    parser.add_argument("--output", required=True, help="Path to save warped output image")
    parser.add_argument("--show",   action="store_true", help="Show before/after visualization")
    args = parser.parse_args()

    main(args.input, args.output, show=args.show)
