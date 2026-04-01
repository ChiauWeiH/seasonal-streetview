"""
Color Transfer for Seasonal Canopy
====================================
Transfers the color style of a seasonal canopy image onto a target canopy,
adapting it to match the lighting and color distribution of the street view scene.

Based on the Reinhard et al. (2001) color transfer algorithm, implemented
via the `colortrans` library. Supports LHM, PCCM, and Reinhard methods.

Usage:
    python color_transfer.py --source canopy_autumn.png --target streetview.png --output result.png

Author: Chiau-Wei Huang
"""

import argparse
import cv2


def color_transfer(source_path, target_path, output_path, method="lhm", show=False):
    """
    Apply color transfer from source image to target image.

    Args:
        source_path:  Path to the source image (seasonal canopy, provides color style)
        target_path:  Path to the target image (street view scene, receives color style)
        output_path:  Path to save the output image
        method:       Color transfer method: 'lhm', 'pccm', or 'reinhard'
        show:         Whether to display before/after comparison
    """
    from colortrans import transfer_lhm, transfer_pccm, transfer_reinhard

    source = cv2.imread(source_path)
    target = cv2.imread(target_path)

    if source is None:
        raise FileNotFoundError(f"Cannot open source image: {source_path}")
    if target is None:
        raise FileNotFoundError(f"Cannot open target image: {target_path}")

    # Apply selected color transfer method
    if method == "lhm":
        result = transfer_lhm(target, source)
    elif method == "pccm":
        result = transfer_pccm(target, source)
    elif method == "reinhard":
        result = transfer_reinhard(target, source)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: lhm, pccm, reinhard")

    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")

    if show:
        cv2.imshow("Source (color style)", source)
        cv2.imshow("Target (original)", target)
        cv2.imshow("Result (after transfer)", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seasonal canopy color transfer")
    parser.add_argument("--source", required=True, help="Source image path (provides color style)")
    parser.add_argument("--target", required=True, help="Target image path (receives color style)")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--method", default="lhm", choices=["lhm", "pccm", "reinhard"],
                        help="Color transfer method (default: lhm)")
    parser.add_argument("--show", action="store_true", help="Show before/after comparison")
    args = parser.parse_args()

    color_transfer(args.source, args.target, args.output, method=args.method, show=args.show)
