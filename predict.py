"""
U-Net Tree Segmentation - Inference
=====================================
Runs inference on a folder of street view images using a pretrained U-Net model,
producing binary segmentation masks that identify tree canopy regions.

Usage:
    python predict.py --input ./images --output ./results --model unet_membrane_HSV6.hdf5

Author: Chiau-Wei Huang
"""

import os
import argparse
from data import testGenerator, saveResult2
from model import IOU, dice_metric, jaccard_distance_loss
from keras.models import load_model


def count_images(folder):
    """Count image files in a folder."""
    extensions = ('.jpg', '.jpeg', '.png')
    return len([f for f in os.listdir(folder) if f.lower().endswith(extensions)])


def predict(input_path, output_path, model_path):
    """
    Run U-Net inference on all images in input_path.

    Args:
        input_path:  Folder containing input street view images (256x256 PNG/JPG)
        output_path: Folder to save segmentation mask results
        model_path:  Path to pretrained U-Net .hdf5 model weights
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input folder not found: {input_path}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output folder: {output_path}")

    num = count_images(input_path)
    if num == 0:
        raise ValueError(f"No images found in: {input_path}")
    print(f"Found {num} images. Loading model...")

    model = load_model(model_path, custom_objects={
        'dice_metric': dice_metric,
        'IOU': IOU,
        'jaccard_distance_loss': jaccard_distance_loss
    })
    print("Model loaded. Running inference...")

    test_gen = testGenerator(input_path, num)
    results = model.predict(test_gen, steps=num, verbose=1)

    saveResult2(output_path, results)
    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net tree canopy segmentation inference")
    parser.add_argument("--input",  required=True, help="Folder containing input images")
    parser.add_argument("--output", required=True, help="Folder to save segmentation results")
    parser.add_argument("--model",  default="unet_membrane_HSV6.hdf5", help="Path to pretrained model weights")
    args = parser.parse_args()

    predict(args.input, args.output, args.model)
