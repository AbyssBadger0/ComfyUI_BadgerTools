import numpy as np
import cv2
from segment_anything import SamPredictor


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


def predict_masks_with_sam(
        img,
        point_coords,
        point_labels,
        ckpt_p,
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    predictor = SamPredictor(ckpt_p)

    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks, scores, logits


def get_masks(img, latest_coords, dilate_kernel_size, sam_ckpt):
    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        [1],
        sam_ckpt,
    )
    masks = masks.astype(np.uint8) * 255

    masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    return masks
