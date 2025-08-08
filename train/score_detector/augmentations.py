# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import random
from collections import defaultdict
from glob import glob
from typing import Optional

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Subset


def denormalize(tensor_img):
    img = tensor_img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def overlay_label(img, label, font_scale=0.5, color=(0, 255, 0)):
    return cv2.putText(img.copy(), label, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def tile_images(images: list[np.ndarray], tiles_per_row: int = 4) -> np.ndarray:
    n_images = len(images)
    if n_images == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    img_h, img_w = images[0].shape[:2]
    rows = math.ceil(n_images / tiles_per_row)
    grid_h = rows * img_h
    grid_w = tiles_per_row * img_w

    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // tiles_per_row
        col = idx % tiles_per_row
        y0 = row * img_h
        x0 = col * img_w
        grid_img[y0 : y0 + img_h, x0 : x0 + img_w] = img

    return grid_img


def save_augmented_batch(dataset, data_dir, epoch, num_samples: int = 16, tiles_per_row: int = 4):
    indices = random.sample(range(len(dataset)), num_samples)
    sample_subset = Subset(dataset, indices)

    vis_images = []
    for i in range(num_samples):
        img_tensor, _label = sample_subset[i]
        img = denormalize(img_tensor)
        # img = overlay_label(img, _label)
        vis_images.append(img)

    grid_img = tile_images(vis_images, tiles_per_row=tiles_per_row)
    img_path = os.path.join(data_dir, f"tile_{epoch}.png")
    cv2.imwrite(img_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
    # print(f"Saved tiled augmentations to {img_path}")


"""
Idea: use AugmentationAnalyzer to create a simple display profile and drive both training augs and cheap runtime fixes.
For each type (score, lives), compute the train vs camera stats (mean brightness, std/contrast, Laplacian var for blur,
and a simple noise proxy). Compare them to get deltas; then pick a preset per type:
- if brightness/contrast deltas are high then enable light RandomBrightnessContrast or CLAHE
- if blur delta is high then raise JPEG quality floor and add a tiny sharpen
- if noise delta is high then bump GaussNoise slightly (train-time only).

Feed those choices into the AugmentationBuilder via the _adjust_for_real step, so training learns the monitor/camera domain.
Since lives are tiny and easy to destroy, consider capping the lives intensity.

Compute the profile at startup (20-50 score/lives crops), and choose one preset per type, and apply only cheap ops to crops
(e.g., mild CLAHE or a small brightness shift, or optional unsharp for skinny fonts). Save the profile (per monitor/camera pair)
and reuse it based on camera ID.
"""


class AugmentationAnalyzer:
    def __init__(self, train_dir: str, camera_dir: Optional[str] = None, sample_limit: int = 50):
        self.train_dir = train_dir
        self.camera_dir = camera_dir
        self.sample_limit = sample_limit

    def _gather_samples(self, folder, label_type):
        files = sorted(glob(os.path.join(folder, f"img_{label_type}_*.png")))
        return files[: self.sample_limit]

    def _analyze_images(self, image_paths):
        brightness_vals = []
        contrast_vals = []
        blur_vals = []
        noise_vals = []

        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32) / 255.0

            brightness_vals.append(np.mean(img))
            contrast_vals.append(np.std(img))

            lap_var = cv2.Laplacian(img.astype(np.float64), cv2.CV_64F).var()
            blur_vals.append(lap_var)

            smoothed = cv2.GaussianBlur(img, (3, 3), 0)
            noise = np.mean(np.abs(img - smoothed))
            noise_vals.append(noise)

        return {
            "brightness": float(np.mean(brightness_vals)) if brightness_vals else 0.0,
            "contrast": float(np.mean(contrast_vals)) if contrast_vals else 0.0,
            "blur": float(np.mean(blur_vals)) if blur_vals else 0.0,
            "noise": float(np.mean(noise_vals)) if noise_vals else 0.0,
            "count": len(brightness_vals),
        }

    def analyze(self):
        result = defaultdict(dict)

        for label in ['score', 'lives']:
            result[label]['train'] = self._analyze_images(self._gather_samples(self.train_dir, label))

            if self.camera_dir and os.path.exists(self.camera_dir):
                result[label]['camera'] = self._analyze_images(self._gather_samples(self.camera_dir, label))
            else:
                result[label]['camera'] = None

        return result


class AugmentationBuilder:
    def __init__(self, game, analysis_result: dict):
        self.game = game
        self.result = analysis_result

    def _geometric_distortions(self, scale=1.0):
        if self.game == "defender":
            return [
                A.Rotate(limit=3, p=0.3 * scale),
                A.Affine(scale=(0.98, 1.02), shear=1, p=0.25 * scale),
            ]
        if self.game == "qbert":
            return [
                A.Rotate(limit=2, p=0.2 * scale),
                A.Affine(scale=(0.99, 1.01), shear=1, p=0.2 * scale),
            ]
        return [
            A.Rotate(limit=3, p=0.3 * scale),
            A.Affine(scale=(0.98, 1.02), shear=2, p=0.3 * scale),
            A.PiecewiseAffine(scale=(0.02, 0.03), nb_rows=4, nb_cols=4, p=0.4 * scale),
        ]

    def _photometric_noise(self, scale=1.0):
        if self.game == "defender":
            return [
                A.MotionBlur(blur_limit=3, p=0.1 * scale),
                A.GaussNoise(var_limit=(3, 8), p=0.25 * scale),
                A.ImageCompression(quality_lower=94, quality_upper=100, p=0.5 * scale),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5 * scale),
                A.CoarseDropout(max_holes=1, max_width=3, max_height=3, p=0.2 * scale),
            ]
        if self.game == "qbert":
            return [
                A.GaussNoise(var_limit=(3, 8), p=0.25 * scale),
                A.ImageCompression(quality_lower=94, quality_upper=100, p=0.5 * scale),
                A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.5 * scale),
            ]
        return [
            A.MotionBlur(blur_limit=3, p=0.2 * scale),
            A.GaussNoise(var_limit=(5, 15), p=0.4 * scale),
            A.ImageCompression(quality_lower=92, quality_upper=100, p=0.6 * scale),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.6 * scale),
            A.CoarseDropout(max_holes=1, max_width=4, max_height=4, p=0.3 * scale),
        ]

    def _score_base_transforms(self, scale):
        return self._geometric_distortions(scale=scale) + self._photometric_noise(scale=scale)

    def _lives_base_transforms(self, scale):
        if self.game == "defender":
            return [
                A.RandomBrightnessContrast(brightness_limit=0.04, contrast_limit=0.06, p=0.3 * scale),
                A.ImageCompression(quality_lower=96, quality_upper=100, p=0.2 * scale),
                A.CLAHE(clip_limit=1.2, p=0.2 * scale),
                A.Sharpen(alpha=(0.08, 0.15), lightness=(0.95, 1.05), p=0.15 * scale),
            ]
        return [
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.1, p=0.67 * scale),
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.67 * scale),
            A.CLAHE(clip_limit=1.2, p=0.67 * scale),
        ]

    def _build_score_font_aug(self, scale=0.3):
        if self.game == "defender":
            return A.OneOf(
                [
                    A.ColorJitter(brightness=0.08, contrast=0.10, saturation=0.06, hue=0.03, p=1.0 * scale),
                    A.CLAHE(clip_limit=1.5, tile_grid_size=(4, 4), p=0.5 * scale),
                    # .CLAHE(clip_limit=1.2, tile_grid_size=(4, 4), p=0.3 * scale),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(1, 3), p=0.4 * scale),
                            # A.GaussianBlur(blur_limit=3, p=0.10*scale),
                            A.MedianBlur(blur_limit=3, p=0.3 * scale),
                        ],
                        p=0.3,
                    ),
                    A.Sharpen(alpha=(0.15, 0.3), lightness=(0.95, 1.05), p=0.4 * scale),
                    # A.Sharpen(alpha=(0.2, 0.35), lightness=(0.95, 1.05), p=0.5 * scale),
                ],
                p=0.4,
            )
        return A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6 * scale),
                A.ColorJitter(brightness=0.07, contrast=0.07, saturation=0.05, hue=0.02, p=0.6 * scale),
                A.CLAHE(clip_limit=1.5, tile_grid_size=(4, 4), p=0.5 * scale),
                A.Downscale(scale_min=0.4, scale_max=0.8, interpolation=cv2.INTER_NEAREST, p=0.6 * scale),
                A.Equalize(mode='cv', p=1.0 * scale),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(1, 3), p=0.4 * scale),
                        A.MedianBlur(blur_limit=3, p=0.4 * scale),
                    ],
                    p=0.4,
                ),
                A.Sharpen(alpha=(0.15, 0.3), lightness=(0.95, 1.05), p=0.4 * scale),
            ],
            p=0.5,
        )

    def _build_lives_font_aug(self, scale=0.3):
        return A.OneOf(
            [
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0 * scale),
                A.ColorJitter(contrast=0.05, brightness=0.05, p=1.0 * scale),
                A.GaussNoise(var_limit=(5, 15), p=1.0 * scale),
                A.Sharpen(alpha=(0.1, 0.2), lightness=(0.9, 1.1), p=0.67 * scale),
                A.Equalize(mode='cv', p=0.67 * scale),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1.0 * scale),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0 * scale),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.05, hue=0.05, p=1.0 * scale),
            ],
            p=0.3,
        )

    def _adjust_for_real(self, label, transforms):
        cam = self.result[label]['camera']
        sim = self.result[label]['train']
        if not cam:
            return transforms

        # adjust based on deltas
        delta_brightness = abs(cam['brightness'] - sim['brightness'])
        delta_contrast = abs(cam['contrast'] - sim['contrast'])
        delta_blur = abs(cam['blur'] - sim['blur'])
        delta_noise = abs(cam['noise'] - sim['noise'])

        for t in transforms:
            if isinstance(t, A.GaussNoise) and delta_noise > 0.01:
                t.var_limit = (20, 50)
            if isinstance(t, A.ImageCompression) and delta_blur > 10.0:
                t.quality_lower = 40
            if isinstance(t, A.RandomBrightnessContrast):
                if delta_brightness > 0.05 or delta_contrast > 0.05:
                    t.brightness_limit = min(0.3, t.brightness_limit * 1.5)
                    t.contrast_limit = min(0.3, t.contrast_limit * 1.5)

        return transforms

    def build_transforms(self, score_scale=0.0, lives_scale=0.0):
        score_trans = self._score_base_transforms(scale=score_scale)
        lives_trans = self._lives_base_transforms(scale=lives_scale)

        if score_scale > 0.0:
            score_trans.append(self._build_score_font_aug(scale=score_scale))
        if lives_scale > 0.0:
            lives_trans.append(self._build_lives_font_aug(scale=lives_scale))

        if self.result['score']['camera']:
            score_trans = self._adjust_for_real('score', score_trans)
        if self.result['lives']['camera']:
            lives_trans = self._adjust_for_real('lives', lives_trans)

        return {
            'score': A.Compose(score_trans),
            'lives': A.Compose(lives_trans),
        }
