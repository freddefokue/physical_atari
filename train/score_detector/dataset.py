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

import os
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_class_weights(dataset, num_classes):
    # score labels use per-digit class balancing; lives labels use frequency of the entire label string.
    digit_counts = [0] * 10  # (0-9)
    lives_label_counts = Counter()

    for label, is_life in zip(dataset.labels, dataset.is_lives):
        if is_life:
            lives_label_counts[label] += 1
        else:
            for digit in label:
                digit_counts[int(digit)] += 1

    # print("Top 5 most common lives labels:")
    # for label, count in lives_label_counts.most_common(5):
    #    print(f"  {label}: {count}")

    total_score_samples = sum(1 for is_life in dataset.is_lives if not is_life)
    total_lives_samples = sum(lives_label_counts.values())

    digit_class_weights = []
    for i, count in enumerate(digit_counts):
        if count == 0:
            print(f"Digit class {i} has zero samples in training set. Using weight=1.0")
            digit_class_weights.append(1.0)
        else:
            w = total_score_samples / count
            digit_class_weights.append(min(w, 100.0))  # clamp to avoid explosion

    sample_weights = []
    for label, is_life in zip(dataset.labels, dataset.is_lives):
        if is_life:
            weight = total_lives_samples / (lives_label_counts[label] + 1e-6)
        else:
            weight = sum(digit_class_weights[int(d)] for d in label)
        sample_weights.append(weight)

    return sample_weights


class CustomNormalize:
    def __init__(self, score_stats, lives_stats):
        self.norm_map = {
            0: transforms.Normalize((score_stats[0],), (score_stats[1],)),  # score subset
            1: transforms.Normalize((lives_stats[0],), (lives_stats[1],)),  # lives subset
        }

    def __call__(self, image, subset_label):
        if subset_label not in self.norm_map:
            raise ValueError(f"Unknown subset label: {subset_label}")
        return self.norm_map[subset_label](image)


class MultiDigitDataset(Dataset):
    def __init__(
        self,
        root_dir,
        max_digits,
        transform=None,
        transform_score=None,
        transform_lives=None,
        score_meanstd=None,
        lives_meanstd=None,
        padding_value=-1,
    ):

        self.root_dir = root_dir
        self.max_digits = max_digits
        self.transform = transform
        self.transform_score = transform_score
        self.transform_lives = transform_lives
        self.padding_value = padding_value
        self.score_meanstd = score_meanstd
        self.lives_meanstd = lives_meanstd

        self.image_paths = []
        self.labels = []
        self.is_lives = []

        number_counter = Counter()
        digit_counter = Counter()
        digit_pos_counter = defaultdict(Counter)

        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            full_path = os.path.join(root_dir, fname)
            label_str = fname.rsplit('_', 1)[-1].split('.')[0]
            self.image_paths.append(full_path)
            self.labels.append(label_str)
            self.is_lives.append(fname.startswith("img_lives"))

            number_counter[label_str] += 1
            for pos, d in enumerate(label_str[::-1]):
                digit_counter[d] += 1
                digit_pos_counter[pos][d] += 1

        if self.score_meanstd and self.lives_meanstd:
            self.normalizer = CustomNormalize(self.score_meanstd, self.lives_meanstd)
        else:
            self.normalizer = None

        # self._print_stats(number_counter, digit_counter, digit_pos_counter)

    def _print_stats(self, number_counter, digit_counter, digit_pos_counter):
        print("\n=== Dataset Label Analysis ===")

        print("\n Full Numbers:")
        for num, count in sorted(number_counter.items()):
            print(f"  {num}: {count}")

        print("\n Overall Digit Frequency:")
        for d in map(str, range(10)):
            print(f"  {d}: {digit_counter[d]}")

        print("\n Digit Frequency by Position (right-to-left):")
        for pos in sorted(digit_pos_counter):
            print(f"  Position {pos} (10^{pos} place):")
            for d in map(str, range(10)):
                print(f"    {d}: {digit_pos_counter[pos][d]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_str = self.labels[idx]
        is_life = self.is_lives[idx]

        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)

        augment_transform = self.transform_lives if is_life else self.transform_score
        if augment_transform:
            if image_np.ndim == 2:
                image_np = np.expand_dims(image_np, axis=-1)  # H, W -> H, W, 1
            try:
                image_np = augment_transform(image=image_np)["image"]
            except Exception as e:
                print(f"Augment failed for idx={idx} with shape={image_np.shape}: {e}")
                raise

        img_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0  # CHW, float [0,1]

        if self.transform:
            img_tensor = self.transform(img_tensor)

        if self.normalizer:
            img_tensor = self.normalizer(img_tensor, int(is_life))

        padded_label = [int(d) for d in label_str] + [self.padding_value] * (self.max_digits - len(label_str))

        return img_tensor, torch.tensor(padded_label, dtype=torch.long)
