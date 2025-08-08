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

import json
import os
import random
import shutil
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CropInfo:
    def __init__(self, x, y, w, h, num_digits):
        self.x_start = x
        self.y_start = y
        self.x_offset = w
        self.y_offset = h
        self.num_digits = num_digits


DATA_EXTS = [".png"]

# digits are classified as (0-9)
# life symbol is classified as 10
# Since we process score and lives separately:
# 0-9 is returned for the score.
# for lives with LIFE_DIGIT = 10, 0=life = 0 ,1 head = 10, 2 heads = 1010.
# the model should return 10, or 1010. then we count the number of 10s to get the number of life symbols.
LIFE_DIGIT = 10


def balance_lives_samples(data_dir, target_ratio=0.5, max_duplication=5000):
    # balance the number of lives images by duplicating each to achieve target_ratio * num_scores total lives

    files = [f for f in os.listdir(data_dir) if os.path.splitext(f)[-1].lower() in DATA_EXTS]

    score_files = [f for f in files if f.startswith('img_score_')]
    lives_files = [f for f in files if f.startswith('img_lives_')]

    if not lives_files:
        print("No lives images found.")
        return

    num_scores = len(score_files)
    num_lives = len(lives_files)

    target_total_lives = int(num_scores * target_ratio)
    copies_per_lives = min(max_duplication, max(1, int(target_total_lives / num_lives)))

    print(f"Score images: {num_scores}")
    print(f"Lives images: {num_lives}")
    print(f"Target lives images: {target_total_lives}")
    print(f"Duplicating each lives image {copies_per_lives} times")

    # group lives images by lives combo to avoid over-representing some
    for f in lives_files:
        fname, ext = os.path.splitext(f)
        vals = fname.rsplit('_')
        lives_str = vals[-1]

        # rename original to dupe_idx = 0
        new_base = f"img_lives_0_{lives_str}{ext}"
        os.rename(os.path.join(data_dir, f), os.path.join(data_dir, new_base))

        # duplicate it
        for i in range(1, copies_per_lives):
            new_filename = f"img_lives_{i}_{lives_str}{ext}"
            src_path = os.path.join(data_dir, new_base)
            dst_path = os.path.join(data_dir, new_filename)
            shutil.copy2(src_path, dst_path)

    total_lives_after = len([f for f in os.listdir(data_dir) if f.startswith("img_lives_")])
    print(f"lives duplication complete. Total lives files: {total_lives_after}")


def balance_score_samples(data_dir, max_duplicates=5000):
    files = [f for f in os.listdir(data_dir) if os.path.splitext(f)[-1].lower() in DATA_EXTS]
    score_files = [f for f in files if f.startswith("img_score_")]

    if not score_files:
        print("No score images found.")
        return

    digit_groups = defaultdict(list)
    score_to_file = {}

    for f in score_files:
        fname, ext = os.path.splitext(f)
        parts = fname.split("_")
        try:
            score_val_str = parts[-1]
            score_val = int(score_val_str)
            digit_len = len(str(score_val))  # ignore leading 0s
            digit_groups[digit_len].append(score_val)
            score_to_file[score_val] = f
        except ValueError:
            print(f"Skipping invalid file: {f}")
            continue

    group_counts = {k: len(v) for k, v in digit_groups.items()}
    max_count = max(group_counts.values())
    max_group = max(group_counts, key=group_counts.get)
    print(f"Balancing digit length groups to {max_count} samples each")

    for digit_len, scores in digit_groups.items():
        if digit_len == max_group:
            print(f"  digit length: {digit_len} is the max group ({len(scores)}); skipping augmentation.")
            continue

        print(f"digit length: {digit_len} num_scores: {len(scores)}")
        needed = min(max_count - len(scores), max_duplicates)
        if needed <= 0:
            continue

        print(f"  digit_length: {digit_len}: curr={len(scores)}: want={max_count} (adding {needed})")

        random.shuffle(scores)

        for i in range(needed):
            score_val = scores[i % len(scores)]
            src_file = score_to_file[score_val]
            fname, ext = os.path.splitext(src_file)
            score_val_str = fname.split("_")[-1]

            # determine next available dupe_index
            dupe_index = 1
            while True:
                dst_file = f"img_score_{dupe_index}_{score_val_str}{ext}"
                dst_path = os.path.join(data_dir, dst_file)
                if not os.path.exists(dst_path):
                    break
                dupe_index += 1

            src_path = os.path.join(data_dir, src_file)
            shutil.copy2(src_path, dst_path)

    total = len([f for f in os.listdir(data_dir) if f.startswith("img_score_")])
    print(f"Score digit balancing complete. Total score files: {total}")


def print_class_distribution(data_dir):
    files = os.listdir(data_dir)

    lives_vals = []
    score_vals = []

    for f in files:
        fname, _ = os.path.splitext(f)
        if f.startswith("img_lives_"):
            try:
                lives_val = int(fname.rsplit("_", 1)[-1])
                lives_vals.append(lives_val)
            except ValueError:
                print(f"Skipping invalid lives filename: {f}")
        elif f.startswith("img_score_"):
            try:
                score_val = int(fname.rsplit("_", 1)[-1])
                score_vals.append(score_val)
            except ValueError:
                print(f"Skipping invalid score filename: {f}")

    print("\nLives Class Distribution:")
    for val, count in sorted(Counter(lives_vals).items()):
        print(f"  Lives = {val}: {count} samples")

    print("\nScore Digit Length Distribution:")
    digit_len_counts = Counter(len(str(v)) for v in score_vals)
    for length in sorted(digit_len_counts):
        print(f"  {length}-digit scores: {digit_len_counts[length]} samples")

    print(f"\nTotals: Lives: {len(lives_vals)}, Scores: {len(score_vals)}\n")

    # score_values = [int(s) for s in score_vals]  # or however you have them stored
    # print(Counter(score_values))


def crop_images(data_dir, processed_dir, game, total_lives, score_config):
    image_filenames = [f for f in os.listdir(data_dir) if (os.path.splitext(f)[-1] in DATA_EXTS)]
    num_files = len(image_filenames)

    print(f"Processing {num_files} files...")

    score_crop_info = score_config.get("score_crop_info")
    lives_crop_info = score_config.get("lives_crop_info")

    score_crop_info = CropInfo(**score_crop_info) if score_crop_info else None
    lives_crop_info = CropInfo(**lives_crop_info) if lives_crop_info else None

    capture_lives = lives_crop_info is not None

    for idx, f in enumerate(image_filenames):
        fname, _ = os.path.splitext(f)
        vals = fname.rsplit('_')
        val_str = vals[-1]

        img_path = os.path.join(data_dir, f)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if f.startswith("img_score_"):
            score = int(val_str)
            crop = image[
                score_crop_info.y_start : score_crop_info.y_start + score_crop_info.y_offset,
                score_crop_info.x_start : score_crop_info.x_start + score_crop_info.x_offset,
                :,
            ]

            # exclude blank images
            gray = np.dot(crop[..., :3], [0.299, 0.587, 0.114])
            if np.std(gray) < 1.0:
                print(f"skipping: blank or solid color for score {score}")
                continue

            # REVIEW: garbage or incorrect data is rendered for these combos
            krull_bad_scores = [13950, 53640, 61540, 69600, 77070, 91920]
            if game == "krull" and score in krull_bad_scores:
                continue
            atlantis_bad_score = [54000]
            if game == "atlantis" and score in atlantis_bad_score:
                continue
            up_n_down_bad_scores = [68920]
            if game == "up_n_down" and score in up_n_down_bad_scores:
                continue

            if game == 'centipede':
                score_str = f"{score:02d}" if score < 10 else str(score)
            elif game == 'defender' or game == 'battle_zone':
                score_str = f"{score:06d}"
            elif game == 'qbert':
                score_str = f"{score:05d}"
            else:
                score_str = str(score)

            out_path = os.path.join(processed_dir, f"img_score_{score_str}.png")
            Image.fromarray(crop).save(out_path)

        elif f.startswith("img_lives_") and capture_lives:
            num_lives = int(val_str)
            num_displayed = lives_crop_info.num_digits
            max_lives = total_lives

            # Adjust for cartridges that show fewer lives than they track
            lives_increment = max(0, max_lives - num_displayed)
            num_lives -= lives_increment

            crop = image[
                lives_crop_info.y_start : lives_crop_info.y_start + lives_crop_info.y_offset,
                lives_crop_info.x_start : lives_crop_info.x_start + lives_crop_info.x_offset,
                :,
            ]

            # don't train blank as 0
            if num_lives <= 0:
                print(f"[SKIP] Blank or zero-life crop: {f}")
                continue

            if np.mean(crop) == 0.0:
                print(f"blank crop image: {f}")

            lives_digits = [LIFE_DIGIT] * num_lives
            lives_str = "".join(str(d) for d in lives_digits)
            out_path = os.path.join(processed_dir, f"img_lives_{lives_str}.png")
            Image.fromarray(crop).save(out_path)

    processed_filenames = [f for f in os.listdir(processed_dir) if os.path.isfile(os.path.join(processed_dir, f))]
    print(f"Wrote {len(processed_filenames)} processed images to {processed_dir}.")


def split_images(preprocessed_dir, output_dir, train_percent=0.8):
    train_folder = os.path.join(output_dir, "train")
    test_folder = os.path.join(output_dir, "test")

    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)

    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)

    os.makedirs(train_folder)
    os.makedirs(test_folder)

    image_filenames = [f for f in os.listdir(preprocessed_dir) if os.path.isfile(os.path.join(preprocessed_dir, f))]

    score_images = [f for f in image_filenames if f.startswith("img_score")]
    lives_images = [f for f in image_filenames if f.startswith("img_lives")]

    def split_and_move(file_list, label):
        np.random.shuffle(file_list)
        split_idx = int(len(file_list) * train_percent)
        train_files = file_list[:split_idx]
        test_files = file_list[split_idx:]
        print(f"{label}: train={len(train_files)} test={len(test_files)}")

        for f in train_files:
            shutil.move(os.path.join(preprocessed_dir, f), os.path.join(train_folder, f))
        for f in test_files:
            shutil.move(os.path.join(preprocessed_dir, f), os.path.join(test_folder, f))

    split_and_move(score_images, label="score")
    if lives_images:
        split_and_move(lives_images, label="lives")

    print(f"Finished splitting dataset into {train_folder} and {test_folder}")
    return train_folder, test_folder


class SubsetImageDataset(Dataset):
    def __init__(self, folder, subset_type, image_width, image_height):
        self.files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.startswith(f"img_{subset_type}_")
        ]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_height, image_width)),
                transforms.Grayscale(),
            ]
        )
        print(f"{subset_type} subset: {len(self.files)} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def compute_stats(dataloader):
    total_mean = 0
    total_std = 0
    total_count = 0

    for imgs in dataloader:
        b, _, _, _ = imgs.shape
        imgs = imgs.view(b, -1)
        total_mean += imgs.mean(dim=1).sum()
        total_std += imgs.std(dim=1).sum()
        total_count += b

    mean = total_mean / total_count
    std = total_std / total_count
    return mean.item(), std.item()


def compute_train_split_stats(train_folder, image_width, image_height):
    score_dataset = SubsetImageDataset(train_folder, "score", image_width, image_height)
    score_loader = DataLoader(score_dataset, batch_size=64, shuffle=False)
    score_mean, score_std = compute_stats(score_loader)

    lives_dataset = SubsetImageDataset(train_folder, "lives", image_width, image_height)
    lives_loader = DataLoader(lives_dataset, batch_size=64, shuffle=False)
    if len(lives_loader) == 0:
        print("No lives generated.")
        lives_mean, lives_std = 0.0, 0.0
    else:
        lives_mean, lives_std = compute_stats(lives_loader)

    print(f"Score mean/std: {score_mean:.3f} / {score_std:.3f}")
    print(f"Lives mean/std: {lives_mean:.3f} / {lives_std:.3f}")

    return (score_mean, score_std), (lives_mean, lives_std)


def preprocess_data(data_dir, output_dir, game, total_lives, score_config, auto_balance=False):
    processed_dir = os.path.join(output_dir, "processed")

    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)

    os.makedirs(processed_dir)

    print(f"Preprocessing data for {game}")

    crop_images(data_dir, processed_dir, game, total_lives, score_config)
    if auto_balance:
        balance_score_samples(processed_dir)
        balance_lives_samples(processed_dir)
    print_class_distribution(processed_dir)

    train_dir, test_dir = split_images(processed_dir, output_dir, train_percent=0.8)
    return train_dir, test_dir


def get_argument_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="preprocess_dataset.py arguments")
    parser.add_argument('--game_config', type=str, default="configs/games/ms_pacman.json")
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'results'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(), 'results'))
    parser.add_argument('--debug', action='store_true')
    return parser


if __name__ == '__main__':
    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    try:
        with open(args.game_config) as gf:
            game_data = gf.read()

        game_config = json.loads(game_data)["game_config"]
        game = game_config["name"]
        total_lives = game_config["lives"]
        score_config = json.loads(game_data)["score_config"]

        preprocess_data(args.data_dir, args.output_dir, game, total_lives, score_config)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")

    exit(0)
