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

import argparse
import pprint

import torch


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n[Checkpoint Loaded] {path}")

    print("\n[Model Config]")
    pprint.pprint(checkpoint.get("model_config", {}))

    print("\n[Train Config]")
    pprint.pprint(checkpoint.get("train_config", {}))

    print("\n[Game Config]")
    pprint.pprint(checkpoint.get("game_config", {}))

    if "train_summary" in checkpoint:
        print("\n[Train Summary]")
        history = checkpoint["train_summary"]
        print(f"  Epochs Trained: {history['epochs']}")
        print(f"  Final Train Loss: {history['train_losses'][-1]:.4f}")
        print(f"  Final Test Loss:  {history['test_losses'][-1]:.4f}")
        print(f"  Final CER:   {history['cer'][-1]:.2f}%")
        print(f"  Final Accuracy:   {history['accuracy'][-1] * 100:.2f}%")

    print("\n[Available Keys in Checkpoint]")
    print(list(checkpoint.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    args = parser.parse_args()
    load_checkpoint(args.checkpoint_path)
