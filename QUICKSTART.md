# Quickstart Guide: Physical Atari Setup

This guide walks you through launching the Physical Atari system.

---

## What You Need

You must have the following hardware:

- **Atari 2600+ Console**
  - Set **TV Type** to `Color`
  - Set **Aspect Ratio** to `4:3`
  - This ensures output matches emulator-rendered frames.
- **Monitor**
  - Connected to the Atari 2600+ console
  - Refer to [setup.md](docs/setup.md) for recommended brightness, color mode, and refresh rate
- **Camera**
  - Recommended: Razer Kiyo Pro (1080p60)
  - If using another camera, create a config in `configs/cameras/` (see `camera_kiyo_pro.json`)
- **Controller**
  - Connected to the **Left Controller Port**
  - Either:
    - [RoboTroller](https://robotroller.keenagi.com) (mechanically actuates a CX40+ joystick)
    - MCC USB-1024LS (sends directional + fire actions via USB I/O)
    - A custom solution (requires custom device class and a matching config under `configs/controllers/`)
- **Linux System**
  - Ubuntu 24.04 LTS
  - NVIDIA GPU with **≥16GB VRAM** if running the provided agent
  - Docker + NVIDIA Container Toolkit

---

## 1. Physical Console and Camera Setup

- Mount the camera directly in front of the monitor
- Frame the Atari screen so it fills the view horizontally
- Ideal pixel mapping: **~2 camera pixels per Atari pixel**

If you're using a non-Razer camera, define your camera settings in a new config JSON under `configs/cameras/`.

---

## 2. Controller

- **RoboTroller**:
  - Follow build instructions at [robotroller.keenagi.com](https://robotroller.keenagi.com)
  - Use `configs/controllers/robotroller.json`
- **Digital I/O (MCC USB-1024LS)**:
  - Follow build instructions at [io_controller.md](docs/io_controller.md)
  - Use `configs/controllers/io_controller.json`
  - If you're using another I/O board, you must write a custom device class and define your pin map config under `configs/controllers/`.

---

## 3. Install Software Stack

Follow the full instructions in [setup.md](docs/setup.md).

You’ll need:

- NVIDIA drivers (Lambda Stack recommended)
- Docker + NVIDIA Container Toolkit
- System performance validation (`check_performance.py`)

---

## 4. Start the System

### Build the Docker Environment

```bash
./docker_build.sh
```

This sets up the runtime environment with all dependencies.

### Run the Container

```bash
./docker_run.sh
```

This gives you an interactive shell inside the container with GPU, USB, X11 forwarding, and code access.

### Launch the Physical Harness

You can now launch the main physical harness. Example (for Ms. Pac-Man with RoboTroller):

```bash
python3 harness_physical.py \
  --detection_config=configs/screen_detection/fixed.json \
  --game_config=configs/games/ms_pacman.json \
  --agent_type=agent_delay_target \
  --reduce_action_set=2 \
  --gpu=0 \
  --joystick_config=configs/controllers/robotroller.json \
  --total_frames=1_000_000
```

This will launch the GUI by default where you can configure the setup before beginning training.

---

## 5. Screen Detection Setup

You must define the 4 corners of the active screen region before any score detection can occur.

There are two options:

### Fixed Corner Selection (Default)

- In the GUI, click the **Configuration▶Screen Detection** config view to give it focus
- Use:
  - `Shift+Tab` to cycle between points
  - `WASD` to move the selected point
- Position corners precisely around the 4:3 screen content (exclude pillarbox bars)
- Click **Save** to persist the region config

### April Tag Detection (Advanced)

- Requires physical AprilTags printed and placed at the four corners of the screen
- Provides automatic detection but is sensitive to lighting, reflections, and tag size
- Best used with a ring light or consistent ambient illumination
- Configuration and tag details are in [setup.md](docs/setup.md)
- To use, specify: `--detection_config=configs/screen_detection/april_tags.json`

---

## 6. Score and Lives Region Configuration

- In the GUI, click the **Configuration▶Score Detection** config view to give it focus
- Use:
  - `Shift+Tab` to toggle between score and lives box
  - `WASD` to adjust box position
- Click **Save** to commit changes

> Reference images for correct placement are available in [docs/setup.md](docs/setup.md#score-and-lives-box-placement)
  Note: When using the default models shipped with this framework, incorrect crop placement may degrade accuracy.
  The default models were trained with tight bounds for certain games to exclude nearby HUD graphics. Some games may
  benefit from looser bounds, and in those cases, retraining with larger crops can improve robustness.

Click the **Game Frame** view to give it focus, where you can use your keyboard to control the game, and verify that score and lives are correctly parsed in real time.

---

## 7. Start Training

Once the screen, score, and lives regions are configured, you can:

- Start training runs from within the GUI
- Monitor training progress in real time via the displayed graphs
- View per-frame details including score, lives, episode count, frame number, action, and other data

Results will be written to the configured `results/` path.

---

## 8. Troubleshooting Performance

Even with setup complete, your system may throttle GPU or CPU performance. If framerate or latency is unstable:

- Run:
  ```bash
  sudo python3 scripts/check_performance.py
  ```
- Apply fixes or reboot if prompted

If performance still degrades after reboot, power down and **fully disconnect** all devices for 30 seconds (flea drain) to reset transient hardware state.

---

Refer to [setup.md](docs/setup.md) for detailed installation instructions, hardware tuning, and system diagnostics.
