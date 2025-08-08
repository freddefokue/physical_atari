# Profiling the Physical System

This document outlines how to **profile performance of the physical setup** using NVIDIA Nsight tools inside a **Docker container** running on an Ubuntu 24.04 system with NVIDIA GPU(s).

The main performance focus is on:

- CPU–GPU memory transfer latency
- USB/IO contention, especially during input polling
- Device-level scheduling issues
- X11 GUI rendering overhead via containerized forwarding
- Multi-threaded behavior within the control loop

---

## Tool Overview

### [Nsight Systems](https://developer.nvidia.com/nsight-systems)

A timeline-based system-wide profiler for analyzing:

- CPU/GPU concurrency
- OS thread scheduling
- CUDA kernel launches
- Blocking I/O behavior (USB, GUI)

### [Nsight Compute](https://developer.nvidia.com/nsight-compute)

A low-level kernel analysis tool for examining:

- GPU kernel throughput
- Memory access patterns
- Warp occupancy and stall reasons

> Nsight Compute is **not yet included** in the Docker image.

---

## Docker Environment

The profiling takes place inside a Docker container that includes:

- CUDA runtime
- Nsight Systems CLI and UI tools
- PyTorch with CUDA + NVTX support
- X11 forwarding for GUI rendering

---

## NVTX Annotations (via PyTorch)

torch.cuda.nvtx annotations are used to instrument critical sections of code, enabling detailed performance analysis in Nsight Systems by:

- Visualizing the timing and concurrency of individual components within the Nsight timeline.
- Identifying CPU or GPU stalls and resource contention, such as between USB polling and GPU tasks.
- Measuring latency and synchronization between key stages like data capture, inference, and control signal output.

```python
import torch.cuda.nvtx as nvtx

for i in range(num_frames):
    nvtx.range_push("env.act")
    env.act()
    nvtx.range_pop()

    nvtx.range_push("env.get_observation")
    observation_rgb8[i] = env.get_observation()
    nvtx.range_pop()

    nvtx.range_push("agent.accept_observation")
    taken_action = agent.accept_observations(observation_rgb8, rewards, end_of_episodes)
    nvtx.range_pop()
```

---

## Running Nsight Systems

Use the following command to launch a 60-second profiling session:

```bash
nsys profile \
  --stats=true \
  --sample=cpu \
  --trace=cuda,cudnn,cublas,nvtx,osrt,oshmem \
  --cudabacktrace=kernel:1000000,sync:1000000,memory:1000000 \
  --delay=1 \
  --duration=60 \
  --wait=all \
  --force-overwrite=true \
  --output="/tmp/nsys_profile" \
  python harness_physical.py --use_gui=0
```

- '--delay=1': Skips early initialization overhead
- '--trace': Includes CUDA, NVTX, and OS runtime events
- '--output': Stores the profile as '/tmp/nsys_profile.nsys-rep'
- '--use_gui': Set to 1 to include the GUI.

---

### Open Results in nsys GUI

Outside the container, copy the result and run:

```bash
nsys-ui /tmp/nsys_profile.nsys-rep
```

Or run the GUI directly from the container as X11 forwarding is supported.

---

## What to Look for in Profiling

| Aspect                | Indicators to Watch For                               |
| --------------------- | ----------------------------------------------------- |
| USB Camera Capture    | Long CPU threads polling USB, USB bandwidth stalls    |
| USB Control Signaling | Delays or blocking on USB writes                      |
| GPU Inference         | GPU kernel stalls, inefficient memory transfers       |
| PyTorch Training      | CPU/GPU imbalance, memory bottlenecks                 |
| Inter-component Sync  | CPU waits between USB, GPU, and training steps        |
| (If GUI enabled)      | Thread stalls during rendering, X11 forwarding delays |
