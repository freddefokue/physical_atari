import argparse
import copy
import json
import math
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ale_py import Action, ALEInterface, LoggerMode, roms
from pynvml import *

from benchmark_runner import BenchmarkConfig, BenchmarkRunner, CycleConfig, GameSpec


def train_function(
    # constants
    input_stack,
    frame_skip,
    train_batch,
    epsilon_tensor,
    discount,
    reward_clip,
    num_actions,
    # variable inputs
    tensor_u,
    filled_frames_tensor,
    observation_ring,
    # policy output
    selected_action_index,
    # output
    policy_actions_buffer,
    episode_buffer,
    aggregated_reward_buffer,
    # updated model
    optimizer,
    training_model,
    target_model,
):
    with torch.no_grad():
        # pull some constants from tensor dimensions
        ring_buffer_size, obs_channels, obs_height, obs_width = observation_ring.shape
        input_channels = input_stack * obs_channels

        num_actions_value = int(num_actions.item())
        frame_skip_val = frame_skip

        # see which samples from the replay buffer are being evaluated
        # Nature DQN samples uniformly from all valid experience tuples.
        filled_frames = min(int(filled_frames_tensor.item()), ring_buffer_size - 1)
        # need at least `input_stack` frames to build a state, and one extra for the next-state target
        min_index = input_stack
        max_index = max(filled_frames - 1, min_index)

        if max_index <= min_index:
            random_action = torch.randint(
                0, num_actions_value, (1,), device=selected_action_index.device, dtype=torch.int64
            )
            selected_action_index.copy_(random_action.squeeze(0))
            return

        block_min = max(0, (min_index - (frame_skip_val - 1) + frame_skip_val - 1) // frame_skip_val)
        block_max = (max_index - (frame_skip_val - 1)) // frame_skip_val
        if block_max < block_min:
            random_action = torch.randint(
                0, num_actions_value, (1,), device=selected_action_index.device, dtype=torch.int64
            )
            selected_action_index.copy_(random_action.squeeze(0))
            return
        blocks = torch.randint(
            block_min,
            block_max + 1,
            (train_batch,),
            device=observation_ring.device,
            dtype=torch.int64,
        )
        buffer_indexes = blocks * frame_skip_val + (frame_skip_val - 1)
        buffer_mod = torch.remainder(buffer_indexes, ring_buffer_size)
        # stack sets of input_stack frames together to make each observation
        final_stack_indexes = buffer_indexes.unsqueeze(dim=1)
        offsets = torch.arange(-input_stack + 1, 1, device='cpu', dtype=torch.int64).unsqueeze(dim=0)
        ring_indexes = final_stack_indexes + offsets

        # ensure indices wrap correctly within the ring buffer
        ring_indexes = torch.remainder(ring_indexes, ring_buffer_size)

        observation_stacks = observation_ring.index_select(
            0, ring_indexes.reshape(-1)
        ).view(train_batch, input_stack, obs_channels, obs_height, obs_width)

        # merge stacked frames into channel dimension as expected by the Q-network
        observation_stacks = observation_stacks.view(train_batch, input_channels, obs_height, obs_width)

    model_param = next(training_model.parameters())
    model_device = model_param.device
    model_dtype = model_param.dtype

    observation_stacks = observation_stacks.to(device=model_device, dtype=model_dtype).div_(255.0)

    # evaluate the model with gradients
    train_values = training_model(observation_stacks)

    with torch.no_grad():
        # build target values for training
        batch_q = train_values.detach()
        num_actions_value = batch_q.shape[1]

        # gather immediate rewards and next-state information (Nature DQN targets)
        next_buffer_mod = torch.remainder(buffer_indexes + frame_skip_val, ring_buffer_size)
        dones = episode_buffer.index_select(0, buffer_mod) != episode_buffer.index_select(0, next_buffer_mod)

        next_final_stack_indexes = (buffer_indexes + frame_skip_val).unsqueeze(dim=1)
        next_ring_indexes = next_final_stack_indexes + offsets
        next_ring_indexes = torch.remainder(next_ring_indexes, ring_buffer_size)

        next_observation_stacks = observation_ring.index_select(
            0, next_ring_indexes.reshape(-1)
        ).view(train_batch, input_stack, obs_channels, obs_height, obs_width)
        next_observation_stacks = next_observation_stacks.view(train_batch, input_channels, obs_height, obs_width)
        next_observation_stacks = next_observation_stacks.to(device=model_device, dtype=model_dtype).div_(255.0)

        actions = policy_actions_buffer.index_select(0, buffer_mod).to(device=model_device, dtype=torch.int64)

        # build targets out of the observed rewards and the bootstrap values
        rewards = aggregated_reward_buffer.index_select(0, buffer_mod).to(device=model_device, dtype=model_dtype)
        discount = discount.to(device=model_device, dtype=model_dtype)
        reward_clip = reward_clip.to(device=model_device, dtype=model_dtype)
        rewards = rewards.clamp(-reward_clip, reward_clip)

        dones = dones.to(device=model_device)
        next_q_values = target_model(next_observation_stacks).max(dim=1).values
        not_done = (~dones).to(device=model_device, dtype=model_dtype)
        targets = rewards + discount * not_done * next_q_values

    action_q = train_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(action_q, targets, reduction='mean')

    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        current_offsets = torch.arange(-input_stack + 1, 1, device='cpu', dtype=torch.int64)
        current_ring_indexes = torch.remainder(tensor_u + current_offsets, ring_buffer_size)
        current_state = observation_ring.index_select(0, current_ring_indexes).view(
            1, input_stack, obs_channels, obs_height, obs_width
        )
        current_state = current_state.view(1, input_channels, obs_height, obs_width).to(
            device=model_device, dtype=model_dtype
        ).div_(255.0)

        current_q = training_model(current_state)
        explore_act = torch.rand((), device=model_device) < epsilon_tensor
        greedy_act = current_q.argmax(dim=1)
        random_act = torch.randint(0, num_actions_value, (1,), device=model_device, dtype=torch.int64)
        chosen_action = torch.where(explore_act, random_act.squeeze(0), greedy_act.squeeze(0))
        selected_action_index.copy_(chosen_action)

class QNetwork(nn.Module):
    """
    Convolutional network matching the architecture from
    "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013).
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_features = self._conv_output_size(in_channels)

        self.fc1 = nn.Linear(conv_out_features, 512)
        self.fc2 = nn.Linear(512, out_channels)

        self._reset_parameters()

    def _conv_output_size(self, in_channels):
        """
        Compute flattened size after the convolutional stack for an 84x84 input.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x.view(1, -1).size(1)

    def _reset_parameters(self):
        """
        Initialize weights with the fan-in uniform scheme used in the original DQN.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.Linear):
                fan_in = layer.in_features
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class cuda_graph_wrapper:
    def __init__(self, func, stream, use_cuda_graphs, args):
        self.func = func
        self.stream = stream
        self.use_cuda_graphs = use_cuda_graphs
        self.args = args
        self.cuda_graph = None

    def __call__(self):
        if not self.use_cuda_graphs:
            self.func(*self.args)
            return

        if self.cuda_graph is None:
            torch.cuda.synchronize()
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph, stream=self.stream):
                self.func(*self.args)
            torch.cuda.synchronize()
        else:
            self.stream.wait_stream(torch.cuda.current_stream())
            self.cuda_graph.replay()


class Agent:
    def __init__(self, data_dir, seed, num_actions, total_frames, **kwargs):
        # defaults that might be overridden by explicit experiment runs
        self.gpu = 0

        # The observation
        self.frame_skip = 4  # the number of observations, rewards, and end_of_episodes processed each call
        self.input_width = 160  # image dimensions provided to the agent
        self.input_height = 210
        self.input_stack = 4  # number of previous 60 fps frames to stack for the input
        self.obs_width = 84  # image dimensions provided to the model
        self.obs_height = 84
        self.obs_channels = 1  # processed grayscale channels
        self.input_obs_channels = 3  # raw RGB channels from ALE

        # exploration (Nature DQN schedule)
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay_frames = 1_000_000
        self.epsilon = self.eps_start

        # The model
        self.load_file = None
        self.seed = seed
        self.num_actions = num_actions  # many games can use a reduced action set for faster learning

        # training
        self.lr = 0.00025
        self.gamma = 0.99
        self.reward_clip = 1.0

        self.train_batch = (
            32  # One will be the most current data, the rest will be randomly sampled from the ring buffer
        )

        self.ring_buffer_size = 1000000

        # number of frames to collect before starting gradient updates
        self.learn_start_frames = 50000

        self.total_frames = total_frames
        self._apply_short_run_overrides()

        # should be strictly a performance optimization, with no behavior change
        self.use_cuda_graphs = False  # disable graphs by default; current ring-buffer update uses capture-unsafe ops

        # dynamically override configuration
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)

        self.dev = f'cuda:{self.gpu}'

        # force the ring buffer to be an exact multiple of frame_skip
        self.ring_buffer_size -= self.ring_buffer_size % self.frame_skip

        # set CuBLAS environment variable so matmuls can be deterministic
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # helps debugging cuda issues
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Make more deterministic
        # must be combined with os.environ['CUBLAS_WORKSPACE_CONFIG']= ':4096:8' before loading torch
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        print('torch version: ', torch.__version__)
        print('cuda version : ', torch.version.cuda)
        print('dev          : ', self.dev)

        # if this isn't done, pytorch seems to use 3-9 cores worth of time per process
        # just to busy wait, such that trying to run 8 processes was getting CPU bound.
        torch.set_num_threads(1)

        # Which GPU everything will be done on.
        torch.set_default_device(self.dev)

        # This is necessary for graph compilation to work on devices other than 0
        torch.cuda.set_device(self.dev)

        # let pytorch print wider tensor dumps
        torch.set_printoptions(linewidth=160)

        self.input_channels = self.input_stack * self.obs_channels

        self.epsilon_tensor = torch.tensor(self.epsilon, dtype=torch.float32, device=self.dev)

        # the environment returns a current observation and episode number, and the reward from the just-executed action,
        # which may have been the end of a different episode.
        self.observation_ring = torch.zeros(
            self.ring_buffer_size, self.obs_channels, self.obs_height, self.obs_width, dtype=torch.uint8, device='cpu'
        )

        # start at -1 so the first processed block lands at indices 0..frame_skip-1
        self.u = -1
        self.tensor_u = torch.tensor(-1, dtype=torch.int64, device='cpu')
        self.filled_frames_tensor = torch.tensor(0, dtype=torch.int64, device='cpu')

        # buffer frame_skip frames for train
        self.frame_count = 0
        self.observation_rgb8 = np.zeros(
            (self.frame_skip, self.input_height, self.input_width, self.input_obs_channels), dtype=np.uint8
        )
        self.prev_raw_frame = np.zeros(
            (self.input_height, self.input_width, self.input_obs_channels), dtype=np.uint8
        )
        self.rewards = np.zeros(self.frame_skip)
        self.end_of_episodes = np.zeros(self.frame_skip)

        # total number of frames pushed into replay so far
        self.stored_frames = 0

        # set by policy for environment
        self.selected_action_index = torch.tensor(0, dtype=torch.int64)
        self.current_action = 0


        self.episode_buffer = torch.full(
            (self.ring_buffer_size,), -999, dtype=torch.int64, device='cpu'
        )
        self.aggregated_reward_buffer = torch.zeros(self.ring_buffer_size, device='cpu')

        # Every frame is assigned the most recent selected_action_index.
        # With the standard frame_skip 4, each selected_action_index will be repeated 4 times.
        self.policy_actions_buffer = torch.full(
            (self.ring_buffer_size,), -999, dtype=torch.int64, device='cpu'
        )

        # Random seed for policy action selection and training index selection
        torch.random.manual_seed(self.seed)

        # epsilon-greedy random action exploration, -1 = take best from policy, otherwise use this random index
        self.episode_number = 0
        fmt = torch.float32
        self.training_model = QNetwork(self.input_channels, None, self.num_actions)
        if self.load_file is not None:
            self.training_model.load_state_dict(torch.load(self.load_file, weights_only=True))

        self.training_model.to(dtype=fmt)
        self.training_model.train()
        self.target_model = copy.deepcopy(self.training_model)
        self.target_model.to(dtype=fmt)
        self.target_model.eval()
        self.target_update_freq = 10_000
        self.train_iterations = 0

        self.optimizer = torch.optim.RMSprop(
            self.training_model.parameters(),
            lr=self.lr,
            alpha=0.95,
            eps=0.01,
            centered=False,
        )

        self.spin_stream = torch.cuda.Stream(priority=0)

        self.train_stream = torch.cuda.Stream(priority=0)
        self.train_graph = cuda_graph_wrapper(
            train_function,
            self.train_stream,
            self.use_cuda_graphs,
            [
                # constants
                self.input_stack,
                self.frame_skip,
                self.train_batch,
                self.epsilon_tensor,
                torch.tensor(self.gamma, device=self.dev),
                torch.tensor(self.reward_clip, device=self.dev),
                torch.tensor(self.num_actions, device=self.dev, dtype=torch.int64),
                # variable state
                self.tensor_u,
                self.filled_frames_tensor,
                self.observation_ring,
                # policy output
                self.selected_action_index,
                # output
                self.policy_actions_buffer,
                self.episode_buffer,
                self.aggregated_reward_buffer,
                # updated models
                self.optimizer,
                self.training_model,
                self.target_model,
            ],
        )

    def _apply_short_run_overrides(self):
        """
        For short training runs (<=200k frames), start learning sooner and anneal epsilon faster.
        """
        if self.total_frames <= 200_000:
            short_eps_decay = max(self.total_frames // 2, 10_000)
            self.eps_decay_frames = min(self.eps_decay_frames, short_eps_decay)
            self.learn_start_frames = min(self.learn_start_frames, 10_000)

    def _preprocess_recent_observations(self):
        """
        Convert the most recent raw RGB frames into the grayscale, resized format expected by the ring buffer.
        """
        with torch.no_grad():
            frames = torch.from_numpy(self.observation_rgb8).permute(0, 3, 1, 2).to(dtype=torch.float32)
            if frames.shape[1] == 3 and self.obs_channels == 1:
                r = frames[:, 0, :, :]
                g = frames[:, 1, :, :]
                b = frames[:, 2, :, :]
                frames = 0.299 * r + 0.587 * g + 0.114 * b
                frames = frames.unsqueeze(1)
            frames = F.interpolate(
                frames, size=(self.obs_height, self.obs_width), mode='bilinear', align_corners=False
            )
            frames = frames.clamp_(min=0.0, max=255.0).to(dtype=torch.uint8)
            return frames

    # --------------------------------
    # Returns the selected action index
    # --------------------------------
    def frame(self, observation_rgb8, reward, end_of_episode):  # [height,width,3]
        expected_shape = (self.input_height, self.input_width, self.input_obs_channels)
        assert observation_rgb8.shape == expected_shape, f'got {observation_rgb8.shape}, expected {expected_shape}'

        i = self.frame_count % self.frame_skip
        observation_rgb8 = np.asarray(observation_rgb8, dtype=np.uint8)
        np.maximum(self.prev_raw_frame, observation_rgb8, out=self.observation_rgb8[i])
        np.copyto(self.prev_raw_frame, observation_rgb8)
        self.rewards[i] = reward
        self.end_of_episodes[i] = end_of_episode
        self.frame_count += 1

        if i != (self.frame_skip - 1):
            return self.selected_action_index.item()

        if self.u > self.total_frames - self.frame_skip:
            # don't overflow any of the buffers
            return 0

        end_flags = self.end_of_episodes > 0
        if np.any(end_flags):
            terminal_offset = int(np.argmax(end_flags))
            block_done = True
        else:
            terminal_offset = self.frame_skip - 1
            block_done = False

        valid_rewards = self.rewards[: terminal_offset + 1]
        block_reward = float(valid_rewards.sum())
        if self.reward_clip > 0.0:
            block_reward = max(-self.reward_clip, min(self.reward_clip, block_reward))

        if block_done and terminal_offset < self.frame_skip - 1:
            terminal_frame = self.observation_rgb8[terminal_offset].copy()
            for idx in range(terminal_offset + 1, self.frame_skip):
                np.copyto(self.observation_rgb8[idx], terminal_frame)
                self.rewards[idx] = 0.0
                self.end_of_episodes[idx] = 0.0

        current_episode_number = self.episode_number

        with torch.cuda.stream(self.spin_stream):
            processed_frames = self._preprocess_recent_observations()

            self.u += self.frame_skip

            frame_range = torch.arange(
                self.u - self.frame_skip + 1, self.u + 1, device='cpu', dtype=torch.int64
            )
            frame_mod = torch.remainder(frame_range, self.ring_buffer_size)
            self.observation_ring.index_copy_(0, frame_mod, processed_frames)
            episode_values = torch.full(
                (self.frame_skip,), current_episode_number, dtype=torch.int64, device='cpu'
            )
            self.episode_buffer.index_copy_(0, frame_mod, episode_values)
            action_values = torch.full(
                (self.frame_skip,), self.current_action, dtype=torch.int64, device='cpu'
            )
            self.policy_actions_buffer.index_copy_(0, frame_mod, action_values)
            write_index = frame_mod[-1]
            self.aggregated_reward_buffer[write_index] = block_reward

        if block_done:
            self.episode_number = current_episode_number + 1

        self.stored_frames += self.frame_skip
        filled_frames = min(self.stored_frames, self.ring_buffer_size)
        self.filled_frames_tensor.fill_(filled_frames)

        decay_progress = min(self.stored_frames / self.eps_decay_frames, 1.0)
        self.epsilon = self.eps_start + (self.eps_end - self.eps_start) * decay_progress
        self.epsilon_tensor.fill_(self.epsilon)

        if self.stored_frames < self.learn_start_frames:
            action = torch.randint(
                low=0, high=self.num_actions, size=(1,), device=self.selected_action_index.device
            ).item()
            self.selected_action_index.fill_(action)
            self.current_action = action
            return action
        elif self.stored_frames == self.learn_start_frames:
            print(f'[Agent] starting training after {self.learn_start_frames} frames')

        # run the policy every four frames
        # sets selected_action_index, which will be returned to the environment,
        # and various things for the training to use
        self.selected_action_index.fill_(-1)
        self.tensor_u.fill_(self.u)
        self.train_graph()
        self.train_iterations += 1
        if self.train_iterations % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.training_model.state_dict())
            self.target_model.eval()
        self.train_stream.synchronize()

        chosen_action = self.selected_action_index.item()
        self.current_action = chosen_action
        return chosen_action

    def save_model(self, filename):
        torch.save(self.training_model.state_dict(), filename)

def run_training_frames(
    agent,
    ale,
    action_set,
    *,
    name,
    data_dir,
    rank,
    delay_frames,
    average_frames,
    max_frames_without_reward,
    lives_as_episodes,
    save_incremental_models,
    last_model_save,
    frame_budget,
    frame_offset=0,
    graph_total_frames=None,
):
    episode_scores = []
    episode_end = []

    noop_start_max = 30
    noop_action_idx = None
    noop_action_value = int(Action.NOOP)
    for idx, candidate in enumerate(action_set):
        if int(candidate) == noop_action_value:
            noop_action_idx = idx
            break

    noop_rng = np.random.RandomState(agent.seed)

    def sample_noop_count():
        if noop_action_idx is None:
            return 0
        return int(noop_rng.randint(0, noop_start_max + 1))

    environment_start = frame_offset
    running_episode_score = 0
    environment_start_time = time.time()

    episode_graph = torch.zeros(1000, device='cpu')
    parms_graph = torch.zeros(1000, len(list(agent.training_model.parameters())))

    episode_number = 0
    frames_without_reward = 0
    previous_lives = ale.lives()
    delayed_actions = [0] * delay_frames
    pending_noop_steps = sample_noop_count()
    warned_missing_noop = False

    taken_action = 0
    avg = 0

    if graph_total_frames is None:
        graph_total_frames = frame_offset + frame_budget
    else:
        graph_total_frames = max(graph_total_frames, frame_offset + frame_budget)

    for local_frame in range(frame_budget):
        global_frame = frame_offset + local_frame
        global_next = global_frame + 1

        if save_incremental_models and global_next // 500_000 != last_model_save:
            last_model_save = global_next // 500_000
            filename = f'{data_dir}/{name}_{global_next}.model'
            print('writing ' + filename)
            agent.save_model(filename)

        points = episode_graph.shape[0]
        if (
            global_frame * points // graph_total_frames
            != global_next * points // graph_total_frames
        ):
            torch.cuda.synchronize()
            i = global_frame * points // graph_total_frames
            i = min(i, points - 1)
            count = 0
            total = 0
            for j in range(len(episode_scores) - 1, -1, -1):
                if episode_end[j] < global_frame - average_frames:
                    break
                count += 1
                total += episode_scores[j]
            if count == 0:
                avg = -999
            else:
                avg = total / count
                for j in range(i - 1, -1, -1):
                    if episode_graph[j] != -999:
                        break
                    episode_graph[j] = avg
            episode_graph[i] = avg

            filename = data_dir + '/' + name + '.score'
            episode_graph.cpu().numpy().tofile(filename)

            for j, p in enumerate(agent.training_model.parameters()):
                parms_graph[i, j] = torch.norm(p.flatten()).item()

        delayed_actions.append(taken_action)

        torch.cuda.nvtx.range_push("act")
        cmd = delayed_actions.pop(0)
        executed_action_idx = cmd
        executed_action = action_set[cmd]
        forced_noop = pending_noop_steps > 0 and noop_action_idx is not None
        if forced_noop:
            pending_noop_steps -= 1
            executed_action_idx = noop_action_idx
            executed_action = Action.NOOP
            delayed_actions.insert(0, cmd)
        reward = ale.act(int(executed_action))
        running_episode_score += reward
        torch.cuda.nvtx.range_pop()
        if reward != 0:
            frames_without_reward = 0
        else:
            frames_without_reward += 1

        end_of_episode = 0

        if lives_as_episodes and ale.lives() < previous_lives:
            previous_lives = ale.lives()
            episode_number += 1
            end_of_episode = 1
        if ale.game_over() or frames_without_reward == max_frames_without_reward:
            torch.cuda.synchronize()
            if frames_without_reward == max_frames_without_reward:
                print(f'terminated at {frames_without_reward} frames without reward')
            episode_number = ((episode_number // 100) + 1) * 100
            end_of_episode = 1
            torch.cuda.nvtx.range_push("reset")
            ale.reset_game()
            previous_lives = ale.lives()
            frames_without_reward = 0
            if noop_action_idx is not None:
                pending_noop_steps = sample_noop_count()
            elif not warned_missing_noop:
                print('Warning: Action.NOOP not available in action set; skipping no-op starts.')
                warned_missing_noop = True

            frames = global_frame - environment_start
            episode_end.append(global_frame)
            environment_start = global_frame
            episode_scores.append(running_episode_score)
            running_episode_score = 0

            now = time.time()
            frames_per_second = frames / (now - environment_start_time)
            environment_start_time = now

            print(
                f'{rank}:{name} frame:{global_frame:7} {frames_per_second:4.0f}/s '
                f'eps {len(episode_scores) - 1:3},{frames:5}={int(episode_scores[-1]):5} '
                f'avg {avg:4.1f}'
            )

            torch.cuda.nvtx.range_pop()

        agent.current_action = executed_action_idx
        taken_action = agent.frame(ale.getScreenRGB(), reward, end_of_episode)

    return last_model_save, episode_scores, episode_end, episode_graph, parms_graph


def main():
    data_dir = './results/my_dqn_for_real_200k'
    os.makedirs(data_dir, exist_ok=True)

    save_model = False
    save_incremental_models = False
    max_frames_without_reward = 18_000
    lives_as_episodes = 1

    allowed_modes = {'atari100k', 'physical', 'continual'}

    parser = argparse.ArgumentParser(
        description='Run the Physical Atari agent in simulator or continual benchmark modes.',
        allow_abbrev=False,
    )
    parser.add_argument(
        '--cycle_frames',
        type=int,
        default=150_000,
        help='Total number of frames to allocate per cycle when running the continual benchmark.',
    )
    parser.add_argument(
        '--game_frame_budgets',
        type=str,
        help='Comma-separated per-game frame budgets for continual mode. Must sum to --cycle_frames.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Override the base random seed used for the agent and environment.',
    )
    args, positional = parser.parse_known_args()

    positional_args = list(positional)
    mode = None
    if positional_args and positional_args[-1] in allowed_modes:
        mode = positional_args.pop()

    rank = 0
    if positional_args:
        try:
            rank = int(positional_args.pop(0))
        except ValueError as exc:
            raise ValueError(
                f"Could not parse rank from arguments: {' '.join(positional)}"
            ) from exc

    if positional_args:
        raise ValueError(f"Unrecognized positional arguments: {' '.join(positional_args)}")

    parms = {'gpu': rank % 8}
    frame_skip = 4
    parms['frame_skip'] = frame_skip
    seed_override = args.seed

    if mode == 'continual':
        average_frames = 100_000
        seed = seed_override if seed_override is not None else rank

        game_order = [
            'ms_pacman'
        ]
        reduce_action_setting = 2
        cycle_frames = 50_000
        if cycle_frames <= 0:
            raise ValueError('--cycle_frames must be a positive integer')

        provided_budgets: Optional[List[int]] = None
        if args.game_frame_budgets is not None:
            try:
                provided_budgets = [int(value.strip()) for value in args.game_frame_budgets.split(',') if value.strip()]
            except ValueError as exc:
                raise ValueError('--game_frame_budgets must be a comma-separated list of integers') from exc

            if len(provided_budgets) != len(game_order):
                raise ValueError(
                    f'--game_frame_budgets expected {len(game_order)} values, received {len(provided_budgets)}'
                )
            if any(budget < 0 for budget in provided_budgets):
                raise ValueError('--game_frame_budgets values must be non-negative integers')
            if sum(provided_budgets) != cycle_frames:
                raise ValueError('--game_frame_budgets values must sum to --cycle_frames')

        if provided_budgets is None:
            if cycle_frames % len(game_order) != 0:
                raise ValueError(
                    f'--cycle_frames ({cycle_frames}) must be divisible by the number of games ({len(game_order)}) '
                    'when --game_frame_budgets is not provided.'
                )
            frame_budget = cycle_frames // len(game_order)
            game_frame_budgets = [frame_budget] * len(game_order)
        else:
            game_frame_budgets = provided_budgets

        cycles = []
        for cycle_index in range(1):
            cycle_games = []
            for game_index, game_name in enumerate(game_order):
                cycle_games.append(
                    GameSpec(
                        name=game_name,
                        frame_budget=game_frame_budgets[game_index],
                        sticky_prob=0.0,
                        delay_frames=6,
                        seed=seed + cycle_index,
                        params={'reduce_action_set': reduce_action_setting},
                    )
                )
            cycles.append(CycleConfig(cycle_index=cycle_index, games=cycle_games))

        action_set_sizes = set()
        for cycle in cycles:
            for spec in cycle.games:
                preview_ale = ALEInterface()
                preview_ale.setLoggerMode(LoggerMode.Error)
                preview_ale.setInt('random_seed', spec.seed if spec.seed is not None else seed)
                preview_ale.setFloat('repeat_action_probability', spec.sticky_prob)
                rom_path = roms.get_rom_path(spec.name)
                preview_ale.loadROM(rom_path)
                reduce_action = spec.params.get('reduce_action_set', reduce_action_setting)
                if reduce_action == 0:
                    preview_actions = preview_ale.getLegalActionSet()
                elif reduce_action == 2 and spec.name in {'ms_pacman', 'qbert'}:
                    preview_actions = [Action.NOOP, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
                else:
                    preview_actions = preview_ale.getMinimalActionSet()
                action_set_sizes.add(len(preview_actions))
        if len(action_set_sizes) != 1:
            raise ValueError(
                f'Continual benchmark configuration requires a consistent action-set size; found {sorted(action_set_sizes)}'
            )
        num_actions = action_set_sizes.pop()

        benchmark_config = BenchmarkConfig(
            cycles=cycles,
            description=(
                f'Continual benchmark: {len(cycles)} cycle(s) x {len(game_order)} game(s) '
                f'(cycle frames={cycle_frames})'
            ),
        )

        total_frames = benchmark_config.total_frames

        agent = Agent(data_dir, seed, num_actions, total_frames, **parms)

        runner = BenchmarkRunner(
            agent,
            benchmark_config,
            frame_runner=run_training_frames,
            data_dir=data_dir,
            rank=rank,
            default_seed=seed,
            reduce_action_set=reduce_action_setting,
            use_canonical_full_actions=False,
            average_frames=average_frames,
            max_frames_without_reward=max_frames_without_reward,
            lives_as_episodes=lives_as_episodes,
            save_incremental_models=save_incremental_models,
        )
        results = runner.run()

        summary_records = []
        for result in results:
            total_score = float(sum(result.episode_scores))
            mean_score = float(np.mean(result.episode_scores)) if result.episode_scores else 0.0
            summary_records.append(
                {
                    'cycle_index': result.cycle_index,
                    'game_index': result.game_index,
                    'game': result.spec.name,
                    'frame_offset': result.frame_offset,
                    'frame_budget': result.frame_budget,
                    'episodes': len(result.episode_scores),
                    'total_episode_score': total_score,
                    'mean_episode_score': mean_score,
                }
            )

        final_cycle_index = benchmark_config.cycles[-1].cycle_index if benchmark_config.cycles else -1
        final_cycle_total = sum(
            record['total_episode_score']
            for record in summary_records
            if record['cycle_index'] == final_cycle_index
        )

        summary_path = os.path.join(data_dir, 'continual_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'description': benchmark_config.description,
                    'final_cycle_index': final_cycle_index,
                    'final_cycle_total_episode_score': final_cycle_total,
                    'records': summary_records,
                },
                f,
                indent=2,
            )

        wrote_loss = False
        for result in results:
            base = os.path.join(data_dir, result.name)

            score_path = base + '.score'
            result.episode_graph.cpu().numpy().tofile(score_path)

            parms_path = base + '.parms'
            result.parms_graph.cpu().numpy().tofile(parms_path)

            if result.episode_scores:
                plots = torch.zeros(len(result.episode_scores), 2)
                for i, (frame_idx, score) in enumerate(zip(result.episode_end, result.episode_scores)):
                    plots[i][0] = frame_idx
                    plots[i][1] = score
                scatter_path = base + '.scatter'
                plots.cpu().numpy().tofile(scatter_path)

            start = result.frame_offset
            end = start + result.frame_budget
            if hasattr(agent, 'policy_actions_buffer'):
                policy_slice = agent.policy_actions_buffer[start:end]
                policy_path = base + '.policy_actions'
                policy_slice.cpu().numpy().tofile(policy_path)

        print(f'Final cycle ({final_cycle_index}) total episode score: {final_cycle_total:.1f}')
        print(f'Wrote summary to {summary_path}')

        if save_model:
            filename = f'{data_dir}/continual_final.model'
            print('writing ' + filename)
            agent.save_model(filename)

        print('done')
        return

    # fallback to legacy single-game runs
    last_model_save = -1

    ale = ALEInterface()
    ale.setLoggerMode(LoggerMode.Error)

    if mode == 'atari100k':
        atari100k_list = [
            'assault',
            'asterix',
            'bank_heist',
            'battle_zone',
            'boxing',
            'breakout',
            'chopper_command',
            'crazy_climber',
            'demon_attack',
            'freeway',
            'frostbite',
            'gopher',
            'hero',
            'jamesbond',
            'kangaroo',
            'krull',
            'kung_fu_master',
            'ms_pacman',
            'pong',
            'private_eye',
            'qbert',
            'road_runner',
            'seaquest',
            'up_n_down',
        ]
        game = atari100k_list[rank % 24]
        seed = seed_override if seed_override is not None else rank // 24
        total_frames = 1_000_000

        ale.setFloat('repeat_action_probability', 0.0)
        reduce_action_set = 1
        delay_frames = 0
    elif mode == 'physical':
        physical_list = ['centipede', 'up_n_down', 'qbert', 'battle_zone', 'krull', 'defender', 'ms_pacman', 'atlantis']
        game = physical_list[rank % 8]
        total_frames = 20_000_000
        seed = seed_override if seed_override is not None else (rank // 8) % 4

        reduce_action_set = 2
        delay_frames = 6
    else:
        reduce_action_set = 2
        total_frames = 200_000
        seed = seed_override if seed_override is not None else 0
        game = 'pong'

        delay_frames = 0

    ale.setInt('random_seed', seed)
    rom_path = roms.get_rom_path(game)
    ale.loadROM(rom_path)
    ale.reset_game()

    if reduce_action_set == 0:
        action_set = ale.getLegalActionSet()
    else:
        if reduce_action_set == 2 and (game == 'ms_pacman' or game == 'qbert'):
            action_set = [Action.NOOP, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        else:
            action_set = ale.getMinimalActionSet()
    num_actions = len(action_set)
    print(f'{num_actions} actions: {action_set}')

    name = f'delay_{game}{delay_frames}'
    for k, v in parms.items():
        if k != 'gpu':
            name += '_'
            name += str(v)
    print(name)

    agent = Agent(data_dir, seed, num_actions, total_frames, **parms)

    average_frames = 100_000

    (
        last_model_save,
        episode_scores,
        episode_end,
        episode_graph,
        parms_graph,
    ) = run_training_frames(
        agent,
        ale,
        action_set,
        name=name,
        data_dir=data_dir,
        rank=rank,
        delay_frames=delay_frames,
        average_frames=average_frames,
        max_frames_without_reward=max_frames_without_reward,
        lives_as_episodes=lives_as_episodes,
        save_incremental_models=save_incremental_models,
        last_model_save=last_model_save,
        frame_budget=agent.total_frames,
        frame_offset=0,
        graph_total_frames=agent.total_frames,
    )

    filename = data_dir + '/' + name + '.policy_actions'
    print('writing ' + filename)
    agent.policy_actions_buffer.cpu().numpy().tofile(filename)

    filename = data_dir + '/' + name + '.score'
    print('writing ' + filename)
    episode_graph.cpu().numpy().tofile(filename)

    filename = data_dir + '/' + name + '.parms'
    print('writing ' + filename)
    parms_graph.cpu().numpy().tofile(filename)

    plots = torch.zeros(len(episode_scores), 2)
    for i in range(len(episode_scores)):
        plots[i][0] = episode_end[i]
        plots[i][1] = episode_scores[i]
    filename = data_dir + '/' + name + '.scatter'
    print('writing ' + filename)
    plots.cpu().numpy().tofile(filename)

    if save_model:
        filename = f'{data_dir}/{name}.model'
        print('writing ' + filename)
        agent.save_model(filename)

    print('done')


if __name__ == '__main__':
    main()
 