# coding=utf-8
# BBF Network Architectures - PyTorch port from JAX implementation
# Based on: https://github.com/google-research/google-research/tree/master/bigger_better_faster

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImpalaCNNBlock(nn.Module):
    """Impala CNN residual block."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + residual


class ImpalaEncoder(nn.Module):
    """Impala CNN encoder with configurable width scaling.
    
    Based on IMPALA architecture from Espeholt et al. 2018.
    BBF uses width_scale=4 and num_blocks=2.
    """
    
    def __init__(self, in_channels=4, width_scale=4, num_blocks=2):
        super().__init__()
        self.width_scale = width_scale
        self.num_blocks = num_blocks
        
        # Base channel multipliers for each stage
        base_channels = [16, 32, 32]
        channels = [int(c * width_scale) for c in base_channels]
        
        # Build encoder stages
        self.stages = nn.ModuleList()
        
        # Stage 1: Input -> channels[0]
        stage1_layers = [nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)]
        stage1_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for _ in range(num_blocks):
            stage1_layers.append(ImpalaCNNBlock(channels[0], channels[0]))
        self.stages.append(nn.Sequential(*stage1_layers))
        
        # Stage 2: channels[0] -> channels[1]
        stage2_layers = [nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)]
        stage2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for _ in range(num_blocks):
            stage2_layers.append(ImpalaCNNBlock(channels[1], channels[1]))
        self.stages.append(nn.Sequential(*stage2_layers))
        
        # Stage 3: channels[1] -> channels[2]
        stage3_layers = [nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)]
        stage3_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for _ in range(num_blocks):
            stage3_layers.append(ImpalaCNNBlock(channels[2], channels[2]))
        self.stages.append(nn.Sequential(*stage3_layers))
        
        # Final ReLU
        self.final_relu = nn.ReLU()
        
        # Output dimension: channels[2] * 11 * 11 for 84x84 input
        self.output_dim = channels[2] * 11 * 11
        
    def forward(self, x):
        """Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Flattened encoding
        """
        for stage in self.stages:
            x = stage(x)
        x = self.final_relu(x)
        return x.reshape(x.size(0), -1)


class TransitionModel(nn.Module):
    """SPR Transition model for predicting future representations."""
    
    def __init__(self, latent_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        
        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, latent_dim)
        
        # Transition network
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)
        
    def forward(self, latent, action):
        """Predict next latent state given current latent and action.
        
        Args:
            latent: Current latent state (batch, latent_dim)
            action: Action taken (batch,)
            
        Returns:
            Next latent state (batch, latent_dim)
        """
        action_emb = self.action_embedding(action)
        x = torch.cat([latent, action_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(latent + x)  # Residual connection
        return x


class ProjectionHead(nn.Module):
    """Projection head for SPR self-supervised learning."""
    
    def __init__(self, input_dim, hidden_dim=2048, output_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x)
        return x


class BBFNetwork(nn.Module):
    """Complete BBF network with encoder, transition model, and Q-head.
    
    This implements the full BBF architecture:
    - Impala CNN encoder (width_scale=4)
    - Transition model for SPR
    - Dueling Q-network head with C51 distributional RL
    - Projection heads for self-supervised learning
    """
    
    def __init__(
        self,
        in_channels=4,
        num_actions=18,
        num_atoms=51,
        width_scale=4,
        hidden_dim=2048,
        dueling=True,
        distributional=True,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.dueling = dueling
        self.distributional = distributional
        
        # Encoder
        self.encoder = ImpalaEncoder(
            in_channels=in_channels,
            width_scale=width_scale,
            num_blocks=2
        )
        
        latent_dim = self.encoder.output_dim
        
        # Transition model for SPR
        self.transition_model = TransitionModel(
            latent_dim=latent_dim,
            num_actions=num_actions,
            hidden_dim=256
        )
        
        # Projection heads for SPR
        self.projection = ProjectionHead(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=256
        )
        self.prediction = ProjectionHead(
            input_dim=256,
            hidden_dim=hidden_dim,
            output_dim=256
        )
        
        # Q-value head
        if dueling:
            # Dueling architecture: separate value and advantage streams
            if distributional:
                self.value_fc = nn.Linear(latent_dim, hidden_dim)
                self.value_head = nn.Linear(hidden_dim, num_atoms)
                self.advantage_fc = nn.Linear(latent_dim, hidden_dim)
                self.advantage_head = nn.Linear(hidden_dim, num_actions * num_atoms)
            else:
                self.value_fc = nn.Linear(latent_dim, hidden_dim)
                self.value_head = nn.Linear(hidden_dim, 1)
                self.advantage_fc = nn.Linear(latent_dim, hidden_dim)
                self.advantage_head = nn.Linear(hidden_dim, num_actions)
        else:
            # Standard DQN head
            if distributional:
                self.q_fc = nn.Linear(latent_dim, hidden_dim)
                self.q_head = nn.Linear(hidden_dim, num_actions * num_atoms)
            else:
                self.q_fc = nn.Linear(latent_dim, hidden_dim)
                self.q_head = nn.Linear(hidden_dim, num_actions)
                
    def forward(self, x, return_latent=False, return_projection=False):
        """Forward pass through the network.
        
        Args:
            x: Input images (batch, channels, height, width)
            return_latent: Whether to return latent representations
            return_projection: Whether to return projected representations
            
        Returns:
            q_values or logits, and optionally latent/projection
        """
        # Encode input
        latent = self.encoder(x)
        
        # Q-values/logits
        if self.dueling:
            value = F.relu(self.value_fc(latent))
            value = self.value_head(value)
            
            advantage = F.relu(self.advantage_fc(latent))
            advantage = self.advantage_head(advantage)
            
            if self.distributional:
                value = value.view(-1, 1, self.num_atoms)
                advantage = advantage.view(-1, self.num_actions, self.num_atoms)
                q_logits = value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                q_logits = q_values
        else:
            q = F.relu(self.q_fc(latent))
            q_logits = self.q_head(q)
            if self.distributional:
                q_logits = q_logits.view(-1, self.num_actions, self.num_atoms)
        
        outputs = {'q_logits': q_logits}
        
        if return_latent:
            outputs['latent'] = latent
            
        if return_projection:
            projection = self.projection(latent)
            outputs['projection'] = projection
            
        return outputs
    
    def spr_rollout(self, latent, actions):
        """Perform SPR rollout: predict future latent states.
        
        Args:
            latent: Initial latent state (batch, latent_dim)
            actions: Action sequence (batch, num_steps)
            
        Returns:
            Sequence of projected future latents (batch, num_steps, proj_dim)
        """
        batch_size, num_steps = actions.shape
        projected_latents = []
        
        current_latent = latent
        for t in range(num_steps):
            # Project current latent
            proj = self.projection(current_latent)
            proj = self.prediction(proj)
            projected_latents.append(proj)
            
            # Transition to next latent
            if t < num_steps - 1:
                current_latent = self.transition_model(current_latent, actions[:, t])
        
        return torch.stack(projected_latents, dim=1)

