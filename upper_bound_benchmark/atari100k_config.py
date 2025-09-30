"""Atari 100K benchmark configuration for validating BBF implementation.

This configuration matches the Atari 100K benchmark from the BBF paper:
- 100,000 environment steps (frames with frameskip=1)
- 26 games
- No sticky actions (for initial validation)
- Single cycle (no multi-game curriculum)
"""

# Atari 100K games (26 games from Kaiser et al. 2020)
ATARI_100K_GAMES = [
    "ALE/Alien-v5",
    "ALE/Amidar-v5",
    "ALE/Assault-v5",
    "ALE/Asterix-v5",
    "ALE/Atlantis-v5",  # We'll validate on this one first
    "ALE/BankHeist-v5",
    "ALE/BattleZone-v5",
    "ALE/Boxing-v5",
    "ALE/Breakout-v5",
    "ALE/Centipede-v5",
    "ALE/ChopperCommand-v5",
    "ALE/CrazyClimber-v5",
    "ALE/DemonAttack-v5",
    "ALE/Freeway-v5",
    "ALE/Frostbite-v5",
    "ALE/Gopher-v5",
    "ALE/Hero-v5",
    "ALE/Jamesbond-v5",
    "ALE/Kangaroo-v5",
    "ALE/Krull-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MsPacman-v5",
    "ALE/Pong-v5",
    "ALE/PrivateEye-v5",
    "ALE/Qbert-v5",
    "ALE/RoadRunner-v5",
    "ALE/Seaquest-v5",
    "ALE/UpNDown-v5",
]

# Human and random scores for normalization (from Atari 100K benchmark)
HUMAN_SCORES = {
    "ALE/Alien-v5": 7127.7,
    "ALE/Amidar-v5": 1719.5,
    "ALE/Assault-v5": 742.0,
    "ALE/Asterix-v5": 8503.3,
    "ALE/Atlantis-v5": 29028.1,
    "ALE/BankHeist-v5": 753.1,
    "ALE/BattleZone-v5": 37187.5,
    "ALE/Boxing-v5": 12.1,
    "ALE/Breakout-v5": 30.5,
    "ALE/Centipede-v5": 12017.0,
    "ALE/ChopperCommand-v5": 7387.8,
    "ALE/CrazyClimber-v5": 35829.4,
    "ALE/DemonAttack-v5": 1971.0,
    "ALE/Freeway-v5": 29.6,
    "ALE/Frostbite-v5": 4334.7,
    "ALE/Gopher-v5": 2412.5,
    "ALE/Hero-v5": 30826.4,
    "ALE/Jamesbond-v5": 302.8,
    "ALE/Kangaroo-v5": 3035.0,
    "ALE/Krull-v5": 2665.5,
    "ALE/KungFuMaster-v5": 22736.3,
    "ALE/MsPacman-v5": 6951.6,
    "ALE/Pong-v5": 14.6,
    "ALE/PrivateEye-v5": 69571.3,
    "ALE/Qbert-v5": 13455.0,
    "ALE/RoadRunner-v5": 7845.0,
    "ALE/Seaquest-v5": 42054.7,
    "ALE/UpNDown-v5": 11693.2,
}

RANDOM_SCORES = {
    "ALE/Alien-v5": 227.8,
    "ALE/Amidar-v5": 5.8,
    "ALE/Assault-v5": 222.4,
    "ALE/Asterix-v5": 210.0,
    "ALE/Atlantis-v5": 12850.0,
    "ALE/BankHeist-v5": 14.2,
    "ALE/BattleZone-v5": 2360.0,
    "ALE/Boxing-v5": 0.1,
    "ALE/Breakout-v5": 1.7,
    "ALE/Centipede-v5": 2090.9,
    "ALE/ChopperCommand-v5": 811.0,
    "ALE/CrazyClimber-v5": 10780.5,
    "ALE/DemonAttack-v5": 152.1,
    "ALE/Freeway-v5": 0.0,
    "ALE/Frostbite-v5": 65.2,
    "ALE/Gopher-v5": 257.6,
    "ALE/Hero-v5": 1027.0,
    "ALE/Jamesbond-v5": 29.0,
    "ALE/Kangaroo-v5": 52.0,
    "ALE/Krull-v5": 1598.0,
    "ALE/KungFuMaster-v5": 258.5,
    "ALE/MsPacman-v5": 307.3,
    "ALE/Pong-v5": -20.7,
    "ALE/PrivateEye-v5": 24.9,
    "ALE/Qbert-v5": 163.9,
    "ALE/RoadRunner-v5": 11.5,
    "ALE/Seaquest-v5": 68.4,
    "ALE/UpNDown-v5": 533.4,
}


def normalize_score(game_id, score):
    """Normalize score to human-normalized scale."""
    human = HUMAN_SCORES.get(game_id, 1.0)
    random = RANDOM_SCORES.get(game_id, 0.0)
    if human == random:
        return 0.0
    return (score - random) / (human - random)

