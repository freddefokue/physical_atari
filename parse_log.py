import re
from collections import defaultdict

log_file = "/home/fred/intern_agent/physical_atari/training_logs_multigame_13_5M.txt"

# Data structure: game -> visit_idx -> list of scores
visit_data = defaultdict(lambda: defaultdict(list))
visit_frames = defaultdict(lambda: defaultdict(list))

with open(log_file, "r") as f:
    for line in f:
        # Example: ... frame:13484020 ... eps 4153, 1635=  725 u ... game qbert cycle 2 visit 8 ended_by terminated ...
        m = re.search(r"frame:\s*(\d+).*?=\s*(-?\d+).*?game (\w+) cycle (\d+) visit (\d+)", line)
        if m:
            frame = int(m.group(1))
            score = float(m.group(2))
            game = m.group(3)
            cycle = int(m.group(4))
            visit = int(m.group(5))
            
            visit_data[game][visit].append(score)
            visit_frames[game][visit].append(frame)

print("Game | Visit | Num Eps | Early Score (First 10) | Late Score (Last 10)")
print("-" * 75)
for game in ["ms_pacman", "qbert", "centipede"]:
    visits = sorted(visit_data[game].keys())
    for v in visits:
        scores = visit_data[game][v]
        early = sum(scores[:10]) / min(len(scores), 10) if len(scores) > 0 else 0
        late = sum(scores[-10:]) / min(len(scores), 10) if len(scores) > 0 else 0
        print(f"{game:<12} | {v:5d} | {len(scores):7d} | {early:20.1f} | {late:20.1f}")
        
print("\nContinual Learning Analysis (Plasticity / Forgetting based on last 10 eps):")
for game in ["ms_pacman", "qbert", "centipede"]:
    visits = sorted(visit_data[game].keys())
    if len(visits) >= 2:
        print(f"\n{game.upper()}:")
        for i in range(len(visits)-1):
            v1 = visits[i]
            v2 = visits[i+1]
            late_v1 = sum(visit_data[game][v1][-10:]) / min(len(visit_data[game][v1]), 10)
            early_v2 = sum(visit_data[game][v2][:10]) / min(len(visit_data[game][v2]), 10)
            late_v2 = sum(visit_data[game][v2][-10:]) / min(len(visit_data[game][v2]), 10)
            
            forgetting = late_v1 - early_v2
            plasticity = late_v2 - early_v2
            
            print(f"  Cycle {i} -> Cycle {i+1} (Visit {v1} to {v2}):")
            print(f"    Forgetting (Drop after switch): {forgetting:6.1f} (Left at {late_v1:.1f}, Returned at {early_v2:.1f})")
            print(f"    Plasticity (Re-learning within visit): +{plasticity:6.1f} (Started at {early_v2:.1f}, Finished at {late_v2:.1f})")

