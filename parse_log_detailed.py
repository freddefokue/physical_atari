import re
from collections import defaultdict
import json

log_file = "/home/fred/intern_agent/physical_atari/training_logs_multigame_13_5M.txt"

# Data structure: game -> visit_idx -> list of dicts {frame, episode, score, avg}
visit_data = defaultdict(lambda: defaultdict(list))
visit_frames = defaultdict(lambda: defaultdict(list))
summary_dict = None

with open(log_file, "r") as f:
    for line in f:
        # Example: 0:delay_multigame frame:13484020   58/s eps 4153, 1635=  725 u 3371005 err 0.6 3.2 loss 1.9 targ 124.9 avg 715.3 game qbert cycle 2 visit 8 ended_by terminated boundary terminated
        m = re.search(r"frame:\s*(\d+).*?eps\s*(\d+).*?=\s*(-?\d+).*?avg\s*(-?[\d.]+).*?game (\w+) cycle (\d+) visit (\d+)", line)
        if m:
            frame = int(m.group(1))
            episode = int(m.group(2))
            score = float(m.group(3))
            avg_score = float(m.group(4))
            game = m.group(5)
            cycle = int(m.group(6))
            visit = int(m.group(7))
            
            visit_data[game][visit].append({
                "frame": frame,
                "episode": episode,
                "score": score,
                "avg_score": avg_score
            })
            visit_frames[game][visit].append(frame)
        elif line.startswith("Summary: "):
            try:
                summary_str = line.replace("Summary: ", "").strip()
                # Use a safer eval since the log outputs a python dict string with single quotes
                summary_dict = eval(summary_str) 
            except Exception as e:
                pass

print("=== Continual Learning Multigame Benchmark Report ===")
print("Agent: Delay Target (KeenAGI)")
if summary_dict:
    print(f"Total Frames: {summary_dict.get('frames')}")
    print(f"Total Episodes: {summary_dict.get('episodes_completed')}")
    print(f"Total Visits: {summary_dict.get('visits_completed')} (3 games x 3 cycles)")

for game in ["ms_pacman", "qbert", "centipede"]:
    visits = sorted(visit_data[game].keys())
    print(f"\n--- {game.upper()} ---")
    for idx, v in enumerate(visits):
        data = visit_data[game][v]
        if not data: continue
        
        start_frame = data[0]['frame']
        end_frame = data[-1]['frame']
        duration = end_frame - start_frame
        
        # Calculate early (first 5%) and late (last 10%) performance
        n_eps = len(data)
        early_n = max(5, int(n_eps * 0.05))
        late_n = max(10, int(n_eps * 0.1))
        
        early_scores = [d['score'] for d in data[:early_n]]
        late_scores = [d['score'] for d in data[-late_n:]]
        
        early_avg = sum(early_scores)/len(early_scores)
        late_avg = sum(late_scores)/len(late_scores)
        peak_score = max([d['score'] for d in data])
        
        print(f"Cycle {idx} (Visit {v}):")
        print(f"  Duration: ~{duration:,} frames ({n_eps} episodes)")
        print(f"  Performance: Started at {early_avg:.1f} pts -> Finished at {late_avg:.1f} pts (Peak: {peak_score:.0f})")

print("\n=== INTERFERENCE DYNAMICS (Forgetting & Plasticity) ===")
for game in ["ms_pacman", "qbert", "centipede"]:
    visits = sorted(visit_data[game].keys())
    if len(visits) >= 2:
        print(f"\n{game.upper()}:")
        for i in range(len(visits)-1):
            v1 = visits[i]
            v2 = visits[i+1]
            
            data_v1 = visit_data[game][v1]
            data_v2 = visit_data[game][v2]
            
            late_n_v1 = max(10, int(len(data_v1) * 0.1))
            early_n_v2 = max(5, int(len(data_v2) * 0.05))
            late_n_v2 = max(10, int(len(data_v2) * 0.1))
            
            late_v1_avg = sum([d['score'] for d in data_v1[-late_n_v1:]]) / late_n_v1
            early_v2_avg = sum([d['score'] for d in data_v2[:early_n_v2]]) / early_n_v2
            late_v2_avg = sum([d['score'] for d in data_v2[-late_n_v2:]]) / late_n_v2
            
            forgetting = late_v1_avg - early_v2_avg
            forgetting_pct = (forgetting / max(1.0, late_v1_avg)) * 100
            
            recovery = late_v2_avg - early_v2_avg
            
            print(f"  Gap between Cycle {i} and Cycle {i+1} (3M frames playing other games):")
            print(f"    Catastrophic Forgetting: Lost {forgetting:.1f} pts (-{forgetting_pct:.1f}%)")
            print(f"      (Ended previous cycle at {late_v1_avg:.1f}, returned at {early_v2_avg:.1f})")
            print(f"    Plasticity / Recovery: Gained +{recovery:.1f} pts over the visit")
            print(f"      (Recovered to {late_v2_avg:.1f} by the end of the visit)")

