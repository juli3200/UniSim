# this script generattes a graph of genome statistics

import extract
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd


OVERLAP_THRESHOLD = 0.9

def bytes_similarity(a: bytes, b: bytes) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 1.0 if len(a) == len(b) else 0.0
    

    same = sum(x == y for x, y in zip(a[:length], b[:length]))
    return same / length


def genome_statistics(world: extract.World) -> list[dict]:
    print("Extracting states...")
    states = []
    while True:
        state = world.get_state()
        if state is None:
            break
        states.append(state)
    print(f"Extracted {len(states)} states.")

    species = []
    species_map = {}  # Map to quickly find species by genome bytes
    data: list[dict] = []

    for i, state in enumerate(states):
        print(f"Processing state {i+1}/{len(states)} with {len(state.entities)} entities...")
        count_map = {}

        for entity in state.entities:
            genome_bytes = entity.genome.raw_bytes

            # Check if genome is already in species_map
            found = False
            for s_id, s in species_map.items():
                if bytes_similarity(genome_bytes, s['genome_bytes']) >= OVERLAP_THRESHOLD:
                    count_map[s_id] = count_map.get(s_id, 0) + 1
                    found = True
                    break

            if not found:
                new_id = len(species)
                species.append({'genome_bytes': genome_bytes, 'id': new_id})
                species_map[new_id] = {'genome_bytes': genome_bytes}
                count_map[new_id] = 1

        # Add counts for this state to the data
        for s_id, count in count_map.items():
            data.append({
                'state': i,
                'id': s_id,
                'count': count,
                'genome_bytes': species[s_id]['genome_bytes']
            })

    return data

def plot_genome_statistics(data: list[dict]):
    # this fn was created by GPT-4

    if not data:
        print("No data to plot.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Pivot to get species counts per state
    pivot = df.pivot(index='state', columns='id', values='count').fillna(0)

    # Optional: normalize to percentages (for 100% stacked area chart)
    pivot_percent = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Plot
    plt.figure(figsize=(12,6))

    plt.stackplot(
        pivot_percent.index,
        pivot_percent.T,       # each species as an area
        labels=[f"Species {i}" for i in pivot_percent.columns]
    )

    plt.xlabel("Time (States)")
    plt.ylabel("Percentage of population")
    plt.title("Genome Species Over Time")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(title="Select World File", filetypes=[("Binary Files", "*.bin"), ("All Files", "*.*")])
    if not file_path:
        print("No file selected. Exiting.")
        quit()

    world = extract.World(file_path)
    world.genome_save = True  # ensure genome data is saved

    if not world.genome_save:
        print("World file does not contain genome data. Exiting.")
        quit()

    data = genome_statistics(world)

    # plotting