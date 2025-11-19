
# this script generattes a graph of genome statistics

import extract
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
try:
    import numpy as np
except Exception:
    np = None

OVERLAP_THRESHOLD = 0.75
PREFIX_BYTES = 32  # length of prefix used for fingerprinting


def bytes_similarity(a: bytes, b: bytes) -> float:
    """Fast similarity: vectorized with numpy if available, fallback to memoryview loop."""
    la = len(a)
    lb = len(b)
    if la == 0 or lb == 0:
        return 1.0 if la == lb else 0.0
    length = min(la, lb)

    # numpy path: very fast for large arrays
    if np is not None:
        arr_a = np.frombuffer(a, dtype=np.uint8, count=length)
        arr_b = np.frombuffer(b, dtype=np.uint8, count=length)
        same = int((arr_a == arr_b).sum())
        return same / length

    # fallback: memoryview (faster than indexing bytes)
    mv_a = memoryview(a)
    mv_b = memoryview(b)
    same = 0
    # iterate in small chunks for speed
    chunk = 128
    i = 0
    while i < length:
        end = i + chunk if i + chunk <= length else length
        sa = mv_a[i:end]
        sb = mv_b[i:end]
        # compare as bytes objects for the chunk
        # counting equal bytes in a chunk
        same += sum(x == y for x, y in zip(sa, sb))
        i = end
    return same / length


def _fingerprint(genome_bytes: bytes):
    """Fingerprint used for fast candidate lookup: (length, prefix-hash)."""
    l = len(genome_bytes)
    prefix = genome_bytes[:PREFIX_BYTES]
    # short stable hash: use hashlib.blake2b with small digest
    h = hashlib.blake2b(prefix, digest_size=6).digest()
    return (l, h)


def genome_statistics(world: extract.World) -> list[dict]:
    """Faster species detection per state using fingerprint index to reduce comparisons."""
    print("Extracting states...")
    states = []
    while True:
        state = world.get_state()
        if state is None:
            break
        states.append(state)
    print(f"Extracted {len(states)} states.")

    species = []                # list of dicts with genome_bytes and id
    fingerprint_index = {}      # fingerprint -> list of species ids
    data = []

    for i, state in enumerate(states):
        print(f"Processing state {i+1}/{len(states)} with {len(state.entities)} entities...")
        count_map = {}

        # cache local refs
        local_species = species
        local_index = fingerprint_index
        threshold = OVERLAP_THRESHOLD

        for entity in state.entities:
            genome_bytes = entity.genome.raw_bytes
            fp = _fingerprint(genome_bytes)
            candidates = local_index.get(fp, ())
            matched = False

            # try exact quick matches first (hash equality)
            for s_id in candidates:
                s_bytes = local_species[s_id]['genome_bytes']
                if genome_bytes == s_bytes:
                    count_map[s_id] = count_map.get(s_id, 0) + 1
                    matched = True
                    break

            if matched:
                continue

            # compare to candidates with similarity test (small set)
            for s_id in candidates:
                s_bytes = local_species[s_id]['genome_bytes']
                if bytes_similarity(genome_bytes, s_bytes) >= threshold:
                    count_map[s_id] = count_map.get(s_id, 0) + 1
                    matched = True
                    break

            if not matched:
                # As fallback, try comparing to a small random subset of species (to catch collisions across fingerprints).
                # Limit checks to keep runtime predictable.
                fallback_checked = 0
                for s_id, s in enumerate(local_species):
                    if fallback_checked >= 10:
                        break
                    # skip same-fingerprint candidates already compared
                    if s_id in candidates:
                        continue
                    if bytes_similarity(genome_bytes, s['genome_bytes']) >= threshold:
                        count_map[s_id] = count_map.get(s_id, 0) + 1
                        matched = True
                        break
                    fallback_checked += 1

            if not matched:
                # new species
                new_id = len(local_species)
                local_species.append({'genome_bytes': genome_bytes, 'id': new_id})
                # index by fingerprint
                local_index.setdefault(fp, []).append(new_id)
                count_map[new_id] = count_map.get(new_id, 0) + 1

        # Add counts for this state to the data
        for s_id, count in count_map.items():
            if count < 2:
                continue  # skip rare species
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
    plot_genome_statistics(data)