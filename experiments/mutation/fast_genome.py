

# this script generattes a graph of genome statistics

import extract
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
import hashlib

import numpy as np
import os



PREFIX_BYTES = 32  # length of prefix used for fingerprinting
def plot_receptor_specs_over_time(specs_data_list, sim_labels, filename):
    """
    Plot the count of each receptor spec over time for multiple simulations.
    specs_data_list: list of lists of dicts (output from receptor_spec_statistics for each sim)
    sim_labels: list of labels for each simulation
    filename: output file path
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.figure(figsize=(14, 7))
    colors = plt.cm.tab20.colors
    for sim_idx, (specs_data, label) in enumerate(zip(specs_data_list, sim_labels)):
        df = pd.DataFrame(specs_data)
        if df.empty:
            continue
        pivot = df.pivot(index='state', columns='receptor_spec', values='count').fillna(0)
        for i, spec in enumerate(pivot.columns):
            plt.plot(
                pivot.index,
                pivot[spec],
                label=f"Spezifikationszahl {spec}",
                color=colors[i % len(colors)],
                linestyle=['-', '--', '-.', ':'][sim_idx % 4]
            )
    plt.xlabel("Zeit (State)")
    plt.ylabel("Anazahl Entitäten mit Rezeptorspezifikation")
    plt.axvline(x=2500, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 5000)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def receptor_spec_statistics(world: extract.World) -> list[dict]:
    """Count how many times each receptor spec appears per state."""
    print("Extracting receptor specs per state...")
    world.end = False
    world.counter = 0
    states = []
    while True:
        state = world.get_state()
        if state is None:
            break
        states.append(state)

    data = []
    
    for i, state in enumerate(states):
        spec_counter = {}
        for entity in state.entities:
            specs = entity.genome.get_receptors_spec()
            for spec in specs:
                spec_counter[spec] = spec_counter.get(spec, 0) + 1
        for spec, count in spec_counter.items():
            data.append({
                'state': i,
                'receptor_spec': spec,
                'count': count
            })

    return data


def bytes_similarity(a: bytes, b: bytes) -> float:
    """Return the fraction of equal bits between two byte sequences."""
    la = len(a)
    lb = len(b)
    if la == 0 or lb == 0:
        return 1.0 if la == lb else 0.0
    length = min(la, lb)
    arr_a = np.frombuffer(a, dtype=np.uint8, count=length)
    arr_b = np.frombuffer(b, dtype=np.uint8, count=length)
    # XOR to find differing bits, then count zeros (equal bits)
    xor = np.bitwise_xor(arr_a, arr_b)
    # Count equal bits: 8 bits per byte minus number of set bits in xor
    # Use vectorized popcount
    bit_counts = np.unpackbits(xor).reshape(-1, 8).sum(axis=1)
    total_bits = length * 8
    equal_bits = total_bits - bit_counts.sum()
    return equal_bits / total_bits




def _fingerprint(genome_bytes: bytes) -> tuple:
    """Fingerprint used for fast candidate lookup: (length, prefix-hash)."""
    l = len(genome_bytes)
    prefix = genome_bytes[:PREFIX_BYTES]
    # short stable hash: use hashlib.blake2b with small digest
    h = hashlib.blake2b(prefix, digest_size=6).digest()
    return (l, h)


def genome_statistics(world: extract.World , ignore_threshold, similarity_threshold) -> list[dict]:
    """Faster species detection per state using fingerprint index to reduce comparisons."""
    print("Extracting states...")
    states = []
    while True:
        state = world.get_state()
        if state is None:
            break
        states.append(state)

    species = []                # list of dicts with genome_bytes and id
    fingerprint_index = {}      # fingerprint -> list of species ids
    data = []

    for i, state in enumerate(states):
        count_map = {}

        # cache local refs
        local_species = species
        local_index = fingerprint_index
        threshold  = similarity_threshold

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
            if count < ignore_threshold:
                continue  # skip rare species
            data.append({
                'state': i,
                'id': s_id,
                'count': count,
                'genome_bytes': species[s_id]['genome_bytes']
            })

    return data

def stackplot_genome_statistics(data: list[dict], filename: str):
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
    # Assign colors, ensuring species with under 3 occurrences per state have the same color
    colors = plt.cm.tab20.colors  # Use a colormap with enough distinct colors
    color_map = {}
    rare_species_color = 'gray'  # Color for rare species

    for i in pivot_percent.columns:
        if (pivot[i] < 3).all():  # Check if species has under 3 occurrences in all states
            color_map[i] = rare_species_color
        else:
            color_map[i] = colors[i % len(colors)]

    # Plot
    plt.figure(figsize=(12, 6))

    plt.stackplot(
        pivot_percent.index,
        pivot_percent.T,  # each species as an area
        labels=[f"Art {i}" for i in pivot_percent.columns],
        colors=[color_map[i] for i in pivot_percent.columns]
    )

    plt.xlabel("Zeit (States)")
    plt.ylabel("Prozentsatz der Population")
    plt.title("Arten im Zeitverlauf")
    plt.legend(
        [f"Art {i}" for i in pivot_percent.columns if color_map[i] != rare_species_color],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0)
    )
    plt.axvline(x=2500, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 5000)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def lineplot_genome_statistics(data: list[dict], filename: str):
    # this fn was created by GPT-4
    if not data:
        print("No data to plot.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Pivot to get species counts per state
    pivot = df.pivot(index='state', columns='id', values='count').fillna(0)

    # Plot
    plt.figure(figsize=(12, 6))

    for species_id in pivot.columns:
        smoothed = pivot[species_id].rolling(window=5, min_periods=1).mean()  # Apply rolling mean for smoothing
        plt.plot(pivot.index, smoothed, label=f"Art {species_id}")

    plt.xlabel("Zeit (States)")
    plt.ylabel("Anzahl der Entitäten")
    plt.title("")
    # Only display legend for species that had over 10 members at least once
    major_species = [species_id for species_id in pivot.columns if (pivot[species_id] > 10).any()]
    # Assign colors to each species for consistency
    colors = plt.cm.tab20.colors
    color_map = {species_id: colors[i % len(colors)] for i, species_id in enumerate(pivot.columns)}

    for species_id in pivot.columns:
        smoothed = pivot[species_id].rolling(window=5, min_periods=1).mean()
        plt.plot(
            pivot.index,
            smoothed,
            label=f"Art {species_id}",
            color=color_map[species_id]
        )
        # Only display legend for species that had over 10 members at least once
        major_species = pivot.columns[(pivot > 10).any(axis=0)].tolist()
        handles = [
            plt.Line2D([0], [0], color=color_map[species_id], lw=2, label=f"Art {species_id}")
            for species_id in major_species
        ]
        plt.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0)
        )
    plt.axvline(x=2500, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 5000)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_num_species_alive(data: list[dict], filename: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the number of species alive (with at least one entity) and total entities at each state.
    Optionally plot and save the figure if filename is provided.
    """
    if not data:
        print("No data to plot.")
        return np.array([]), np.array([])

    df = pd.DataFrame(data)
    # Pivot to get species counts per state
    pivot = df.pivot(index='state', columns='id', values='count').fillna(0)
    # Count number of species with count > 0 at each state
    num_species_alive = (pivot > 0).sum(axis=1).values
    # Total number of entities at each state
    num_entities = pivot.sum(axis=1).values

    if filename:
        plt.figure(figsize=(12, 6))
        plt.plot(pivot.index, num_species_alive, label="Anzahl Arten")
        plt.plot(pivot.index, num_entities, label="Anzahl Entitäten")
        plt.xlabel("Zeit (States)")
        plt.ylabel("Anzahl")
        plt.title("Anzahl Arten und Entitäten im Zeitverlauf")
        plt.axvline(x=2500, color='black', linestyle='--', linewidth=2)
        plt.xlim(0, 5000)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    return num_species_alive, num_entities


def plot_different_species_alive(species_alive: list[np.ndarray], n_entities: list[np.ndarray], mut_rates: list[float], filename: str):
    """Plot number of species alive and total entities for different mutation rates."""
    plt.figure(figsize=(12, 6))

    for num_species, num_entities, mut_rate in zip(species_alive, n_entities, mut_rates):
        plt.plot(num_species, label=f"Arten: {mut_rate}")
        plt.plot(num_entities, linestyle='--', label=f"Entitäten: {mut_rate}")

    plt.xlabel("Zeit (States)")
    plt.ylabel("Anzahl")
    plt.title("Anzahl der lebenden Arten und Entitäten im Zeitverlauf für verschiedene Mutationsraten")
    plt.axvline(x=2500, color='black', linestyle='--', linewidth=2)
    plt.xlim(0, 5000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":

    species_alive = []
    n_entities = []

    overlap_thresholds = [1, 1, 1, 1, 1,1, 1, 1, 1]
    similarity_thresholds = [0.9, 0.9, 0.9, 0.9, 0.9,0.9, 0.9, 0.9, 0.9]
    mut_rates = [ 0]
    specs = []

    for idx, mut_rate in enumerate(mut_rates):
        file_path = f"mutation_rate_{mut_rate}.bin"
        world = extract.World(file_path)


        data = genome_statistics(world, overlap_thresholds[idx], similarity_thresholds[idx])
        print(overlap_thresholds[idx], len(data))
        os.makedirs(f"plots/{mut_rate}", exist_ok=True)
        # plotting
        #lineplot_genome_statistics(data, f"plots/{mut_rate}/lineplot.png")
        #stackplot_genome_statistics(data, f"plots/{mut_rate}/stackplot.png")
        num = plot_num_species_alive(data)
        species_alive.append(num[0])
        n_entities.append(num[1])
        specs.append(receptor_spec_statistics(world))


    plot_receptor_specs_over_time(specs, [str(m) for m in mut_rates], "plots/receptor_specs0.png")    
    plot_different_species_alive(species_alive, n_entities, mut_rates, "plots/different_mutation_rates_species_alive.png")