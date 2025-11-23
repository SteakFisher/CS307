import matplotlib.pyplot as plt
import numpy as np

from hopfield_network import HopfieldNetwork, add_noise


def create_letter_patterns():
    T = np.array(
        [
            [1, 1, 1, 1, 1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
        ]
    ).flatten()

    L = np.array(
        [
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1],
        ]
    ).flatten()

    X = np.array(
        [
            [1, -1, -1, -1, 1],
            [-1, 1, -1, 1, -1],
            [-1, -1, 1, -1, -1],
            [-1, 1, -1, 1, -1],
            [1, -1, -1, -1, 1],
        ]
    ).flatten()

    O = np.array(
        [
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [-1, 1, 1, 1, -1],
        ]
    ).flatten()

    return np.array([T, L, X, O])


def demo_pattern_recognition():
    print("=" * 60)
    print("Pattern Recognition Demo")
    print("=" * 60)

    patterns = create_letter_patterns()
    pattern_names = ["T", "L", "X", "O"]

    hopfield = HopfieldNetwork(25)
    hopfield.train(patterns)

    print(f"\nStored {len(patterns)} letter patterns (5x5)")
    print(f"Network: {hopfield.num_neurons} neurons")

    print("\nClean recall:")
    for i, (pattern, name) in enumerate(zip(patterns, pattern_names)):
        recalled, energy_history = hopfield.recall(pattern)
        match = np.array_equal(recalled, pattern)
        print(f"  {name}: {'OK' if match else 'FAIL'} (E={energy_history[-1]:.2f})")

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle("Letter Pattern Recognition", fontsize=16)

    for i, (pattern, name) in enumerate(zip(patterns, pattern_names)):
        axes[0, i].imshow(
            pattern.reshape(5, 5),
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        axes[0, i].set_title(f"Original '{name}'")
        axes[0, i].axis("off")

        noisy = add_noise(pattern, 0.3)
        axes[1, i].imshow(
            noisy.reshape(5, 5), cmap="RdYlGn", vmin=-1, vmax=1, interpolation="nearest"
        )
        axes[1, i].set_title(f"Noisy (30%)")
        axes[1, i].axis("off")

        recalled, _ = hopfield.recall(noisy, max_iter=1000)
        axes[2, i].imshow(
            recalled.reshape(5, 5),
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        accuracy = np.mean(recalled == pattern) * 100
        axes[2, i].set_title(f"Recalled ({accuracy:.0f}%)")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig("demo_letters.png", dpi=150, bbox_inches="tight")
    print("\nSaved: demo_letters.png")
    plt.close()


def demo_noise_robustness():
    print("\n" + "=" * 60)
    print("Noise Robustness Test")
    print("=" * 60)

    patterns = create_letter_patterns()
    hopfield = HopfieldNetwork(25)
    hopfield.train(patterns)

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    print("\nTesting 'T' pattern:")
    print("Noise | Flips | Accuracy")
    print("-" * 35)

    pattern = patterns[0]

    for noise in noise_levels:
        noisy = add_noise(pattern, noise)
        recalled, _ = hopfield.recall(noisy, max_iter=2000)

        bits_flipped = int(noise * len(pattern))
        accuracy = np.mean(recalled == pattern) * 100

        print(f"{noise * 100:4.0f}% | {bits_flipped:2d}    | {accuracy:6.1f}%")

    threshold = 0
    for noise in noise_levels:
        noisy = add_noise(pattern, noise)
        recalled, _ = hopfield.recall(noisy, max_iter=2000)
        if np.mean(recalled == pattern) >= 0.9:
            threshold = noise

    print(f"\nMax reliable correction: {threshold * 100:.0f}%")


def demo_capacity():
    print("\n" + "=" * 60)
    print("Network Capacity Test")
    print("=" * 60)

    pattern_size = 50
    print(f"\nNetwork: {pattern_size} neurons")
    print(f"Theoretical: ~{int(0.138 * pattern_size)} patterns")

    for num_patterns in [2, 4, 6, 8, 10]:
        patterns = np.random.choice([-1, 1], size=(num_patterns, pattern_size))
        hopfield = HopfieldNetwork(pattern_size)
        hopfield.train(patterns)

        correct = 0
        for pattern in patterns:
            recalled, _ = hopfield.recall(pattern, max_iter=1000)
            if np.array_equal(recalled, pattern):
                correct += 1

        accuracy = correct / num_patterns * 100
        status = "OK" if accuracy == 100 else "PARTIAL" if accuracy >= 80 else "FAIL"

        print(
            f"  {num_patterns:2d} patterns: {accuracy:6.1f}% ({correct}/{num_patterns}) {status}"
        )


def demo_energy():
    print("\n" + "=" * 60)
    print("Energy Minimization")
    print("=" * 60)

    patterns = create_letter_patterns()[:2]
    hopfield = HopfieldNetwork(25)
    hopfield.train(patterns)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (pattern, name) in enumerate(zip(patterns, ["T", "L"])):
        noisy = add_noise(pattern, 0.4)
        recalled, energy_history = hopfield.recall(noisy, max_iter=500)

        axes[i].plot(energy_history, "b-", linewidth=2)
        axes[i].axhline(
            y=energy_history[-1],
            color="r",
            linestyle="--",
            label=f"Final: {energy_history[-1]:.1f}",
        )
        axes[i].axhline(
            y=energy_history[0],
            color="g",
            linestyle="--",
            label=f"Initial: {energy_history[0]:.1f}",
        )
        axes[i].set_xlabel("Iteration", fontsize=12)
        axes[i].set_ylabel("Energy", fontsize=12)
        axes[i].set_title(f"Pattern '{name}'", fontsize=14)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

        accuracy = np.mean(recalled == pattern) * 100
        print(f"  {name}: {len(energy_history)} iterations, {accuracy:.1f}% recovered")

    plt.tight_layout()
    plt.savefig("demo_energy.png", dpi=150, bbox_inches="tight")
    print("\nSaved: demo_energy.png")
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("HOPFIELD NETWORK DEMONSTRATIONS")
    print("=" * 60 + "\n")

    np.random.seed(42)

    demo_pattern_recognition()
    demo_noise_robustness()
    demo_capacity()
    demo_energy()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("\nKey findings:")
    print("  - Hopfield nets work well for associative memory")
    print("  - Can correct 10-30% noise depending on patterns")
    print("  - Capacity limited to ~0.138N patterns")
    print("  - Energy always decreases (guaranteed convergence)")
    print()


if __name__ == "__main__":
    main()
