import matplotlib.pyplot as plt
import numpy as np

from hopfield_network import (
    EightRookProblem,
    HopfieldNetwork,
    TSPHopfield,
    add_noise,
    generate_random_patterns,
)


def problem1_associative_memory():
    print("PROBLEM 1")

    pattern_size = 100
    num_patterns = 5

    patterns = []

    p1 = np.ones(100)
    p1[::2] = -1
    patterns.append(p1)

    p2 = np.ones(100)
    for i in range(0, 100, 20):
        p2[i : i + 10] = -1
    patterns.append(p2)

    p3 = np.ones(100)
    for i in range(10):
        for j in range(10):
            if (i + j) % 2 == 0:
                p3[i * 10 + j] = -1
    patterns.append(p3)

    p4 = np.ones(100)
    for i in range(10):
        p4[i * 10 + 5] = -1
        p4[5 * 10 + i] = -1
    patterns.append(p4)

    p5 = np.ones(100)
    for i in range(10):
        p5[i * 10 + i] = -1
        p5[i * 10 + (9 - i)] = -1
    patterns.append(p5)

    patterns = np.array(patterns)

    hopfield = HopfieldNetwork(pattern_size)
    hopfield.train(patterns)

    print(f"\nNetwork: {pattern_size} neurons, {num_patterns} patterns stored")
    print(f"Weight matrix: {hopfield.weights.shape}")

    print("\nRecall test (clean patterns):")
    for i, pattern in enumerate(patterns):
        recalled, energy_history = hopfield.recall(pattern)
        match = np.array_equal(recalled, pattern)
        print(
            f"  Pattern {i + 1}: {'OK' if match else 'FAIL'} (energy: {energy_history[-1]:.2f})"
        )

    print("\nRecall test (20% noise):")
    for i, pattern in enumerate(patterns):
        noisy = add_noise(pattern, 0.2)
        recalled, energy_history = hopfield.recall(noisy)
        match = np.array_equal(recalled, pattern)
        accuracy = np.mean(recalled == pattern) * 100
        print(f"  Pattern {i + 1}: {accuracy:.1f}% match")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("10x10 Associative Memory", fontsize=16)

    for i, pattern in enumerate(patterns):
        axes[0, i].imshow(pattern.reshape(10, 10), cmap="RdYlGn", vmin=-1, vmax=1)
        axes[0, i].set_title(f"Pattern {i + 1}")
        axes[0, i].axis("off")

        noisy = add_noise(pattern, 0.2)
        recalled, _ = hopfield.recall(noisy)
        axes[1, i].imshow(recalled.reshape(10, 10), cmap="RdYlGn", vmin=-1, vmax=1)
        axes[1, i].set_title("Recalled")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("problem1_associative_memory.png", dpi=150, bbox_inches="tight")
    print("\nSaved: problem1_associative_memory.png")
    plt.close()


def problem2_network_capacity():
    print("PROBLEM 2")

    pattern_size = 100
    max_patterns = 30

    print(f"\nTesting capacity for {pattern_size} neurons...")

    hopfield = HopfieldNetwork(pattern_size)
    accuracies = []
    pattern_counts = []

    for num_patterns in range(1, max_patterns + 1):
        patterns = generate_random_patterns(num_patterns, pattern_size)
        hopfield.train(patterns)

        correct = 0
        for pattern in patterns:
            recalled, _ = hopfield.recall(pattern, max_iter=1000)
            if np.array_equal(recalled, pattern):
                correct += 1

        accuracy = correct / num_patterns
        accuracies.append(accuracy)
        pattern_counts.append(num_patterns)

        print(
            f"  {num_patterns:2d} patterns: {accuracy * 100:5.1f}% ({correct}/{num_patterns})"
        )

        if accuracy < 0.8 and num_patterns > 5:
            print(f"\nCapacity reached at ~{num_patterns - 1} patterns")
            break

    theoretical = 0.138 * pattern_size
    print(f"\nTheoretical capacity: ~{theoretical:.1f} patterns")
    print(f"Observed capacity: ~{pattern_counts[-1] - 1} patterns")
    print(f"Ratio: {(pattern_counts[-1] - 1) / pattern_size:.3f}")

    plt.figure(figsize=(10, 6))
    plt.plot(pattern_counts, [a * 100 for a in accuracies], "b-o", linewidth=2)
    plt.axhline(y=100, color="g", linestyle="--", alpha=0.5, label="Perfect recall")
    plt.axhline(y=80, color="r", linestyle="--", alpha=0.5, label="80% threshold")
    plt.axvline(
        x=theoretical,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label=f"Theoretical (~{theoretical:.0f})",
    )
    plt.xlabel("Number of Stored Patterns", fontsize=12)
    plt.ylabel("Recall Accuracy (%)", fontsize=12)
    plt.title("Network Capacity (100 neurons)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("problem2_capacity.png", dpi=150, bbox_inches="tight")
    print("\nSaved: problem2_capacity.png")
    plt.close()


def problem3_error_correction():
    print("PROBLEM 3")

    pattern_size = 100
    num_patterns = 5

    patterns = generate_random_patterns(num_patterns, pattern_size)
    hopfield = HopfieldNetwork(pattern_size)
    hopfield.train(patterns)

    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    print(f"\nTesting with {num_patterns} patterns...")
    print("Noise | Accuracy | Success")
    print("-" * 40)

    results = []
    for noise_level in noise_levels:
        accuracies = []
        successes = 0

        for pattern in patterns:
            noisy = add_noise(pattern, noise_level)
            recalled, _ = hopfield.recall(noisy, max_iter=2000)
            accuracy = np.mean(recalled == pattern)
            accuracies.append(accuracy)

            if np.array_equal(recalled, pattern):
                successes += 1

        avg_accuracy = np.mean(accuracies)
        success_rate = successes / num_patterns
        results.append((noise_level, avg_accuracy, success_rate))

        print(
            f"{noise_level * 100:4.0f}% | {avg_accuracy * 100:6.1f}% | {success_rate * 100:5.1f}% ({successes}/{num_patterns})"
        )

    max_noise = 0
    for noise_level, avg_acc, success_rate in results:
        if success_rate >= 0.8:
            max_noise = noise_level

    print(f"\nMax reliable noise correction: {max_noise * 100:.0f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    noise_pct = [r[0] * 100 for r in results]
    accuracies = [r[1] * 100 for r in results]
    success_rates = [r[2] * 100 for r in results]

    ax1.plot(noise_pct, accuracies, "b-o", linewidth=2, label="Avg Accuracy")
    ax1.axhline(y=100, color="g", linestyle="--", alpha=0.5)
    ax1.axhline(y=80, color="r", linestyle="--", alpha=0.5, label="80% threshold")
    ax1.set_xlabel("Noise Level (%)", fontsize=12)
    ax1.set_ylabel("Recall Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs Noise", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(noise_pct, success_rates, "r-s", linewidth=2, label="Success Rate")
    ax2.axhline(y=100, color="g", linestyle="--", alpha=0.5)
    ax2.axhline(y=80, color="orange", linestyle="--", alpha=0.5, label="80% threshold")
    ax2.set_xlabel("Noise Level (%)", fontsize=12)
    ax2.set_ylabel("Perfect Recall Rate (%)", fontsize=12)
    ax2.set_title("Success Rate vs Noise", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("problem3_error_correction.png", dpi=150, bbox_inches="tight")
    print("\nSaved: problem3_error_correction.png")
    plt.close()


def problem4_eight_rook():
    print("PROBLEM 4")

    print("\nEnergy function:")
    print("  E = A*sum(row_sum-1)^2 + B*sum(col_sum-1)^2 + C*(total-8)^2")
    print("\nWeight selection:")
    print("  A = 2.0  (row constraint)")
    print("  B = 2.0  (column constraint)")
    print("  C = 1.0  (total count)")
    print("\nReason: A and B equal for symmetry, C lower as it's redundant")

    rook_problem = EightRookProblem()

    print(f"\nNetwork: {rook_problem.num_neurons} neurons (8x8)")
    print(f"Weights: {rook_problem.weights.shape}")
    print(f"Total connections: {rook_problem.num_neurons * rook_problem.num_neurons}")

    print("\nSolving...")
    solution, energy, is_valid = rook_problem.solve(max_iter=5000, num_attempts=20)

    print(f"Result: {'Valid solution' if is_valid else 'Suboptimal solution'}")
    print(f"Energy: {energy:.6f}")

    binary_state = (solution + 1) / 2
    board = binary_state.reshape(8, 8)

    print("\nBoard configuration:")
    for i, row in enumerate(board):
        print(f"  Row {i}: ", end="")
        for val in row:
            print(f"{int(val)} ", end="")
        print()

    row_sums = np.sum(board, axis=1)
    col_sums = np.sum(board, axis=0)
    total_rooks = np.sum(board)

    print(f"\nRow sums: {row_sums}")
    print(f"Col sums: {col_sums}")
    print(f"Total rooks: {int(total_rooks)}")

    row_valid = np.allclose(row_sums, 1)
    col_valid = np.allclose(col_sums, 1)
    total_valid = np.isclose(total_rooks, 8)

    print(
        f"\nConstraints: Row {'OK' if row_valid else 'FAIL'}, Column {'OK' if col_valid else 'FAIL'}, Total {'OK' if total_valid else 'FAIL'}"
    )

    fig = rook_problem.visualize_solution(solution)
    plt.savefig("problem4_eight_rook.png", dpi=150, bbox_inches="tight")
    print("\nSaved: problem4_eight_rook.png")
    plt.close()


def problem5_tsp():
    print("PROBLEM 5")

    num_cities = 10

    np.random.seed(42)
    city_positions = np.random.rand(num_cities, 2) * 100

    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(
                    city_positions[i] - city_positions[j]
                )

    print(f"\nCities: {num_cities}")
    print(f"Distance matrix: {distance_matrix.shape}")

    tsp = TSPHopfield(distance_matrix)

    print(f"\nNetwork architecture:")
    print(f"  Neurons: {tsp.num_neurons} (NxN = {num_cities}x{num_cities})")
    print(f"  Neuron x_i,j = city i at position j")
    print(f"\nNumber of weights: {tsp.num_neurons * tsp.num_neurons:,}")
    print(f"  (N^2xN^2 = {num_cities}^2x{num_cities}^2 = {num_cities**4:,})")

    print(f"\nWeight parameters:")
    print(f"  A = {tsp.A} (one city per position)")
    print(f"  B = {tsp.B} (each city once)")
    print(f"  C = {tsp.C} (total cities)")
    print(f"  D = {tsp.D} (distance)")

    print("\nSolving...")
    solution, energy, is_valid = tsp.solve(max_iter=10000, num_attempts=30)

    print(f"Result: {'Valid tour' if is_valid else 'Invalid tour'}")
    print(f"Energy: {energy:.2f}")

    binary_state = (solution + 1) / 2
    tour_matrix = binary_state.reshape(num_cities, num_cities)

    print("\nTour matrix:")
    print("     ", end="")
    for j in range(num_cities):
        print(f"{j:4d}", end="")
    print()

    for i in range(num_cities):
        print(f"C{i:2d}: ", end="")
        for j in range(num_cities):
            print(f"{int(tour_matrix[i][j]):4d}", end="")
        print()

    row_sums = np.sum(tour_matrix, axis=1)
    col_sums = np.sum(tour_matrix, axis=0)

    print(f"\nRow sums: {row_sums}")
    print(f"Col sums: {col_sums}")

    row_valid = np.allclose(row_sums, 1, atol=0.1)
    col_valid = np.allclose(col_sums, 1, atol=0.1)

    print(f"Row constraint: {'OK' if row_valid else 'FAIL'}")
    print(f"Column constraint: {'OK' if col_valid else 'FAIL'}")

    if is_valid or (row_valid and col_valid):
        tour_distance = tsp.get_tour_distance(solution)
        print(f"\nTour distance: {tour_distance:.2f}")

        tour_order = []
        for t in range(num_cities):
            cities = np.where(tour_matrix[:, t] > 0.5)[0]
            if len(cities) > 0:
                tour_order.append(cities[0])

        print(f"Tour: {tour_order}")
    else:
        print("\nTour invalid, distance not computed")

    fig = tsp.visualize_tour(solution, city_positions)
    plt.savefig("problem5_tsp.png", dpi=150, bbox_inches="tight")
    print("\nSaved: problem5_tsp.png")
    plt.close()

    greedy_tour = greedy_tsp(distance_matrix)
    greedy_distance = calculate_tour_distance(greedy_tour, distance_matrix)
    print(f"\nGreedy NN tour: {greedy_tour}")
    print(f"Greedy distance: {greedy_distance:.2f}")

    if is_valid or (row_valid and col_valid):
        ratio = tour_distance / greedy_distance
        print(f"Hopfield/Greedy: {ratio:.2f}x")


def greedy_tsp(distance_matrix):
    n = len(distance_matrix)
    unvisited = set(range(1, n))
    tour = [0]
    current = 0

    while unvisited:
        nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return tour


def calculate_tour_distance(tour, distance_matrix):
    distance = 0
    for i in range(len(tour)):
        distance += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
    return distance


def main():
    print("\n" + "=" * 80)
    print("WEEK 6 LAB: HOPFIELD NETWORKS")
    print("=" * 80 + "\n")

    np.random.seed(42)

    problem1_associative_memory()
    problem2_network_capacity()
    problem3_error_correction()
    problem4_eight_rook()
    problem5_tsp()


if __name__ == "__main__":
    main()
