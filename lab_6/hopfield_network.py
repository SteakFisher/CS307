from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class HopfieldNetwork:
    def __init__(self, num_neurons: int):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
        self.patterns = []

    def train(self, patterns: np.ndarray):
        self.patterns = patterns
        num_patterns = patterns.shape[0]
        self.weights = np.zeros((self.num_neurons, self.num_neurons))

        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)

        self.weights /= num_patterns
        np.fill_diagonal(self.weights, 0)

    def energy(self, state: np.ndarray) -> float:
        return -0.5 * np.dot(state, np.dot(self.weights, state))

    def update_async(
        self, state: np.ndarray, max_iter: int = 1000
    ) -> Tuple[np.ndarray, List[float]]:
        state = state.copy()
        energy_history = [self.energy(state)]

        for iteration in range(max_iter):
            i = np.random.randint(0, self.num_neurons)
            activation = np.dot(self.weights[i], state)
            state[i] = 1 if activation >= 0 else -1

            current_energy = self.energy(state)
            energy_history.append(current_energy)

            if len(energy_history) > 10 and np.allclose(
                energy_history[-10:], current_energy
            ):
                break

        return state, energy_history

    def update_sync(
        self, state: np.ndarray, max_iter: int = 100
    ) -> Tuple[np.ndarray, List[float]]:
        state = state.copy()
        energy_history = [self.energy(state)]

        for iteration in range(max_iter):
            activations = np.dot(self.weights, state)
            new_state = np.where(activations >= 0, 1, -1)

            current_energy = self.energy(new_state)
            energy_history.append(current_energy)

            if np.array_equal(state, new_state):
                break

            state = new_state

        return state, energy_history

    def recall(
        self, pattern: np.ndarray, async_update: bool = True, max_iter: int = 1000
    ) -> Tuple[np.ndarray, List[float]]:
        if async_update:
            return self.update_async(pattern, max_iter)
        else:
            return self.update_sync(pattern, max_iter)

    def test_capacity(
        self, pattern_size: int, max_patterns: int = 50
    ) -> Tuple[int, List[float]]:
        accuracies = []

        for num_patterns in range(1, max_patterns + 1):
            patterns = np.random.choice([-1, 1], size=(num_patterns, pattern_size))
            self.train(patterns)

            correct = 0
            for pattern in patterns:
                recalled, _ = self.recall(pattern, max_iter=1000)
                if np.array_equal(recalled, pattern):
                    correct += 1

            accuracy = correct / num_patterns
            accuracies.append(accuracy)

            if accuracy < 0.9:
                return num_patterns - 1, accuracies

        return max_patterns, accuracies

    def test_error_correction(self, noise_levels: List[float] = None) -> dict:
        if noise_levels is None:
            noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        if len(self.patterns) == 0:
            raise ValueError("Network must be trained first")

        results = {noise: [] for noise in noise_levels}

        for pattern in self.patterns:
            for noise_level in noise_levels:
                noisy_pattern = pattern.copy()
                num_flips = int(noise_level * len(pattern))
                flip_indices = np.random.choice(len(pattern), num_flips, replace=False)
                noisy_pattern[flip_indices] *= -1

                recalled, _ = self.recall(noisy_pattern, max_iter=1000)
                accuracy = np.mean(recalled == pattern)
                results[noise_level].append(accuracy)

        for noise_level in noise_levels:
            results[noise_level] = np.mean(results[noise_level])

        return results


class EightRookProblem:
    def __init__(self):
        self.board_size = 8
        self.num_neurons = self.board_size * self.board_size
        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.bias = np.zeros(self.num_neurons)
        self._setup_energy_function()

    def _neuron_index(self, row: int, col: int) -> int:
        return row * self.board_size + col

    def _position_from_index(self, idx: int) -> Tuple[int, int]:
        return idx // self.board_size, idx % self.board_size

    def _setup_energy_function(self):
        A = 2.0
        B = 2.0
        C = 1.0

        self.weights = np.zeros((self.num_neurons, self.num_neurons))

        for i in range(self.num_neurons):
            row_i, col_i = self._position_from_index(i)

            for j in range(self.num_neurons):
                if i == j:
                    continue

                row_j, col_j = self._position_from_index(j)

                if row_i == row_j:
                    self.weights[i][j] -= 2 * A

                if col_i == col_j:
                    self.weights[i][j] -= 2 * B

                self.weights[i][j] -= 2 * C

        for i in range(self.num_neurons):
            self.bias[i] = A + B + 2 * C * self.board_size - C

    def energy(self, state: np.ndarray) -> float:
        binary_state = (state + 1) / 2
        board = binary_state.reshape(self.board_size, self.board_size)

        A, B, C = 2.0, 2.0, 1.0

        row_energy = A * np.sum((np.sum(board, axis=1) - 1) ** 2)
        col_energy = B * np.sum((np.sum(board, axis=0) - 1) ** 2)
        total_energy = C * (np.sum(board) - self.board_size) ** 2

        return row_energy + col_energy + total_energy

    def solve(
        self, max_iter: int = 5000, num_attempts: int = 10
    ) -> Tuple[np.ndarray, float, bool]:
        best_state = None
        best_energy = float("inf")

        for attempt in range(num_attempts):
            state = np.random.choice([-1, 1], size=self.num_neurons)

            for iteration in range(max_iter):
                i = np.random.randint(0, self.num_neurons)
                activation = np.dot(self.weights[i], state) + self.bias[i]
                state[i] = 1 if activation >= 0 else -1

                if iteration % 100 == 0:
                    current_energy = self.energy(state)

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = state.copy()

                    if current_energy < 1e-6:
                        return best_state, best_energy, True

        is_valid = best_energy < 1e-6
        return best_state, best_energy, is_valid

    def visualize_solution(self, state: np.ndarray):
        binary_state = (state + 1) / 2
        board = binary_state.reshape(self.board_size, self.board_size)

        plt.figure(figsize=(8, 8))
        plt.imshow(board, cmap="RdYlGn", interpolation="nearest")
        plt.title("Eight-Rook Problem Solution")
        plt.colorbar(label="Rook Present")

        for i in range(self.board_size + 1):
            plt.axhline(i - 0.5, color="black", linewidth=1)
            plt.axvline(i - 0.5, color="black", linewidth=1)

        plt.xticks(range(self.board_size))
        plt.yticks(range(self.board_size))
        plt.tight_layout()
        return plt.gcf()


class TSPHopfield:
    def __init__(self, distance_matrix: np.ndarray):
        self.num_cities = distance_matrix.shape[0]
        self.distance_matrix = distance_matrix
        self.num_neurons = self.num_cities * self.num_cities

        self.A = 500
        self.B = 500
        self.C = 200
        self.D = 500

        self.weights = np.zeros((self.num_neurons, self.num_neurons))
        self.bias = np.zeros(self.num_neurons)
        self._setup_weights()

    def _neuron_index(self, city: int, time: int) -> int:
        return city * self.num_cities + time

    def _position_from_index(self, idx: int) -> Tuple[int, int]:
        return idx // self.num_cities, idx % self.num_cities

    def _setup_weights(self):
        N = self.num_cities

        for i in range(N):
            for j in range(N):
                idx1 = self._neuron_index(i, j)

                for k in range(N):
                    for l in range(N):
                        idx2 = self._neuron_index(k, l)

                        if idx1 == idx2:
                            continue

                        weight = 0

                        if i != k and j == l:
                            weight -= self.A

                        if i == k and j != l:
                            weight -= self.B

                        weight -= self.C

                        if j != l and ((l == (j + 1) % N) or (l == (j - 1) % N)):
                            weight -= self.D * self.distance_matrix[i][k]

                        self.weights[idx1][idx2] = weight

                self.bias[idx1] = self.A + self.B + self.C * N

    def energy(self, state: np.ndarray) -> float:
        binary_state = (state + 1) / 2
        tour = binary_state.reshape(self.num_cities, self.num_cities)

        N = self.num_cities

        energy_a = self.A / 2 * np.sum((np.sum(tour, axis=0) - 1) ** 2)
        energy_b = self.B / 2 * np.sum((np.sum(tour, axis=1) - 1) ** 2)
        energy_c = self.C / 2 * (np.sum(tour) - N) ** 2

        energy_d = 0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    j_next = (j + 1) % N
                    energy_d += (
                        self.D
                        / 2
                        * self.distance_matrix[i][k]
                        * tour[i][j]
                        * tour[k][j_next]
                    )

        return energy_a + energy_b + energy_c + energy_d

    def solve(
        self, max_iter: int = 10000, num_attempts: int = 20, temperature: float = 0.1
    ) -> Tuple[np.ndarray, float, bool]:
        best_state = None
        best_energy = float("inf")

        for attempt in range(num_attempts):
            state = np.random.choice([-1, 1], size=self.num_neurons, p=[0.4, 0.6])

            for iteration in range(max_iter):
                i = np.random.randint(0, self.num_neurons)
                activation = np.dot(self.weights[i], state) + self.bias[i]

                prob = 1 / (1 + np.exp(-2 * activation / temperature))
                state[i] = 1 if np.random.random() < prob else -1

                if iteration % 500 == 0:
                    current_energy = self.energy(state)

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = state.copy()

        binary_state = (best_state + 1) / 2
        tour = binary_state.reshape(self.num_cities, self.num_cities)

        row_valid = np.allclose(np.sum(tour, axis=1), 1)
        col_valid = np.allclose(np.sum(tour, axis=0), 1)
        is_valid = row_valid and col_valid

        return best_state, best_energy, is_valid

    def get_tour_distance(self, state: np.ndarray) -> float:
        binary_state = (state + 1) / 2
        tour = binary_state.reshape(self.num_cities, self.num_cities)

        tour_order = []
        for t in range(self.num_cities):
            city = np.argmax(tour[:, t])
            tour_order.append(city)

        total_distance = 0
        for i in range(len(tour_order)):
            city1 = tour_order[i]
            city2 = tour_order[(i + 1) % len(tour_order)]
            total_distance += self.distance_matrix[city1][city2]

        return total_distance

    def visualize_tour(self, state: np.ndarray, city_positions: np.ndarray = None):
        binary_state = (state + 1) / 2
        tour = binary_state.reshape(self.num_cities, self.num_cities)

        tour_order = []
        for t in range(self.num_cities):
            cities = np.where(tour[:, t] > 0.5)[0]
            if len(cities) > 0:
                tour_order.append(cities[0])

        if city_positions is None:
            np.random.seed(42)
            city_positions = np.random.rand(self.num_cities, 2) * 100

        plt.figure(figsize=(10, 10))

        plt.scatter(
            city_positions[:, 0], city_positions[:, 1], c="red", s=200, zorder=3
        )

        for i, (x, y) in enumerate(city_positions):
            plt.text(
                x, y, str(i), ha="center", va="center", fontsize=12, fontweight="bold"
            )

        if len(tour_order) > 0:
            tour_order.append(tour_order[0])
            for i in range(len(tour_order) - 1):
                city1 = tour_order[i]
                city2 = tour_order[i + 1]
                plt.plot(
                    [city_positions[city1, 0], city_positions[city2, 0]],
                    [city_positions[city1, 1], city_positions[city2, 1]],
                    "b-",
                    linewidth=2,
                    alpha=0.6,
                )

        plt.title(f"TSP Solution - Tour: {tour_order[:-1]}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


def generate_random_patterns(num_patterns: int, pattern_size: int) -> np.ndarray:
    return np.random.choice([-1, 1], size=(num_patterns, pattern_size))


def add_noise(pattern: np.ndarray, noise_level: float) -> np.ndarray:
    noisy = pattern.copy()
    num_flips = int(noise_level * len(pattern))
    flip_indices = np.random.choice(len(pattern), num_flips, replace=False)
    noisy[flip_indices] *= -1
    return noisy
