import numpy as np
from scipy.stats import poisson

# ------------------------------
# Problem Parameters
# ------------------------------
MAX_BIKES = 20
MAX_MOVE = 5
RENT_REWARD = 10
MOVE_COST = 2
PARKING_LIMIT = 10
PARKING_PENALTY = 4
DISCOUNT = 0.9

# Free move from location 1 → 2: 1 bike is free
FREE_MOVE_DIR = 1  # 1 = L1→L2 free, -1 = L2→L1 free

# Poisson parameters
RENT_L1 = 3
RENT_L2 = 4
RET_L1 = 3
RET_L2 = 2

# Poisson probability limit (truncate tail)
POISSON_CUTOFF = 11

# ------------------------------
# Precompute Poisson probabilities
# ------------------------------
poisson_cache = dict()

def poisson_pmf(n, lam):
    key = (n, lam)
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

# ------------------------------
# Expected Reward + Next State Calculation
# ------------------------------
def expected_return(state, action, V):
    # State is (b1, b2)
    b1, b2 = state

    # Apply action (move bikes)
    moved = int(action)

    # Free move from L1→L2
    cost = 0
    if moved > 0 and FREE_MOVE_DIR == 1:
        # first bike free
        cost -= MOVE_COST * max(0, moved - 1)
    elif moved < 0 and FREE_MOVE_DIR == 1:
        # opposite direction: no free move
        cost -= MOVE_COST * abs(moved)
    else:
        cost -= MOVE_COST * abs(moved)

    # New bike counts after movement
    b1 -= moved
    b2 += moved

    # Enforce capacity limit
    if b1 > MAX_BIKES: b1 = MAX_BIKES
    if b2 > MAX_BIKES: b2 = MAX_BIKES
    if b1 < 0: b1 = 0
    if b2 < 0: b2 = 0

    # Parking penalty
    if b1 > PARKING_LIMIT:
        cost -= PARKING_PENALTY
    if b2 > PARKING_LIMIT:
        cost -= PARKING_PENALTY

    value = cost  # Start with movement & parking cost

    # Expected returns over all Poisson rentals & returns
    for rent1 in range(POISSON_CUTOFF):
        for rent2 in range(POISSON_CUTOFF):
            p_rent = (
                poisson_pmf(rent1, RENT_L1) *
                poisson_pmf(rent2, RENT_L2)
            )

            # Actual rentals that can be satisfied
            real_rent1 = min(b1, rent1)
            real_rent2 = min(b2, rent2)

            reward = (real_rent1 + real_rent2) * RENT_REWARD

            b1_after_rent = b1 - real_rent1
            b2_after_rent = b2 - real_rent2

            # Returns
            for ret1 in range(POISSON_CUTOFF):
                for ret2 in range(POISSON_CUTOFF):

                    p_ret = (
                        poisson_pmf(ret1, RET_L1) *
                        poisson_pmf(ret2, RET_L2)
                    )

                    prob = p_rent * p_ret

                    b1_final = min(b1_after_rent + ret1, MAX_BIKES)
                    b2_final = min(b2_after_rent + ret2, MAX_BIKES)

                    value += prob * (reward + DISCOUNT * V[b1_final, b2_final])

    return value

# ------------------------------
# POLICY ITERATION
# ------------------------------
def policy_iteration():
    V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)

    stable = False

    while not stable:
        print("Policy Evaluation...")
        # Policy Evaluation
        while True:
            delta = 0
            new_V = np.copy(V)
            for i in range(MAX_BIKES + 1):
                for j in range(MAX_BIKES + 1):
                    action = policy[i, j]
                    new_V[i, j] = expected_return((i, j), action, V)

            delta = np.abs(V - new_V).max()
            V = new_V

            if delta < 1e-2:
                break

        print("Policy Improvement...")
        # Policy Improvement
        stable = True
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):

                old_action = policy[i, j]

                # Try all actions between -5 to +5
                best_value = -1e10
                best_action = 0

                for action in range(-MAX_MOVE, MAX_MOVE + 1):
                    # Check movement feasibility
                    if action > i:               # cannot move more than available at L1
                        continue
                    if -action > j:             # cannot move more bikes from L2 than available
                        continue
                    if j + action > MAX_BIKES:  # after movement, L2 cannot exceed 20
                        continue
                    if i - action > MAX_BIKES:  # after movement, L1 cannot exceed 20
                        continue

                    val = expected_return((i, j), action, V)
                    if val > best_value:
                        best_value = val
                        best_action = action

                policy[i, j] = best_action
                if best_action != old_action:
                    stable = False

    return policy, V

# ------------------------------
# RUN & PRINT RESULTS
# ------------------------------
policy, V = policy_iteration()

print("\nFINAL OPTIMAL POLICY:")
print(policy)

print("\nFINAL VALUE FUNCTION:")
print(V)
