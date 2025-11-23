import json
import random
import math
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

# --- Board Utils ---

def to_chars(s):
    return list(s)

def to_str(l):
    return ''.join(l)

def rot_cw(b):
    # Rotate board 90 deg
    return [b[6], b[3], b[0],
            b[7], b[4], b[1],
            b[8], b[5], b[2]]

def flip_h(b):
    # Mirror board
    return [b[2], b[1], b[0],
            b[5], b[4], b[3],
            b[8], b[7], b[6]]

def gen_variants(b):
    curr = b[:]
    for _ in range(4):
        yield curr
        yield flip_h(curr)
        curr = rot_cw(curr)

def get_min_state(b):
    min_s = None
    for v in gen_variants(b):
        s_val = ''.join(v)
        if min_s is None or s_val < min_s:
            min_s = s_val
    return min_s

def get_empties(b):
    return [idx for idx, val in enumerate(b) if val == '.']

def eval_state(b):
    wins = [(0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)]

    for x, y, z in wins:
        if b[x] != '.' and b[x] == b[y] == b[z]:
            return b[x], False

    if '.' not in b:
        return None, True
    return None, False

# --- MENACE Implementation ---

class MatchboxLearner:
    def __init__(self, start_wt=4, r_win=3, r_draw=1, r_loss=1, min_wt=1):
        self.memory = {}
        self.start_wt = start_wt
        self.rew_win = r_win
        self.rew_draw = r_draw
        self.pen_loss = r_loss
        self.floor = min_wt

    def _check_box(self, state_key):
        if state_key not in self.memory:
            opts = get_empties(list(state_key))
            self.memory[state_key] = Counter({o: self.start_wt for o in opts})

    def pick_action(self, state_key):
        self._check_box(state_key)
        current_box = self.memory[state_key]

        if not current_box:
            opts = get_empties(list(state_key))
            self.memory[state_key] = Counter({o: self.start_wt for o in opts})
            current_box = self.memory[state_key]

        w_sum = sum(current_box.values())
        rand_val = random.uniform(0, w_sum)
        run_sum = 0.0

        for act, wt in current_box.items():
            run_sum += wt
            if rand_val <= run_sum:
                return act

        return random.choice(list(current_box.keys()))

    def run_match(self, opp_mode='random'):
        grid = ['.'] * 9
        trace = []
        # 0 is Learner (X), 1 is Opponent (O)
        curr_p = 0

        while True:
            w, d = eval_state(grid)
            if w == 'X':
                self._learn(trace, 1)
                return 1
            elif w == 'O':
                self._learn(trace, -1)
                return -1
            elif d:
                self._learn(trace, 0)
                return 0

            if curr_p == 0:
                s_rep = get_min_state(grid)
                act = self.pick_action(s_rep)
                grid[act] = 'X'
                trace.append((s_rep, act))
            else:
                valid = get_empties(grid)
                if opp_mode == 'greedy':
                    if grid[4] == '.':
                        grid[4] = 'O'
                    else:
                        crnrs = [c for c in [0,2,6,8] if grid[c] == '.']
                        if crnrs:
                            grid[random.choice(crnrs)] = 'O'
                        else:
                            grid[random.choice(valid)] = 'O'
                else:
                    grid[random.choice(valid)] = 'O'

            curr_p = 1 - curr_p

    def _learn(self, path, res):
        for s, a in path:
            if s not in self.memory: continue

            if res == 1:
                self.memory[s][a] += self.rew_win
            elif res == 0:
                self.memory[s][a] += self.rew_draw
            else:
                self.memory[s][a] = max(self.floor, self.memory[s][a] - self.pen_loss)

    def practice(self, episodes=1000, mode='random', debug=False):
        outcomes = []
        for e in range(episodes):
            outcomes.append(self.run_match(mode))
            if debug and (e+1) % (episodes//10) == 0:
                pass
        return outcomes

# --- Bandits ---

class BaseBandit:
    def __init__(self, probs, seed=None):
        self.p_vals = np.array(probs, dtype=float)
        if seed:
            np.random.seed(seed)

    def act(self, idx):
        rnd = random.random()
        return 1 if rnd < self.p_vals[idx] else 0

class BanditOne(BaseBandit):
    def __init__(self, seed=None):
        super().__init__([0.7, 0.3], seed)

class BanditTwo(BaseBandit):
    def __init__(self, seed=None):
        super().__init__([0.2, 0.8], seed)

class WalkerBandit:
    def __init__(self, arms=10, start=0.0, drift_sd=0.01, noise=1.0, seed=None):
        self.n_arms = arms
        self.means = np.full(arms, start, dtype=float)
        self.d_sd = drift_sd
        self.n_sd = noise
        if seed:
            np.random.seed(seed)

    def drift(self):
        shift = np.random.normal(0.0, self.d_sd, self.n_arms)
        self.means += shift

    def act(self, idx):
        self.drift()
        return np.random.normal(self.means[idx], self.n_sd)

    def get_opt(self):
        return np.argmax(self.means)

# --- Agents ---

class AvgGreedy:
    def __init__(self, n_act=2, eps=0.1):
        self.n = n_act
        self.e = eps
        self.q_est = np.zeros(n_act)
        self.counts = np.zeros(n_act, dtype=int)

    def choose(self):
        if random.random() < self.e:
            return random.randrange(self.n)
        mx = np.max(self.q_est)
        opts = np.where(self.q_est == mx)[0]
        return int(np.random.choice(opts))

    def learn(self, a, r):
        self.counts[a] += 1
        err = r - self.q_est[a]
        self.q_est[a] += err / self.counts[a]

    def clear(self):
        self.q_est.fill(0)
        self.counts.fill(0)

class AlphaGreedy:
    def __init__(self, n_act=10, eps=0.1, step_sz=0.1):
        self.n = n_act
        self.e = eps
        self.lr = step_sz
        self.q_est = np.zeros(n_act)

    def choose(self):
        if random.random() < self.e:
            return random.randrange(self.n)
        mx = np.max(self.q_est)
        opts = np.where(self.q_est == mx)[0]
        return int(np.random.choice(opts))

    def learn(self, a, r):
        err = r - self.q_est[a]
        self.q_est[a] += self.lr * err

    def clear(self):
        self.q_est.fill(0)

# --- Execution ---

def test_stat_bandit(env_class, agent, steps=1000, trials=100, seed=None):
    hist_rew = np.zeros(steps)
    hist_opt = np.zeros(steps)

    for i in range(trials):
        s_val = (seed + i) if seed is not None else None
        if s_val:
            random.seed(s_val)
            np.random.seed(s_val)

        env = env_class()
        agent.clear()
        best_arm = np.argmax(env.p_vals)

        for t in range(steps):
            act = agent.choose()
            rew = env.act(act)
            agent.learn(act, rew)
            hist_rew[t] += rew
            if act == best_arm:
                hist_opt[t] += 1

    return (hist_rew / trials), (hist_opt / trials * 100)

def test_drift_bandit(env_gen, agent, steps=10000, trials=50, seed=None):
    hist_rew = np.zeros(steps)
    hist_opt = np.zeros(steps)

    for i in range(trials):
        s_val = (seed + i) if seed is not None else None
        if s_val:
            random.seed(s_val)
            np.random.seed(s_val)

        env = env_gen()
        agent.clear()

        for t in range(steps):
            act = agent.choose()
            rew = env.act(act)
            agent.learn(act, rew)
            hist_rew[t] += rew

            if act == env.get_opt():
                hist_opt[t] += 1

    return (hist_rew / trials), (hist_opt / trials * 100)

def viz_data(y_vals, y_opt=None, head='Data', x_lab='Iter', f_name=None):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(y_vals, color='navy', alpha=0.7)
    plt.title(f'{head}: Reward')
    plt.xlabel(x_lab)
    plt.ylabel('Avg R')

    if y_opt is not None:
        plt.subplot(1, 2, 2)
        plt.plot(y_opt, color='darkgreen', alpha=0.7)
        plt.title(f'{head}: % Optimal')
        plt.xlabel(x_lab)
        plt.ylabel('%')
        plt.ylim(0, 100)

    plt.tight_layout()
    if f_name:
        plt.savefig(f"{f_name}.png")
        print(f"Saved: {f_name}.png")

    # Only try showing if display available
    try:
        plt.show()
    except:
        pass

def main_driver():
    print(">>> LAB 7 START <<<")

    # Part 1
    print("\n[1] Matchbox Engine Training...")
    bot = MatchboxLearner()
    res = bot.practice(episodes=500, debug=True)
    w_cnt = res.count(1)
    d_cnt = res.count(0)
    l_cnt = res.count(-1)
    print(f"Stats (500 games): W:{w_cnt} D:{d_cnt} L:{l_cnt}")

    # Part 2
    print("\n[2] Stationary Bandits...")
    p2_agent = AvgGreedy(n_act=2, eps=0.1)

    r_a, opt_a = test_stat_bandit(BanditOne, p2_agent, steps=500, trials=50, seed=0)
    viz_data(r_a, opt_a, head='Bandit A', f_name='bandit_a')

    r_b, opt_b = test_stat_bandit(BanditTwo, p2_agent, steps=500, trials=50, seed=1)
    viz_data(r_b, opt_b, head='Bandit B', f_name='bandit_b')

    # Part 3
    print("\n[3] Drifting Bandits...")
    def make_env():
        return WalkerBandit(arms=10)

    ag_std = AvgGreedy(n_act=10, eps=0.1)
    r_s, o_s = test_drift_bandit(make_env, ag_std, steps=2000, trials=20, seed=42)
    viz_data(r_s, o_s, head='Drift (Avg)', f_name='nonstat_sample_avg')

    ag_fix = AlphaGreedy(n_act=10, eps=0.1, step_sz=0.1)
    r_f, o_f = test_drift_bandit(make_env, ag_fix, steps=2000, trials=20, seed=42)
    viz_data(r_f, o_f, head='Drift (Alpha)', f_name='nonstat_alpha')

    print(">>> DONE <<<")

if __name__ == "__main__":
    main_driver()