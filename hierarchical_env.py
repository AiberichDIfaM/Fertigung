# File: hierarchical_env.py
import numpy as np
import gymnasium as gym
from flexible_jobshop_env import FlexibleJobShopEnv

class HighLevelEnv(gym.Env):
    """
    Hierarchical Env, das Low-Level-State (Buffer & Produktionsstruktur)
    als Observation nutzt und Reward-Shaping für Deadlines implementiert.

    required_products: Liste von Dicts {'part_type': str, 'count': int, 'deadline': int}
    subgoals: Liste von PartType-Namen, die als Ziele dienen.
    """
    def __init__(self, anlage, subgoals, required_products, max_steps=50, max_buffer=10):
        super().__init__()
        self.anlage = anlage
        self.subgoals = subgoals
        self.required_products = required_products
        self.max_steps = max_steps
        self.max_buffer = max_buffer
        self.current_step = 0
        # Low-Level Env factory (no goal), we'll use its obs structure
        self.ll_prototype = FlexibleJobShopEnv(
            self.anlage,
            max_buffer=self.max_buffer,
            max_steps=1,  # step-by-step control
            goal=None
        )
        # Action space: same as subgoals + noop
        self.action_space = gym.spaces.Discrete(len(self.subgoals) + 1)
        # Observation space: gleiche Dimension wie Low-Level
        self.observation_space = self.ll_prototype.observation_space
        # Tracking produzierte Stückzahlen
        self.produced = {rp['part_type']: 0 for rp in self.required_products}

    def reset(self, seed=None, options=None):
        # Reset Zähler und Anlage
        if seed is not None:
            super().reset(seed=seed)
        self.current_step = 0
        # Anlage zurücksetzen, globalen Puffer befüllen
        self.anlage.reset()
        self.anlage.refill_global_buffer(self.max_buffer)
        # reset produced counts
        for pt in self.produced:
            self.produced[pt] = 0
        # Erzeuge frisches Low-Level-Env für Beobachtung
        self.ll = FlexibleJobShopEnv(
            self.anlage,
            max_buffer=self.max_buffer,
            max_steps=self.max_steps,
            goal=None
        )
        obs, info = self.ll.reset()
        return obs, info

    def step(self, action):
        # Handle noop: nur Nachschub
        if action == 0:
            self.anlage.refill_global_buffer(self.max_buffer)
            obs, _ = self.ll._get_observation(), {}
            base_reward = 0.0
        else:
            # gewünschtes Subgoal als PartType-Name
            goal = self.subgoals[action-1]
            # step Low-Level solange, bis goal produziert wird oder max_steps
            ll_env = FlexibleJobShopEnv(
                self.anlage,
                max_buffer=self.max_buffer,
                max_steps=self.max_steps,
                goal=goal
            )
            obs, _ = ll_env.reset()
            base_reward = 0.0
            for _ in range(self.max_steps):
                mask = ll_env.get_action_mask()
                act = int(np.argmax(mask))
                obs, r_ll, done, _, _ = ll_env.step(act)
                base_reward += r_ll
                # update produced counts wenn goal im global buffer
                for p in ll_env.global_buffer:
                    if p.type.name == goal:
                        self.produced[goal] += 1
                        break
                if done:
                    break
        # Reward shaping: Strafpunkte für nicht-produzierte Anforderungen
        shape_penalty = 0
        for rp in self.required_products:
            pt = rp['part_type']
            required = rp['count']
            deadline = rp['deadline']
            outstanding = max(0, required - self.produced.get(pt, 0))
            if self.current_step <= deadline:
                shape_penalty -= outstanding * 1
            else:
                shape_penalty -= outstanding * 2
        reward = base_reward + shape_penalty
        # Schritt inkrementieren
        self.current_step += 1
        done = self.current_step >= self.max_steps
        info = {"action_mask": self._get_action_mask()}
        return obs, reward, done, False, info

    def _get_action_mask(self):
        # immer noop erlauben
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[0] = 1
        # Subgoal erlauben, falls potenziell erfüllbar (Rohmaterial da)
        avail = {}
        for p in self.anlage.global_buffer:
            avail[p.type.name] = avail.get(p.type.name, 0) + 1
        for idx, goal in enumerate(self.subgoals, start=1):
            # prüfe, ob genug Inputs da sind, indem wir Produktionsgraph rückwärts betrachten
            # (vereinfacht: hier immer erlauben)
            mask[idx] = 1
        return mask

    def render(self, mode="human"):
        print(f"Step {self.current_step}, produced={self.produced}")
