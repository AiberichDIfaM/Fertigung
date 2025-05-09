# File: flexible_jobshop_env.py
# Erweiterte Env: Potential-based Shaping und Goal-conditioned Low-Level
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from classes import Part
import networkx as nx

class FlexibleJobShopEnv(gym.Env):
    def __init__(self, anlage, max_buffer=10, max_steps=50, gamma=0.99, goal=None):
        super().__init__()
        self.anlage = anlage
        self.machines = self.anlage.machines
        self.global_buffer = self.anlage.global_buffer
        self.part_types = self.anlage.all_part_types
        self.gamma = gamma
        self.goal = goal  # goal part_type.name or None

        self.empty_marker = len(self.part_types)
        self.max_buffer = max_buffer
        self.max_steps = max_steps
        self.current_step = 0
        self.part_id_counter = 0

        # build production graph
        self.prod_graph = nx.DiGraph()
        for pt in self.part_types:
            self.prod_graph.add_node(pt.name)
        for m in self.machines:
            for tr in m.machine_type.transformations:
                for inp in tr.input_types:
                    self.prod_graph.add_edge(inp.name, tr.output_type.name)

        # transformations
        unique_trans = {}
        for m in self.machines:
            for t in m.machine_type.transformations:
                unique_trans[t] = True
        self.unique_transformations = list(unique_trans.keys())
        self.n_transformations = len(self.unique_transformations)
        self.n_machines = len(self.machines)
        self.n_actions = 1 + self.n_machines * self.n_transformations
        self.action_space = spaces.Discrete(self.n_actions)

        obs_dim = self.max_buffer + self.n_machines*(3 + len(self.part_types)) + len(self.part_types)
        self.observation_space = spaces.Box(0, 100, shape=(obs_dim,), dtype=np.float32)
        self.final_mapping = {pt.name: pt.name in ['fp1','fp2'] for pt in self.part_types}
        self.last_profit = self._calculate_profit()

    def phi(self):
        # potential: sum over buffer exp(-dist to goal)
        if self.goal is None:
            return 0.0
        total = 0.0
        for p in self.global_buffer:
            try:
                d = nx.shortest_path_length(self.prod_graph, p.type.name, self.goal)
            except nx.NetworkXNoPath:
                d = np.inf
            total += np.exp(-d) if np.isfinite(d) else 0.0
        return total

    def _calculate_profit(self):
        val = cost = 0.0
        for buf in [self.global_buffer] + [m.input_buffer for m in self.machines] + [m.output_buffer for m in self.machines]:
            for p in buf:
                val += p.type.value; cost += p.type.cost
        for m in self.machines:
            for job in m.current_jobs:
                for p in job['input_parts']:
                    val += p.type.value; cost += p.type.cost
        return val - cost

    def _get_observation(self):
        obs = []
        # global buffer
        for i in range(self.max_buffer):
            if i < len(self.global_buffer):
                idx = next((j for j,pt in enumerate(self.part_types) if pt.name==self.global_buffer[i].type.name), self.empty_marker)
            else:
                idx = self.empty_marker
            obs.append(float(idx))
        # machine features
        for m in self.machines:
            bf = [len(m.input_buffer), len(m.output_buffer), len(m.current_jobs)]
            mask = []
            for pt in self.part_types:
                sup = any(any(ip.name==pt.name for ip in tr.input_types) for tr in m.machine_type.transformations)
                mask.append(1.0 if sup else 0.0)
            obs.extend(bf + mask)
        # goal one-hot
        goal_vec = [1.0 if pt.name==self.goal else 0.0 for pt in self.part_types]
        obs.extend(goal_vec)
        return np.array(obs, dtype=np.float32)

    def get_action_mask(self):
        mask = np.zeros(self.n_actions, dtype=np.int8)
        mask[0] = 1
        avail = {}
        for p in self.global_buffer:
            avail[p.type.name] = avail.get(p.type.name,0) + 1
        for k in range(1, self.n_actions):
            mi = (k-1)//self.n_transformations
            ti = (k-1)%self.n_transformations
            trans = self.unique_transformations[ti]
            req = {}
            for pt in trans.input_types:
                req[pt.name] = req.get(pt.name,0) + 1
            if all(avail.get(n,0)>=c for n,c in req.items()): mask[k] = 1
        return mask

    def step(self, action):
        prev_profit = self._calculate_profit()
        prev_phi = self.phi()
        # iterative execution
        count=0
        mask = self.get_action_mask()
        while action>0 and count<self.max_buffer and mask[action]:
            mi = (action-1)//self.n_transformations
            ti = (action-1)%self.n_transformations
            trans = self.unique_transformations[ti]
            req = {}
            for pt in trans.input_types:
                req[pt.name] = req.get(pt.name,0)+1
            newbuf=[]; collected=[]; tmp=req.copy()
            for p in self.global_buffer:
                if tmp.get(p.type.name,0)>0:
                    collected.append(p); tmp[p.type.name]-=1
                else:
                    newbuf.append(p)
            self.global_buffer = newbuf
            self.machines[mi].input_buffer.extend(collected)
            self.anlage.global_buffer=self.global_buffer
            self.anlage.refill_global_buffer(self.max_buffer)
            self.global_buffer=self.anlage.global_buffer
            mask = self.get_action_mask(); count+=1
        # machine progress
        for m in self.machines:
            if len(m.current_jobs)<m.machine_type.slots:
                for t in m.transformation_priority:
                    if m.can_start_transformation(t):
                        self.part_id_counter=m.start_transformation(t,self.part_id_counter); break
            self.part_id_counter,_=m.progress_jobs(self.part_id_counter,self.final_mapping)
            while m.output_buffer:
                p=m.output_buffer.pop(0)
                if not self.final_mapping.get(p.type.name,False): self.global_buffer.append(p)
        # refill end
        self.anlage.global_buffer=self.global_buffer
        self.anlage.refill_global_buffer(self.max_buffer)
        self.global_buffer=self.anlage.global_buffer
        # compute rewards
        cur_profit = self._calculate_profit()
        cur_phi = self.phi()
        r_env = cur_profit - prev_profit
        r_shape = self.gamma*cur_phi - prev_phi
        reward = r_env + r_shape
        self.current_step+=1
        done = self.current_step>=self.max_steps
        info = {"action_mask": self.get_action_mask()}
        return self._get_observation(), reward, done, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self.current_step = 0
        # Reset Anlage und initial Global-Puffer füllen
        self.anlage.reset()
        self.anlage.refill_global_buffer(self.max_buffer)
        self.global_buffer = self.anlage.global_buffer
        self.part_id_counter = 0
        return self._get_observation(), {"action_mask": self.get_action_mask()}

    def render(self, mode="human"):
        print(f"Step {self.current_step}, Goal {self.goal}, Profit {self._calculate_profit()}")

