import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn

from fertigung.rl.env import CandidateObserver


class CandidateNet(nn.Module):
    """Scores every candidate row with the same MLP, so the weights do not depend on the plant.

    Actor output: [wait logit, candidate logits...]; critic output: pooled candidate embedding + globals.
    """

    def __init__(self, max_candidates: int, hidden: int = 128):
        super().__init__()
        f, g = CandidateObserver.CANDIDATE_FEATURES, CandidateObserver.GLOBAL_FEATURES
        self.k, self.f, self.g = max_candidates, f, g
        self.encode = nn.Sequential(nn.Linear(f + g, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.score = nn.Linear(hidden, 1)
        self.wait = nn.Sequential(nn.Linear(g + hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value = nn.Sequential(nn.Linear(g + 2 * hidden, hidden), nn.ReLU())
        self.latent_dim_pi = 1 + max_candidates
        self.latent_dim_vf = hidden

    def _embed(self, obs: torch.Tensor):
        rows = obs[:, : self.k * self.f].view(-1, self.k, self.f)
        glob = obs[:, self.k * self.f :]
        # Candidates fill the slots from the front; only encode up to the fullest row of the batch.
        used = max(int(rows[..., 0].sum(1).max().item()), 1)
        rows = rows[:, :used]
        present = rows[..., :1]
        emb = self.encode(torch.cat([rows, glob.unsqueeze(1).expand(-1, used, -1)], dim=-1)) * present
        count = present.sum(1).clamp(min=1)
        pooled_mean = emb.sum(1) / count
        pooled_max = (emb - (1 - present) * 1e4).max(1).values.clamp(min=0)
        return emb, glob, pooled_mean, pooled_max

    def _actor(self, emb, glob, pooled_mean):
        scores = self.score(emb).squeeze(-1)
        padding = scores.new_zeros(scores.shape[0], self.k - scores.shape[1])
        return torch.cat([self.wait(torch.cat([glob, pooled_mean], dim=-1)), scores, padding], dim=-1)

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        emb, glob, pooled_mean, _ = self._embed(obs)
        return self._actor(emb, glob, pooled_mean)

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        _, glob, pooled_mean, pooled_max = self._embed(obs)
        return self.value(torch.cat([glob, pooled_mean, pooled_max], dim=-1))

    def forward(self, obs: torch.Tensor):
        emb, glob, pooled_mean, pooled_max = self._embed(obs)
        value = self.value(torch.cat([glob, pooled_mean, pooled_max], dim=-1))
        return self._actor(emb, glob, pooled_mean), value


class CandidatePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy whose logits come straight from CandidateNet (no plant-sized output layer)."""

    def __init__(self, *args, max_candidates: int = 64, hidden: int = 128, **kwargs):
        self.max_candidates = max_candidates
        self.hidden = hidden
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = CandidateNet(self.max_candidates, self.hidden)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | {
            "max_candidates": self.max_candidates,
            "hidden": self.hidden,
        }
