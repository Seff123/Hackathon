"""Planner controllers: BayesGreedyPlanner, HeuristicFloodPlanner, SurvivalPlanner (section 11).

All three share FleetBrain (belief, detection, scenarios, scoring, claims); they differ only in the per-cell
value function: 1 (expected finds), U (appendix bonus) or V (marginal survival)."""
from __future__ import annotations

import dataclasses

from ..controllers import Controller
from ..runner import register
from .fleet import FleetBrain
from .params import PlannerParams, SwarmParams


class _PlannerBase(Controller):
    mode = "survival"
    name = "survival"

    def __init__(self, params: PlannerParams | None = None, swarm: SwarmParams | None = None):
        self.params = params or PlannerParams()
        self.swarm = swarm or SwarmParams()
        self.brain = FleetBrain(self.mode, self.params, self.swarm, name=self.name)

    @property
    def decision_log(self):
        return self.brain.decision_log

    def reset(self, views):
        v0 = next(iter(views.values()))
        self.brain._setup(v0)
        self.brain.observe({i: v for i, v in views.items()})
        return self.brain.decide(views)

    def observe(self, views_all):
        self.brain.observe(views_all)

    def decide(self, views_due):
        return self.brain.decide(views_due)

    def debug_snapshot(self, t):
        return self.brain.debug_snapshot(t)

    def diagnostics(self) -> dict:
        b = self.brain
        if not b.ready:
            return {}
        return {"w_eff": [round(float(x), 4) for x in b.belief.w_eff()], "count_scale": round(b.belief.count_scale(), 4),
                "bias_hat": round(b.bias.mean(), 4), "lost_mass": float(b.belief.lost_mass.sum()), **b.diag}

    def params_dict(self):
        from ..config import to_jsonable
        return {"planner": to_jsonable(self.params), "swarm": to_jsonable(self.swarm)}


class BayesGreedyPlanner(_PlannerBase):
    mode = "bayes"
    name = "bayes"


class HeuristicFloodPlanner(_PlannerBase):
    mode = "heuristic"
    name = "heuristic"


class SurvivalPlanner(_PlannerBase):
    mode = "survival"
    name = "survival"


ABLATIONS = {
    "no_state_pd": {"state_pd": False},
    "no_scenarios": {"scenarios": False},
    "no_bias": {"bias_learning": False},
    "scen_bonus": {"beta_scen": 1.0},
    "no_prediction": {"prediction": False},
}


def _build(cls, cfg, params: dict | None = None, swarm: dict | None = None, ablations=(), comms=None, **kw):
    p = PlannerParams(e_soar=cfg.controller.e_soar, w_enter=cfg.controller.w_enter)
    for a in ablations or ():
        p = dataclasses.replace(p, **ABLATIONS[a])
    if params:
        p = dataclasses.replace(p, **params)
    s = SwarmParams()
    if swarm:
        s = dataclasses.replace(s, **swarm)
    if comms:
        s = dataclasses.replace(s, comms_mode=comms)
    c = cls(p, s)
    if ablations or comms:
        c.name = cls.name + "".join(f"+{a}" for a in ablations or ()) + (f"+comms_{comms}" if comms else "")
    return c


register("bayes")(lambda cfg, **kw: _build(BayesGreedyPlanner, cfg, **kw))
register("heuristic")(lambda cfg, **kw: _build(HeuristicFloodPlanner, cfg, **kw))
register("survival")(lambda cfg, **kw: _build(SurvivalPlanner, cfg, **kw))
