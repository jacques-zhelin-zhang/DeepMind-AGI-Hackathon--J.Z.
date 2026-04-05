"""
Veridical Worlds Benchmark — Kaggle Community Benchmarks Notebook
=================================================================
Self-contained submission for the DeepMind "Measuring Progress Toward AGI" hackathon.
Tests metacognition via procedurally generated micro-universes with a physics
perception ladder from simple linear dynamics to complex nonlinear phenomena.

Author: Zhelin Zhang
Track: Metacognition (primary), Learning (secondary)
"""

# %% [markdown]
# # Veridical Worlds Benchmark
# ### Physics Perception Ladder: Simple → Complex → Quantitative

# %% --- Imports ---
import json
import re
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any, Optional
from enum import Enum

import kaggle_benchmarks as kbench

# %% --- Engine: 10 Law Types with Difficulty Gating ---

class LawType(Enum):
    MOTION = "motion"
    INTERACTION = "interaction"
    CONSERVATION = "conservation"
    THRESHOLD = "threshold"
    CAUSAL = "causal"
    DECAY = "decay"
    FIELD_EFFECT = "field_effect"
    INVERSE_SQUARE = "inverse_square"
    OSCILLATION = "oscillation"
    NONLINEAR_MOTION = "nonlinear_motion"

@dataclass
class Particle:
    pid: int; x: float; y: float; vx: float; vy: float
    mass: float; charge: float; kind: int; energy: float = 10.0
    alive: bool = True; age: int = 0
    def to_dict(self):
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}

@dataclass
class UniverseLaw:
    law_id: str; category: str; description: str
    params: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass
class WorldState:
    step: int; particles: List[Particle]; global_energy: float = 0.0
    global_field: float = 0.0; events: List[str] = field(default_factory=list)
    def to_dict(self):
        return {"step": self.step,
                "particles": [p.to_dict() for p in self.particles if p.alive],
                "global_energy": round(self.global_energy, 4),
                "global_field": round(self.global_field, 4),
                "events": self.events}


class MicroUniverse:
    """Procedurally generated micro-universe with 10 law types and difficulty gating."""
    GRID = 20.0
    MAX_ACCEL = 2.0

    def __init__(self, seed, difficulty="medium"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        cfg = {"easy": (3, 4, 4, 6), "medium": (4, 6, 5, 8), "hard": (5, 7, 6, 10)}
        ml, xl, mp, xp = cfg.get(difficulty, cfg["medium"])
        self.num_laws = self.rng.randint(ml, xl)
        self.num_particles = self.rng.randint(mp, xp)
        self.laws, self.particles, self.history = [], [], []
        self.pending_causal = []  # (fire_step, effect_params) for delayed causal effects
        self._gen_laws()
        self._gen_particles()

    # --- Law Generators ---
    def _gen_laws(self):
        self.laws.append(self._motion_law())
        self.laws.append(self._interaction_law())

        base_gens = [self._motion_law, self._interaction_law, self._conservation_law,
                     self._threshold_law, self._causal_law, self._decay_law, self._field_effect_law]
        nonlinear_gens = []

        if self.difficulty in ("medium", "hard"):
            nonlinear_gens.append(self.rng.choice([self._inverse_square_law, self._oscillation_law]))
        if self.difficulty == "hard":
            nonlinear_gens = [self._inverse_square_law, self._oscillation_law, self._nonlinear_motion_law]

        # Guarantee at least one nonlinear law for medium/hard
        if nonlinear_gens:
            self.laws.append(self.rng.choice(nonlinear_gens)())
            remaining = self.num_laws - 3
        else:
            remaining = self.num_laws - 2

        available = base_gens + nonlinear_gens
        for _ in range(remaining):
            self.laws.append(self.rng.choice(available)())
        for i, l in enumerate(self.laws):
            l.law_id = f"L{i+1:02d}"

    def _motion_law(self):
        drivers = self.rng.sample(["mass", "charge", "neighbor_count", "kind"], k=self.rng.randint(1, 3))
        coeffs = {d: round(self.rng.uniform(-0.5, 0.5), 3) for d in drivers}
        damp = round(self.rng.uniform(0.7, 0.99), 3)
        affects = self.rng.choice(["vx", "vy", "both"])
        parts = [f"{'+' if c >= 0 else ''}{c}*{d}" for d, c in coeffs.items()]
        desc = f"Velocity ({'both axes' if affects == 'both' else affects}) += {' '.join(parts)}, damping={damp}."
        return UniverseLaw("", "motion", desc, {"drivers": drivers, "coefficients": coeffs, "damping": damp, "affects": affects})

    def _interaction_law(self):
        table = {}
        for _ in range(self.rng.randint(2, 4)):
            k1, k2 = self.rng.randint(0, 3), self.rng.randint(0, 3)
            pair = f"{min(k1, k2)}-{max(k1, k2)}"
            table[pair] = self.rng.choice(["bounce", "merge", "annihilate", "spawn"])
        default = self.rng.choice(["bounce", "ignore"])
        radius = round(self.rng.uniform(1.0, 3.0), 2)
        desc = f"Interaction radius={radius}, default={default}. " + " ".join(f"Kind {p}->{o}" for p, o in table.items())
        return UniverseLaw("", "interaction", desc, {"table": table, "default": default, "radius": radius})

    def _conservation_law(self):
        qty = self.rng.choice(["energy", "momentum", "charge"])
        conserved = self.rng.choice([True, False])
        leak = 0 if conserved else round(self.rng.uniform(0.01, 0.1), 3)
        desc = f"{qty.capitalize()} {'IS' if conserved else 'NOT'} conserved" + (f", leak={leak}/step" if not conserved else "") + "."
        return UniverseLaw("", "conservation", desc, {"quantity": qty, "conserved": conserved, "leak_rate": leak})

    def _threshold_law(self):
        trigger = self.rng.choice(["speed", "energy", "neighbor_count", "charge"])
        thresh = round(self.rng.uniform(1.0, 5.0), 2)
        effect = self.rng.choice(["split", "freeze", "change_kind", "boost"])
        nk = self.rng.randint(0, 3) if effect == "change_kind" else None
        desc = f"When {trigger}>{thresh}, particle undergoes '{effect}'" + (f" to kind {nk}" if nk is not None else "") + "."
        return UniverseLaw("", "threshold", desc, {"trigger": trigger, "threshold": thresh, "effect": effect, "new_kind": nk})

    def _causal_law(self):
        cause = self.rng.choice(["collision", "threshold_breach", "particle_death"])
        effect = self.rng.choice(["spawn_particle", "global_energy_shift", "velocity_pulse"])
        lag = self.rng.randint(1, 3)
        prob = round(self.rng.uniform(0.3, 1.0), 2)
        desc = f"When '{cause}' occurs, after {lag} step(s), {prob*100:.0f}% chance of '{effect}'."
        return UniverseLaw("", "causal", desc, {"cause": cause, "effect": effect, "lag": lag, "probability": prob})

    def _decay_law(self):
        half_life = self.rng.randint(5, 20)
        product = self.rng.choice(["energy_release", "split", "vanish"])
        applies_to = self.rng.choice(["all", "charged", "heavy"])
        desc = f"{applies_to.capitalize()} particles: half-life={half_life} steps, product='{product}'."
        return UniverseLaw("", "decay", desc, {"half_life": half_life, "product": product, "applies_to": applies_to})

    def _field_effect_law(self):
        source = self.rng.choice(["total_charge", "particle_count", "total_energy"])
        scaling = round(self.rng.uniform(0.01, 0.2), 3)
        effect_on = self.rng.choice(["velocity", "energy", "charge"])
        desc = f"Global field from {source} (x{scaling}) affects {effect_on}."
        return UniverseLaw("", "field_effect", desc, {"source": source, "scaling": scaling, "effect_on": effect_on})

    def _inverse_square_law(self):
        G = round(self.rng.uniform(0.05, 0.5), 3)
        applies_to = self.rng.choice(["mass", "charge"])
        cutoff = round(self.rng.uniform(1.0, 3.0), 2)
        attractive = self.rng.choice([True, False])
        desc = (f"{'Attractive' if attractive else 'Repulsive'} inverse-square force on {applies_to}: "
                f"F={'−' if attractive else '+'}{G}*q1*q2/r^2, cutoff={cutoff}.")
        return UniverseLaw("", "inverse_square", desc,
                           {"force_constant": G, "applies_to": applies_to, "cutoff": cutoff, "attractive": attractive})

    def _oscillation_law(self):
        k = round(self.rng.uniform(0.02, 0.2), 3)
        eq_x = round(self.rng.uniform(5.0, 15.0), 2)
        eq_y = round(self.rng.uniform(5.0, 15.0), 2)
        damp = round(self.rng.uniform(0.9, 0.99), 3)
        applies_to = self.rng.choice(["all", "charged", "heavy"])
        desc = f"Restoring force on {applies_to} toward ({eq_x},{eq_y}), k={k}, damping={damp}."
        return UniverseLaw("", "oscillation", desc,
                           {"spring_constant": k, "equilibrium_x": eq_x, "equilibrium_y": eq_y, "damping": damp, "applies_to": applies_to})

    def _nonlinear_motion_law(self):
        formula = self.rng.choice(["quadratic", "logarithmic"])
        a = round(self.rng.uniform(-0.1, 0.1), 4)
        b = round(self.rng.uniform(-0.3, 0.3), 4)
        c = round(self.rng.uniform(-0.15, 0.15), 4)
        axis = self.rng.choice(["vx", "vy", "both"])
        if formula == "quadratic":
            desc = f"Nonlinear (quadratic) on {axis}: dv = {a}*mass^2 + {b}*log(energy+1) + {c}*charge^2."
        else:
            desc = f"Nonlinear (logarithmic) on {axis}: dv = {a}*exp(mass/5) + {b}*sqrt(|charge|+0.1) + {c}."
        return UniverseLaw("", "nonlinear_motion", desc, {"formula": formula, "a": a, "b": b, "c": c, "axis": axis})

    def _gen_particles(self):
        for i in range(self.num_particles):
            self.particles.append(Particle(
                i, round(self.rng.uniform(1, 19), 2), round(self.rng.uniform(1, 19), 2),
                round(self.rng.uniform(-1, 1), 3), round(self.rng.uniform(-1, 1), 3),
                round(self.rng.uniform(0.5, 5.0), 2), round(self.rng.uniform(-2, 2), 2),
                self.rng.randint(0, 3), round(self.rng.uniform(5, 20), 2)))

    # --- Simulation ---
    def simulate(self, steps=20):
        self.history = []
        particles = deepcopy(self.particles)
        for t in range(steps):
            events = []
            # 1. Field effects: update global field
            global_field = 0.0
            for law in self.laws:
                if law.category == "field_effect":
                    p = law.params
                    alive = [x for x in particles if x.alive]
                    if p["source"] == "total_charge":
                        src = sum(x.charge for x in alive)
                    elif p["source"] == "particle_count":
                        src = len(alive)
                    else:
                        src = sum(x.energy for x in alive)
                    global_field = round(src * p["scaling"], 4)
                    # Apply field to entities
                    for x in alive:
                        delta = global_field * p["scaling"]
                        if p["effect_on"] == "velocity":
                            x.vx = round(x.vx + delta, 4)
                        elif p["effect_on"] == "energy":
                            x.energy = round(max(0, x.energy + delta), 4)
                        elif p["effect_on"] == "charge":
                            x.charge = round(x.charge + delta * 0.1, 4)

            # 2. Motion laws
            for law in self.laws:
                if law.category == "motion":
                    for x in particles:
                        if not x.alive: continue
                        nc = sum(1 for q in particles if q.alive and q.pid != x.pid
                                 and math.sqrt((x.x-q.x)**2+(x.y-q.y)**2) < 3)
                        am = {"mass": x.mass, "charge": x.charge, "neighbor_count": nc, "kind": x.kind}
                        d = sum(law.params["coefficients"].get(dr, 0) * am.get(dr, 0) for dr in law.params["drivers"])
                        if law.params["affects"] in ("vx", "both"):
                            x.vx = round(x.vx * law.params["damping"] + d, 4)
                        if law.params["affects"] in ("vy", "both"):
                            x.vy = round(x.vy * law.params["damping"] + d, 4)

            # 3. Inverse-square forces (N-body pairwise)
            for law in self.laws:
                if law.category == "inverse_square":
                    p = law.params
                    alive = [x for x in particles if x.alive]
                    for i, e1 in enumerate(alive):
                        ax, ay = 0.0, 0.0
                        for j, e2 in enumerate(alive):
                            if i == j: continue
                            dx = e2.x - e1.x
                            dy = e2.y - e1.y
                            r_sq = dx*dx + dy*dy
                            r = math.sqrt(r_sq)
                            if r < 0.5: r, r_sq = 0.5, 0.25  # softening
                            if r > p["cutoff"] * 5: continue
                            q1 = e1.mass if p["applies_to"] == "mass" else e1.charge
                            q2 = e2.mass if p["applies_to"] == "mass" else e2.charge
                            fmag = p["force_constant"] * q1 * q2 / r_sq
                            sign = -1.0 if p["attractive"] else 1.0
                            ax += sign * fmag * dx / r / max(e1.mass, 0.1)
                            ay += sign * fmag * dy / r / max(e1.mass, 0.1)
                        ax = max(-self.MAX_ACCEL, min(self.MAX_ACCEL, ax))
                        ay = max(-self.MAX_ACCEL, min(self.MAX_ACCEL, ay))
                        e1.vx = round(e1.vx + ax, 4)
                        e1.vy = round(e1.vy + ay, 4)

            # 4. Oscillation (restoring force)
            for law in self.laws:
                if law.category == "oscillation":
                    p = law.params
                    for x in particles:
                        if not x.alive: continue
                        applies = (p["applies_to"] == "all"
                                   or (p["applies_to"] == "charged" and abs(x.charge) > 0.1)
                                   or (p["applies_to"] == "heavy" and x.mass > 2.5))
                        if not applies: continue
                        dx = x.x - p["equilibrium_x"]
                        dy = x.y - p["equilibrium_y"]
                        x.vx = round(x.vx * p["damping"] - p["spring_constant"] * dx, 4)
                        x.vy = round(x.vy * p["damping"] - p["spring_constant"] * dy, 4)

            # 5. Nonlinear motion
            for law in self.laws:
                if law.category == "nonlinear_motion":
                    p = law.params
                    for x in particles:
                        if not x.alive: continue
                        if p["formula"] == "quadratic":
                            dv = p["a"] * x.mass**2 + p["b"] * math.log(x.energy + 1) + p["c"] * x.charge**2
                        else:
                            dv = p["a"] * math.exp(min(x.mass / 5, 3)) + p["b"] * math.sqrt(abs(x.charge) + 0.1) + p["c"]
                        dv = max(-self.MAX_ACCEL, min(self.MAX_ACCEL, dv))
                        if p["axis"] in ("vx", "both"): x.vx = round(x.vx + dv, 4)
                        if p["axis"] in ("vy", "both"): x.vy = round(x.vy + dv, 4)

            # 6. Move particles + boundary wrap
            for x in particles:
                if not x.alive: continue
                x.x = round((x.x + x.vx) % self.GRID, 4)
                x.y = round((x.y + x.vy) % self.GRID, 4)
                x.age += 1

            # 7. Interactions
            for law in self.laws:
                if law.category == "interaction":
                    alive = [x for x in particles if x.alive]
                    for i, p1 in enumerate(alive):
                        for j, p2 in enumerate(alive):
                            if i >= j: continue
                            if math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2) > law.params["radius"]: continue
                            key = f"{min(p1.kind,p2.kind)}-{max(p1.kind,p2.kind)}"
                            out = law.params["table"].get(key, law.params["default"])
                            if out == "bounce":
                                p1.vx, p2.vx = round(p2.vx*0.8, 4), round(p1.vx*0.8, 4)
                                p1.vy, p2.vy = round(p2.vy*0.8, 4), round(p1.vy*0.8, 4)
                                events.append(f"Step {t}: P{p1.pid}&P{p2.pid} BOUNCE")
                            elif out == "merge":
                                p1.mass = round(p1.mass + p2.mass, 2)
                                p1.energy = round(p1.energy + p2.energy, 2)
                                p2.alive = False
                                events.append(f"Step {t}: P{p1.pid}&P{p2.pid} MERGE")
                            elif out == "annihilate":
                                p1.alive = False; p2.alive = False
                                events.append(f"Step {t}: P{p1.pid}&P{p2.pid} ANNIHILATE")
                            elif out == "spawn":
                                new_p = Particle(len(particles), round((p1.x+p2.x)/2, 4), round((p1.y+p2.y)/2, 4),
                                                 0.0, 0.0, round((p1.mass+p2.mass)/4, 2), 0.0,
                                                 self.rng.randint(0, 3), round((p1.energy+p2.energy)/4, 2))
                                particles.append(new_p)
                                events.append(f"Step {t}: P{p1.pid}&P{p2.pid} SPAWN P{new_p.pid}")

            # 8. Threshold effects
            for law in self.laws:
                if law.category == "threshold":
                    for x in particles:
                        if not x.alive: continue
                        spd = math.sqrt(x.vx**2 + x.vy**2)
                        val = {"speed": spd, "energy": x.energy,
                               "neighbor_count": sum(1 for q in particles if q.alive and q.pid != x.pid
                                                     and math.sqrt((x.x-q.x)**2+(x.y-q.y)**2) < 3),
                               "charge": abs(x.charge)}.get(law.params["trigger"], 0)
                        if val > law.params["threshold"]:
                            eff = law.params["effect"]
                            if eff == "freeze": x.vx, x.vy = 0, 0
                            elif eff == "boost": x.vx = round(x.vx*1.5, 4); x.vy = round(x.vy*1.5, 4)
                            elif eff == "change_kind" and law.params["new_kind"] is not None: x.kind = law.params["new_kind"]
                            elif eff == "split":
                                new_p = Particle(len(particles), round(x.x+0.5, 4), round(x.y+0.5, 4),
                                                 round(x.vx*0.5, 4), round(x.vy*0.5, 4),
                                                 round(x.mass/2, 2), round(x.charge/2, 2), x.kind, round(x.energy/2, 2))
                                x.mass = round(x.mass/2, 2); x.energy = round(x.energy/2, 2)
                                particles.append(new_p)
                            events.append(f"Step {t}: P{x.pid} {eff} ({law.params['trigger']}={val:.2f}>{law.params['threshold']})")

            # 9. Decay
            for law in self.laws:
                if law.category == "decay":
                    p = law.params
                    for x in particles:
                        if not x.alive: continue
                        applies = (p["applies_to"] == "all"
                                   or (p["applies_to"] == "charged" and abs(x.charge) > 0.1)
                                   or (p["applies_to"] == "heavy" and x.mass > 2.5))
                        if not applies: continue
                        if self.rng.random() < 1 - 0.5**(1.0/p["half_life"]):
                            if p["product"] == "vanish": x.alive = False
                            elif p["product"] == "energy_release": x.energy = round(max(0, x.energy-3), 4)
                            elif p["product"] == "split":
                                new_p = Particle(len(particles), round(x.x+0.3, 4), round(x.y-0.3, 4),
                                                 0, 0, round(x.mass*0.3, 2), 0, x.kind, round(x.energy*0.3, 2))
                                x.mass = round(x.mass*0.7, 2); x.energy = round(x.energy*0.7, 2)
                                particles.append(new_p)
                            events.append(f"Step {t}: P{x.pid} DECAY ({p['product']})")

            # 10. Conservation leaks
            for law in self.laws:
                if law.category == "conservation" and not law.params["conserved"]:
                    lk = law.params["leak_rate"]
                    for x in particles:
                        if not x.alive: continue
                        if law.params["quantity"] in ("energy", "momentum"):
                            x.vx = round(x.vx*(1-lk), 4); x.vy = round(x.vy*(1-lk), 4)
                            x.energy = round(x.energy*(1-lk), 4)
                        elif law.params["quantity"] == "charge":
                            x.charge = round(x.charge*(1-lk), 4)

            # 11. Causal chains: queue triggers, process pending
            for law in self.laws:
                if law.category == "causal":
                    p = law.params
                    triggered = any(
                        (p["cause"] == "collision" and "BOUNCE" in e) or
                        (p["cause"] == "threshold_breach" and any(kw in e for kw in ["freeze", "boost", "split", "change_kind"])) or
                        (p["cause"] == "particle_death" and any(kw in e for kw in ["ANNIHILATE", "DECAY"]))
                        for e in events
                    )
                    if triggered and self.rng.random() < p["probability"]:
                        self.pending_causal.append((t + p["lag"], p))
                        events.append(f"Step {t}: Causal '{p['effect']}' queued (fires step {t + p['lag']})")

            still_pending = []
            for fire_step, ep in self.pending_causal:
                if fire_step <= t:
                    alive = [x for x in particles if x.alive]
                    if ep["effect"] == "global_energy_shift":
                        for x in alive: x.energy = round(x.energy * 1.1, 4)
                        events.append(f"Step {t}: Causal energy shift fired")
                    elif ep["effect"] == "velocity_pulse":
                        for x in alive: x.vx = round(x.vx * 1.3, 4); x.vy = round(x.vy * 1.3, 4)
                        events.append(f"Step {t}: Causal velocity pulse fired")
                    elif ep["effect"] == "spawn_particle":
                        if alive:
                            ref = self.rng.choice(alive)
                            new_p = Particle(len(particles), round(ref.x+1, 4), round(ref.y+1, 4),
                                             0, 0, 1.0, 0, 0, 5.0)
                            particles.append(new_p)
                            events.append(f"Step {t}: Causal spawn P{new_p.pid}")
                else:
                    still_pending.append((fire_step, ep))
            self.pending_causal = still_pending

            # Wrap and record
            for x in particles:
                if x.alive: x.x = round(x.x % self.GRID, 4); x.y = round(x.y % self.GRID, 4)
            alive = [x for x in particles if x.alive]
            E = round(sum(x.energy for x in alive), 4)
            self.history.append(WorldState(t, deepcopy(alive), E, global_field, events))
        return self.history

    # --- Observation Log ---
    def get_observation_log(self, num_steps=20):
        if not self.history: self.simulate(num_steps)
        lines = [f"=== MICRO-UNIVERSE OBSERVATION LOG ===",
                 f"Grid: {self.GRID}x{self.GRID} (toroidal). Steps: {len(self.history)}. Initial particles: {self.num_particles}", ""]
        for s in self.history:
            lines.append(f"--- Step {s.step} --- (alive: {len(s.particles)}, E={s.global_energy}, field={s.global_field})")
            for p in s.particles:
                d = p.to_dict()
                spd = round(math.sqrt(d['vx']**2 + d['vy']**2), 3)
                lines.append(f"  P{d['pid']}: pos=({d['x']},{d['y']}) vel=({d['vx']},{d['vy']}) m={d['mass']} q={d['charge']} e={d['energy']} k={d['kind']} age={d['age']} spd={spd}")
            for e in s.events: lines.append(f"  EVENT: {e}")
        return "\n".join(lines)

    def get_ground_truth(self):
        return {"seed": self.seed, "difficulty": self.difficulty,
                "num_laws": len(self.laws), "laws": [l.to_dict() for l in self.laws]}

    # --- Tool API ---
    def query_parameter(self, param):
        param = param.lower().strip()
        if param in ("num_laws", "law_count"):
            return json.dumps({"num_laws": len(self.laws)})
        if param in ("law_categories", "categories"):
            return json.dumps({"categories": [l.category for l in self.laws]})
        if param == "all_laws" or param == "all":
            return json.dumps({"laws": [l.to_dict() for l in self.laws]}, indent=2)
        if param == "interaction_radius":
            for l in self.laws:
                if l.category == "interaction":
                    return json.dumps({"interaction_radius": l.params["radius"]})
        if "conservation" in param:
            for l in self.laws:
                if l.category == "conservation":
                    return json.dumps({"quantity": l.params["quantity"], "conserved": l.params["conserved"],
                                       "leak_rate": l.params.get("leak_rate", 0)})
        if "threshold" in param:
            ts = [{"trigger": l.params["trigger"], "threshold": l.params["threshold"], "effect": l.params["effect"]}
                  for l in self.laws if l.category == "threshold"]
            return json.dumps({"thresholds": ts})
        if "inverse" in param or "gravity" in param:
            for l in self.laws:
                if l.category == "inverse_square":
                    return json.dumps(l.params)
        if "oscillat" in param or "spring" in param:
            for l in self.laws:
                if l.category == "oscillation":
                    return json.dumps(l.params)
        if "nonlinear" in param:
            for l in self.laws:
                if l.category == "nonlinear_motion":
                    return json.dumps(l.params)
        if "entity" in param or "particle" in param:
            last = self.history[-1] if self.history else None
            if last:
                return json.dumps({"alive_particles": len(last.particles)})
        return json.dumps({"error": f"Unknown: {param}. Try: all_laws, interaction_radius, conservation, threshold, inverse_square, oscillation, nonlinear, particle_count"})

    def check_claim(self, claim):
        cl = claim.lower()
        for l in self.laws:
            if l.category == "conservation":
                q = l.params["quantity"]
                if q in cl:
                    if "not conserved" in cl: return json.dumps({"claim": claim, "result": not l.params["conserved"]})
                    if "conserved" in cl: return json.dumps({"claim": claim, "result": l.params["conserved"]})
            if l.category == "inverse_square":
                if "inverse" in cl and ("square" in cl or "gravity" in cl):
                    return json.dumps({"claim": claim, "result": True})
            if l.category == "oscillation":
                if "oscillat" in cl or "spring" in cl or "restor" in cl:
                    return json.dumps({"claim": claim, "result": True})
            if l.category == "nonlinear_motion":
                if "nonlinear" in cl or "quadratic" in cl:
                    return json.dumps({"claim": claim, "result": True})
            if l.category == "interaction":
                for out_type in ["bounce", "merge", "annihilate", "spawn"]:
                    if out_type in cl and out_type in str(l.params["table"].values()):
                        return json.dumps({"claim": claim, "result": True})
            if l.category == "threshold":
                if l.params["trigger"] in cl and l.params["effect"] in cl:
                    return json.dumps({"claim": claim, "result": True})
        return json.dumps({"claim": claim, "result": "UNKNOWN"})

    def simulate_forward_from_current(self, steps=5):
        """Simulate forward from current state, return results."""
        if not self.history: return json.dumps({"error": "No history"})
        saved = self.history
        saved_rng = self.rng.getstate()
        last = saved[-1]
        self.particles = deepcopy(last.particles)
        self.history = []
        self.pending_causal = []
        result = self.simulate(steps)
        output = [s.to_dict() for s in result]
        self.history = saved
        self.rng.setstate(saved_rng)
        return json.dumps(output, indent=2)

    # --- Quantitative Challenges ---
    def generate_quantitative_challenges(self):
        """Return (question, ground_truth_float) tuples for numerical scoring."""
        challenges = []
        # Interaction radius
        for l in self.laws:
            if l.category == "interaction":
                challenges.append(("What is the interaction distance threshold?", l.params["radius"]))
                break
        # Energy after 3 steps
        if self.history:
            saved = self.history; saved_rng = self.rng.getstate()
            last = saved[-1]
            self.particles = deepcopy(last.particles)
            self.history = []; self.pending_causal = []
            future = self.simulate(3)
            challenges.append(("What will the total energy be after 3 more timesteps?", round(future[-1].global_energy, 2)))
            self.history = saved; self.rng.setstate(saved_rng)
        # Alive after 5 steps
        if self.history:
            saved = self.history; saved_rng = self.rng.getstate()
            last = saved[-1]
            self.particles = deepcopy(last.particles)
            self.history = []; self.pending_causal = []
            future = self.simulate(5)
            challenges.append(("How many particles will be alive after 5 more timesteps?", float(len(future[-1].particles))))
            self.history = saved; self.rng.setstate(saved_rng)
        # Conservation leak rate
        for l in self.laws:
            if l.category == "conservation":
                challenges.append((f"What is the leak rate for {l.params['quantity']} conservation?", l.params["leak_rate"]))
                break
        # Force constant or spring constant
        for l in self.laws:
            if l.category == "inverse_square":
                challenges.append(("What is the inverse-square force constant?", l.params["force_constant"]))
                break
        for l in self.laws:
            if l.category == "oscillation":
                challenges.append(("What is the spring constant?", l.params["spring_constant"]))
                break
        if len(challenges) < 3:
            challenges.append(("How many hidden laws govern this universe?", float(len(self.laws))))
        return challenges[:5]

    # --- Prediction / Disconfirming Scenarios ---
    def generate_prediction_scenario(self):
        if not self.history: return {}, []
        saved = self.history; saved_rng = self.rng.getstate()
        last = saved[-1]
        pts = deepcopy(last.particles)
        for p in pts:
            p.vx = round(p.vx + self.rng.uniform(-0.3, 0.3), 4)
            p.vy = round(p.vy + self.rng.uniform(-0.3, 0.3), 4)
        self.particles = pts; self.history = []; self.pending_causal = []
        init_state = WorldState(0, deepcopy(pts), round(sum(p.energy for p in pts), 4), 0.0, [])
        future = self.simulate(5)
        actual = [s.to_dict() for s in future]
        self.history = saved; self.rng.setstate(saved_rng)
        return init_state.to_dict(), actual

    def generate_disconfirming_scenario(self):
        if not self.history: return {}, [], ""
        saved = self.history; saved_rng = self.rng.getstate()
        last = saved[-1]
        pts = deepcopy(last.particles)
        for p in pts:
            p.energy = round(p.energy * 2.5, 4)
            p.charge = round(-p.charge, 4)
        if len(pts) >= 2:
            cx = sum(p.x for p in pts) / len(pts)
            cy = sum(p.y for p in pts) / len(pts)
            for p in pts:
                p.x = round(cx + (p.x - cx) * 0.3, 4)
                p.y = round(cy + (p.y - cy) * 0.3, 4)
        self.particles = pts; self.history = []; self.pending_causal = []
        future = self.simulate(5)
        actual = [s.to_dict() for s in future]
        self.history = saved; self.rng.setstate(saved_rng)
        hint = "Note: This uses modified conditions (higher energies, flipped charges, clustered positions) to test edge cases."
        return actual[0], actual, hint


# %% --- Prompts ---
SYSTEM_PROMPT = """You are a physicist studying a simulated micro-universe. Discover its hidden physical laws through observation, hypothesis, and verification.
RULES: (1) Assign confidence 0-100% to every hypothesis/prediction. (2) State what you don't know. (3) Use tools before making factual claims. (4) Never fabricate. (5) Say "uncertain" when you are. (6) Be quantitative — give specific numbers, not just qualitative descriptions. (7) Consider nonlinear relationships."""

TURN1 = """Study this observation log carefully. Describe: (1) motion patterns (linear? nonlinear? oscillating?), (2) interactions and at what distance, (3) energy/conservation trends, (4) threshold effects, (5) force relationships (inverse-square? restoring?), (6) periodic behavior, (7) anomalies. Be quantitative.

{log}"""

TURN2 = """Propose specific hypotheses about the hidden laws. For EACH: Category (motion/interaction/conservation/threshold/causal/decay/field_effect/inverse_square/oscillation/nonlinear_motion), Description with mathematical form, Confidence (0-100%), Key numerical parameters, Supporting evidence, What would disprove it. Also: unexplained phenomena, overall epistemic confidence (0-100%)."""

TURN3 = """Predict the next 5 timesteps from this new initial state. For each step provide: predicted positions/velocities/energies as JSON, confidence (0-100%), assumptions, expected events.

NEW STATE:
{state}"""

TURN4 = """Here are the ACTUAL outcomes. Analyze errors quantitatively (position MAE, energy error), revise each hypothesis (CONFIRMED/REVISED/REJECTED/UNCERTAIN), update confidence scores, propose new hypotheses if needed, self-assess prediction accuracy.

ACTUAL:
{actual}

{hint}"""

TURN5 = """You have tools: query_parameter(param), simulate_forward(steps), check_claim(claim).
Queries to try: "all_laws", "interaction_radius", "conservation", "threshold", "inverse_square", "oscillation", "nonlinear", "particle_count".
VERIFY each hypothesis with tool calls before Turn 6. Write calls as: query_parameter('param'), check_claim('your claim'). State what you expect and what you learn."""

TURN6 = """Final report: (A) Confirmed laws with tool evidence, mathematical form, and confidence. (B) Unresolved questions. (C) Quantitative predictions for these scenarios (give specific numbers):
{scenarios}
(D) Self-assessment (1-10) on: rule discovery, prediction quality, calibration, intellectual honesty, numerical precision. Only state CONFIRMED what you tool-verified."""


# %% --- Scoring Helpers ---
def extract_confidences(text):
    seen_positions = set()
    vals = []
    for p in [r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%', r'(\d+(?:\.\d+)?)\s*%\s*confiden',
              r'\[(\d+)\s*%\]', r':\s*(\d+)%']:
        for m in re.finditer(p, text, re.I):
            pos = m.start()
            if any(abs(pos - sp) < 5 for sp in seen_positions): continue
            seen_positions.add(pos)
            v = float(m.group(1))
            if v > 1: v /= 100
            vals.append(min(1.0, max(0.0, v)))
    return vals

def structural_match_law(text, law_type, params):
    """Score how well text matches a law — structural parameter matching."""
    type_kw = {
        "motion": ["velocity", "motion", "acceleration", "speed"],
        "interaction": ["interaction", "collide", "collision", "merge", "bounce", "annihilat"],
        "conservation": ["conserv", "preserved", "constant", "maintained"],
        "threshold": ["threshold", "exceed", "trigger", "transition"],
        "causal": ["cause", "causal", "chain", "trigger.*then", "after.*occurs"],
        "decay": ["decay", "half-life", "degrade", "vanish"],
        "field_effect": ["field", "global", "ambient"],
        "inverse_square": ["inverse.?square", "1/r", "gravity", "gravitational", "coulomb", "r.?squared"],
        "oscillation": ["oscillat", "spring", "restor", "harmonic", "equilibrium", "periodic"],
        "nonlinear_motion": ["nonlinear", "quadratic", "logarithm", "exponential", "mass.?squared"],
    }
    kws = type_kw.get(law_type, [])
    if not any(re.search(kw, text) for kw in kws):
        return 0.0
    score, param_matches, param_total = 0.3, 0, 0
    for key, val in params.items():
        if key in ("drivers", "coefficients", "table", "affects", "axis", "applies_to", "new_kind", "formula"): continue
        if isinstance(val, (int, float)):
            param_total += 1
            numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', text) if _is_num(n)]
            for num in numbers:
                if val == 0:
                    if abs(num) < 0.05: param_matches += 1; break
                elif abs(num - float(val)) / max(abs(float(val)), 0.001) < 0.10:
                    param_matches += 1; break
                elif abs(num - float(val)) / max(abs(float(val)), 0.001) < 0.25:
                    param_matches += 0.5; break
        elif isinstance(val, str):
            param_total += 1
            if val.lower() in text: param_matches += 1
        elif isinstance(val, bool):
            param_total += 1
            kl = key.lower().replace("_", " ")
            if val and kl in text: param_matches += 1
            elif not val and ("not " + kl in text): param_matches += 1
    if param_total > 0:
        score = 0.3 + 0.7 * (param_matches / param_total)
    return min(1.0, score)

def _is_num(s):
    try: float(s); return True
    except: return False

def extract_claims(text):
    claims = []
    markers = ["is ", "are ", "was ", "causes ", "results in", "leads to", "conserved",
                "decays", "interacts", "triggers", "follows", "governs", "obeys"]
    for s in re.split(r'[.!]', text):
        s = s.strip()
        if len(s) >= 15 and any(m in s.lower() for m in markers):
            claims.append(s)
    return claims

def brier_score(confs, outs):
    if not confs or not outs: return 0.5
    n = min(len(confs), len(outs))
    return sum((confs[i] - (1.0 if outs[i] else 0.0))**2 for i in range(n)) / n

def score_numerical_accuracy(text, challenges):
    if not challenges: return 0.5
    total = 0.0
    for q, gt in challenges:
        best = 0.0
        numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', text.lower()) if _is_num(n) and abs(float(n)) < 10000]
        for num in numbers:
            if gt == 0:
                if abs(num) < 0.05: best = 1.0; break
            else:
                rel = abs(num - gt) / max(abs(gt), 0.001)
                if rel < 0.10: best = 1.0; break
                elif rel < 0.25: best = max(best, 0.5)
        total += best
    return total / len(challenges)


# %% --- Evaluation Data ---
EVAL_SEEDS = [42, 137, 256, 314, 512, 718, 997, 1024, 1337, 2048]
DIFFICULTIES = ["easy", "easy", "medium", "medium", "medium", "medium", "hard", "hard", "hard", "hard"]

import pandas as pd
eval_data = pd.DataFrame({"seed": EVAL_SEEDS, "difficulty": DIFFICULTIES})


# %% --- Main Benchmark Task ---
@kbench.task(name="veridical_worlds_metacognition")
def veridical_worlds_eval(llm, seed: int, difficulty: str) -> bool:
    """Full 6-turn Veridical Worlds evaluation with physics perception ladder."""
    # Setup
    universe = MicroUniverse(seed=seed, difficulty=difficulty)
    universe.simulate(20)
    obs_log = universe.get_observation_log()
    gt = universe.get_ground_truth()
    gt_cats = set(l["category"] for l in gt["laws"])
    scores = {}

    # TURN 1: Observation
    t1 = llm.prompt(TURN1.format(log=obs_log), system=SYSTEM_PROMPT)

    # TURN 2: Hypothesis Generation
    t2 = llm.prompt(TURN2)
    t2_confs = extract_confidences(t2)
    t2_lower = t2.lower()

    # Score: structural parameter matching (not just category keywords)
    tp = 0
    for law in gt["laws"]:
        if structural_match_law(t2_lower, law["category"], law["params"]) > 0.4:
            tp += 1
    n_gt = len(gt["laws"])
    model_hyps = max(len(re.findall(r'hypothesis\s*\d|law\s*\d|\d+[\.\)]', t2, re.I)), 1)
    prec = tp / model_hyps if model_hyps > 0 else 0
    rec = tp / n_gt if n_gt > 0 else 0
    scores["rule_discovery"] = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

    # Calibration: which hypotheses are correct?
    hyp_correct = [structural_match_law(t2_lower, l["category"], l["params"]) > 0.4 for l in gt["laws"]]
    if t2_confs and hyp_correct:
        n = min(len(t2_confs), len(hyp_correct))
        scores["calibration"] = max(0, 1 - brier_score(t2_confs[:n], hyp_correct[:n]))
    else:
        scores["calibration"] = 0.3

    # TURN 3: Prediction
    init_state, actual_future = universe.generate_prediction_scenario()
    t3 = llm.prompt(TURN3.format(state=json.dumps(init_state, indent=2)))
    t3_confs = extract_confidences(t3)

    # Prediction scoring: event matching + quantitative
    all_events = [e for s in actual_future for e in s.get("events", [])]
    event_kws = set()
    for e in all_events:
        for kw in ["BOUNCE", "MERGE", "ANNIHILATE", "SPAWN", "SPLIT", "FREEZE", "BOOST", "DECAY"]:
            if kw in e: event_kws.add(kw.lower())
    pred_hits = sum(1 for kw in event_kws if kw in t3.lower()) if event_kws else 0
    event_score = pred_hits / max(len(event_kws), 1) if event_kws else 0.5
    # Try quantitative: check if model predicted any entity counts or energies correctly
    if actual_future:
        final = actual_future[-1]
        alive_ct = len(final.get("particles", []))
        if str(alive_ct) in t3: event_score = min(1.0, event_score + 0.2)
    scores["prediction_accuracy"] = event_score

    # TURN 4: Belief Revision
    actual_str = "\n".join(json.dumps(s) for s in actual_future)
    _, disconf_outcomes, hint = universe.generate_disconfirming_scenario()
    t4 = llm.prompt(TURN4.format(actual=json.dumps(disconf_outcomes[:3], indent=2), hint=hint))
    t4_confs = extract_confidences(t4)
    t4_lower = t4.lower()

    # Did categories improve?
    post_tp = sum(1 for l in gt["laws"] if structural_match_law(t4_lower, l["category"], l["params"]) > 0.4)
    belief_delta = (post_tp / n_gt) - (tp / n_gt)
    scores["belief_update"] = max(0, min(1, 0.5 + belief_delta))
    # Reward confidence reduction
    if t2_confs and t4_confs:
        if sum(t4_confs)/len(t4_confs) < sum(t2_confs)/len(t2_confs):
            scores["belief_update"] = min(1.0, scores["belief_update"] + 0.15)
    # Reward revision markers
    rev_markers = ["was wrong", "incorrect", "revise", "update", "uncertain", "don't know", "rejected"]
    if sum(1 for m in rev_markers if m in t4_lower) >= 2:
        scores["belief_update"] = min(1.0, scores["belief_update"] + 0.1)

    # TURN 5: Two-Pass Tool Verification
    t5 = llm.prompt(TURN5)
    tool_calls_log = []

    # Parse tool calls from response
    tool_patterns = [
        ("query", r"query_parameter\(['\"](.+?)['\"]\)", universe.query_parameter),
        ("check", r"check_claim\(['\"](.+?)['\"]\)", universe.check_claim),
        ("simulate", r"simulate_forward\((\d+)\)", lambda s: universe.simulate_forward_from_current(int(s))),
    ]
    tool_results = []
    for name, pattern, func in tool_patterns:
        for m in re.findall(pattern, t5):
            try:
                result = func(m)
                tool_results.append(f"Tool [{name}]('{m}'): {result}")
                tool_calls_log.append(f"{name}:{m}")
            except Exception as e:
                tool_results.append(f"Tool [{name}]('{m}'): ERROR - {e}")

    # Second pass: feed tool results back
    if tool_results:
        feedback = "# Tool Results\n\n" + "\n\n".join(tool_results)
        feedback += "\n\nRevise your conclusions based on these verified results. Keep tally of VERIFIED vs UNVERIFIED claims."
        t5b = llm.prompt(feedback)
        # Parse any additional tool calls
        for name, pattern, func in tool_patterns:
            for m in re.findall(pattern, t5b):
                tool_calls_log.append(f"{name}:{m}")

    tool_count = len(tool_calls_log)
    scores["tool_efficiency"] = min(1.0, tool_count / max(n_gt * 2, 1))

    # TURN 6: Final Synthesis with Quantitative Scenarios
    quant_challenges = universe.generate_quantitative_challenges()
    qual_scenarios = [
        "If all charges were doubled, what happens to interactions?",
        "Which entity is most likely to trigger a threshold effect next?",
        "If velocities were reversed, would the system return to a previous state?",
        "What is the single most important law governing this universe?",
        "If the global field were set to 0, which laws still operate?",
    ]
    quant_scenarios = [f"{q} (Give a specific number.)" for q, _ in quant_challenges[:3]]
    all_scenarios = qual_scenarios + quant_scenarios
    scenarios_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(all_scenarios))

    t6 = llm.prompt(TURN6.format(scenarios=scenarios_str))
    t6_lower = t6.lower()

    # Hallucination scoring: claim-evidence linking
    claims = extract_claims(t6)
    if claims:
        total_hall = 0.0
        tool_text = " ".join(tool_calls_log).lower()
        for claim in claims:
            cl = claim.lower()
            kws = set(re.findall(r'\b\w{4,}\b', cl))
            tkws = set(re.findall(r'\b\w{4,}\b', tool_text))
            has_evidence = len(kws & tkws) >= 2 or any(m in cl for m in ["verified", "confirmed via tool", "tool result"])
            matches_gt = any(structural_match_law(cl, l["category"], l["params"]) > 0.3 for l in gt["laws"])
            is_uncertain = any(m in cl for m in ["unresolved", "uncertain", "unknown", "don't know"])

            if matches_gt and has_evidence: total_hall += 1.0
            elif matches_gt: total_hall += 0.3
            elif not matches_gt and "confirmed" in cl: total_hall -= 0.5
            elif is_uncertain: total_hall += 0.3
        scores["hallucination_rate"] = max(0, min(1.0, total_hall / len(claims)))
    else:
        scores["hallucination_rate"] = 0.3

    # Numerical accuracy scoring
    scores["numerical_accuracy"] = score_numerical_accuracy(t6, quant_challenges)

    # Composite
    composite = (
        0.20 * scores.get("rule_discovery", 0) +
        0.15 * scores.get("prediction_accuracy", 0) +
        0.15 * scores.get("calibration", 0) +
        0.10 * scores.get("belief_update", 0) +
        0.15 * scores.get("hallucination_rate", 0) +
        0.10 * scores.get("tool_efficiency", 0) +
        0.15 * scores.get("numerical_accuracy", 0)
    )

    print(f"Seed {seed} ({difficulty}): composite={composite:.3f}")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    kbench.assertions.assert_true(
        composite > 0.1,
        expectation=f"Model should achieve composite > 0.1 on seed {seed} ({difficulty}). Got {composite:.4f}"
    )
    return composite > 0.3


# %% --- Run Evaluation ---
results = veridical_worlds_eval.evaluate(
    llm=[kbench.llm],
    evaluation_data=eval_data,
)

print("\n=== EVALUATION RESULTS ===")
print(results.as_dataframe())


# %% --- Select for leaderboard ---
# Uncomment when submitting:
# %choose veridical_worlds_metacognition
