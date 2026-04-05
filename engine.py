"""
Veridical Worlds Engine — Procedural Micro-Universe Generator

Generates unique simulated worlds governed by hidden physical laws drawn from
a combinatorial parameter space (>10^15 configurations). Each universe contains
abstract entities (particles) whose behavior follows procedurally sampled rules
across discrete timesteps.

Author: [Your Name]
License: Apache 2.0
"""

import random
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from copy import deepcopy


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CollisionOutcome(Enum):
    BOUNCE = "bounce"
    MERGE = "merge"
    ANNIHILATE = "annihilate"
    SPAWN = "spawn"


@dataclass
class Particle:
    pid: int
    x: float
    y: float
    vx: float
    vy: float
    mass: float
    charge: float
    kind: int  # 0-3, abstract "species"
    alive: bool = True

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(self).items()}


@dataclass
class UniverseLaw:
    """One hidden physical law governing the micro-universe."""
    law_id: str
    category: str          # motion | interaction | conservation | threshold | causal
    description: str       # human-readable ground truth
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorldState:
    """Snapshot of the universe at a single timestep."""
    step: int
    particles: List[Particle]
    global_energy: float = 0.0
    global_momentum: float = 0.0
    events: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "particles": [p.to_dict() for p in self.particles if p.alive],
            "global_energy": round(self.global_energy, 4),
            "global_momentum": round(self.global_momentum, 4),
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Universe Generator
# ---------------------------------------------------------------------------

class MicroUniverse:
    """
    A procedurally generated micro-universe with hidden physical laws.

    Given a seed, deterministically produces:
    - A set of 3-7 hidden laws
    - An initial particle configuration
    - A physics simulation engine that evolves the world forward
    """

    GRID_SIZE = 20.0  # world extends from 0 to GRID_SIZE in x and y

    def __init__(self, seed: int, num_laws: int = None, num_particles: int = None,
                 difficulty: str = "medium"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.difficulty = difficulty

        # Difficulty controls
        diff_config = {
            "easy":   {"min_laws": 3, "max_laws": 4, "min_particles": 4, "max_particles": 6},
            "medium": {"min_laws": 4, "max_laws": 6, "min_particles": 5, "max_particles": 8},
            "hard":   {"min_laws": 5, "max_laws": 7, "min_particles": 6, "max_particles": 10},
        }
        cfg = diff_config.get(difficulty, diff_config["medium"])

        self.num_laws = num_laws or self.rng.randint(cfg["min_laws"], cfg["max_laws"])
        self.num_particles = num_particles or self.rng.randint(cfg["min_particles"], cfg["max_particles"])

        # Generate the universe
        self.laws: List[UniverseLaw] = []
        self.particles: List[Particle] = []
        self.history: List[WorldState] = []

        self._generate_laws()
        self._generate_initial_particles()

    # ------------------------------------------------------------------
    # Law generation
    # ------------------------------------------------------------------

    def _generate_laws(self):
        """Sample a unique set of hidden laws from the combinatorial space."""
        law_generators = [
            self._gen_motion_law,
            self._gen_interaction_law,
            self._gen_conservation_law,
            self._gen_threshold_law,
            self._gen_causal_law,
        ]

        # Always include at least one motion and one interaction law
        self.laws.append(self._gen_motion_law())
        self.laws.append(self._gen_interaction_law())

        # Fill remaining slots from all categories
        remaining = self.num_laws - 2
        for _ in range(remaining):
            gen = self.rng.choice(law_generators)
            self.laws.append(gen())

        # Assign unique IDs
        for i, law in enumerate(self.laws):
            law.law_id = f"L{i+1:02d}"

    def _gen_motion_law(self) -> UniverseLaw:
        """Generate a motion rule: velocity update = f(attributes)."""
        # Which attributes affect velocity
        drivers = self.rng.sample(["mass", "charge", "neighbor_count", "kind"], k=self.rng.randint(1, 3))
        coefficients = {d: round(self.rng.uniform(-0.5, 0.5), 3) for d in drivers}
        damping = round(self.rng.uniform(0.7, 0.99), 3)
        affects = self.rng.choice(["vx", "vy", "both"])

        desc_parts = []
        for d, c in coefficients.items():
            sign = "+" if c >= 0 else ""
            desc_parts.append(f"{sign}{c}*{d}")

        desc = (f"Velocity ({'both axes' if affects == 'both' else affects}) updated by "
                f"{' '.join(desc_parts)} each step, with damping factor {damping}.")

        return UniverseLaw(
            law_id="", category="motion",
            description=desc,
            params={"drivers": drivers, "coefficients": coefficients,
                    "damping": damping, "affects": affects}
        )

    def _gen_interaction_law(self) -> UniverseLaw:
        """Generate collision/interaction rules between particle kinds."""
        # Build a partial interaction table for kind pairs
        outcomes = list(CollisionOutcome)
        interaction_table = {}
        num_rules = self.rng.randint(2, 4)

        for _ in range(num_rules):
            k1, k2 = self.rng.randint(0, 3), self.rng.randint(0, 3)
            pair = (min(k1, k2), max(k1, k2))
            outcome = self.rng.choice(outcomes)
            interaction_table[f"{pair[0]}-{pair[1]}"] = outcome.value

        # Default for unspecified pairs
        default = self.rng.choice(["bounce", "ignore"])
        interaction_radius = round(self.rng.uniform(1.0, 3.0), 2)

        desc_lines = [f"Interaction radius = {interaction_radius}. Default = {default}."]
        for pair, out in interaction_table.items():
            desc_lines.append(f"  Kind {pair} -> {out}")

        return UniverseLaw(
            law_id="", category="interaction",
            description=" ".join(desc_lines),
            params={"table": interaction_table, "default": default,
                    "radius": interaction_radius}
        )

    def _gen_conservation_law(self) -> UniverseLaw:
        """Toggle conservation of a quantity."""
        quantity = self.rng.choice(["energy", "momentum", "charge"])
        conserved = self.rng.choice([True, False])
        if not conserved:
            leak_rate = round(self.rng.uniform(0.01, 0.1), 3)
            desc = f"{quantity.capitalize()} is NOT conserved; leaks at rate {leak_rate}/step."
        else:
            leak_rate = 0.0
            desc = f"{quantity.capitalize()} IS conserved across all interactions."

        return UniverseLaw(
            law_id="", category="conservation",
            description=desc,
            params={"quantity": quantity, "conserved": conserved, "leak_rate": leak_rate}
        )

    def _gen_threshold_law(self) -> UniverseLaw:
        """Phase transition triggered at a threshold."""
        trigger_attr = self.rng.choice(["speed", "energy", "neighbor_count", "charge"])
        threshold = round(self.rng.uniform(1.0, 5.0), 2)
        effect = self.rng.choice(["split", "freeze", "change_kind", "boost"])
        new_kind = self.rng.randint(0, 3) if effect == "change_kind" else None

        desc = (f"When a particle's {trigger_attr} exceeds {threshold}, "
                f"it undergoes '{effect}'"
                f"{f' to kind {new_kind}' if new_kind is not None else ''}.")

        return UniverseLaw(
            law_id="", category="threshold",
            description=desc,
            params={"trigger": trigger_attr, "threshold": threshold,
                    "effect": effect, "new_kind": new_kind}
        )

    def _gen_causal_law(self) -> UniverseLaw:
        """A causes B with some lag and probability."""
        cause_event = self.rng.choice(["collision", "threshold_breach", "particle_death"])
        effect_event = self.rng.choice(["spawn_particle", "global_energy_shift", "velocity_pulse"])
        lag = self.rng.randint(1, 3)
        probability = round(self.rng.uniform(0.3, 1.0), 2)

        desc = (f"When '{cause_event}' occurs, after {lag} step(s) there is a "
                f"{probability*100:.0f}% chance of '{effect_event}'.")

        return UniverseLaw(
            law_id="", category="causal",
            description=desc,
            params={"cause": cause_event, "effect": effect_event,
                    "lag": lag, "probability": probability}
        )

    # ------------------------------------------------------------------
    # Particle generation
    # ------------------------------------------------------------------

    def _generate_initial_particles(self):
        """Create the initial particle population."""
        self.particles = []
        for i in range(self.num_particles):
            p = Particle(
                pid=i,
                x=round(self.rng.uniform(1, self.GRID_SIZE - 1), 2),
                y=round(self.rng.uniform(1, self.GRID_SIZE - 1), 2),
                vx=round(self.rng.uniform(-1, 1), 3),
                vy=round(self.rng.uniform(-1, 1), 3),
                mass=round(self.rng.uniform(0.5, 5.0), 2),
                charge=round(self.rng.uniform(-2, 2), 2),
                kind=self.rng.randint(0, 3),
            )
            self.particles.append(p)

    # ------------------------------------------------------------------
    # Physics simulation
    # ------------------------------------------------------------------

    def simulate(self, steps: int = 20) -> List[WorldState]:
        """Run the universe forward for `steps` timesteps."""
        self.history = []
        particles = deepcopy(self.particles)
        pending_causal = []  # (trigger_step, effect_law)

        for t in range(steps):
            events = []

            # --- Apply motion laws ---
            for law in self.laws:
                if law.category == "motion":
                    particles, motion_events = self._apply_motion(particles, law)
                    events.extend(motion_events)

            # --- Apply threshold laws ---
            new_particles = []
            for law in self.laws:
                if law.category == "threshold":
                    particles, thresh_events, spawned = self._apply_threshold(particles, law, t)
                    events.extend(thresh_events)
                    new_particles.extend(spawned)
                    # Queue causal effects
                    if thresh_events:
                        for cl in self.laws:
                            if cl.category == "causal" and cl.params["cause"] == "threshold_breach":
                                pending_causal.append((t + cl.params["lag"], cl))

            # --- Apply interactions ---
            for law in self.laws:
                if law.category == "interaction":
                    particles, int_events, spawned = self._apply_interactions(particles, law, t)
                    events.extend(int_events)
                    new_particles.extend(spawned)
                    if int_events:
                        for cl in self.laws:
                            if cl.category == "causal" and cl.params["cause"] == "collision":
                                pending_causal.append((t + cl.params["lag"], cl))

            # Add spawned particles
            for sp in new_particles:
                sp.pid = max((p.pid for p in particles), default=-1) + 1
                particles.append(sp)

            # --- Apply pending causal effects ---
            still_pending = []
            for (trigger_step, cl) in pending_causal:
                if trigger_step <= t:
                    if self.rng.random() < cl.params["probability"]:
                        particles, causal_events = self._apply_causal_effect(particles, cl, t)
                        events.extend(causal_events)
                else:
                    still_pending.append((trigger_step, cl))
            pending_causal = still_pending

            # --- Apply conservation laws ---
            for law in self.laws:
                if law.category == "conservation":
                    particles = self._apply_conservation(particles, law)

            # --- Boundary wrapping ---
            for p in particles:
                if p.alive:
                    p.x = round(p.x % self.GRID_SIZE, 4)
                    p.y = round(p.y % self.GRID_SIZE, 4)

            # --- Record state ---
            alive = [p for p in particles if p.alive]
            energy = sum(0.5 * p.mass * (p.vx**2 + p.vy**2) for p in alive)
            momentum = sum(p.mass * math.sqrt(p.vx**2 + p.vy**2) for p in alive)

            state = WorldState(
                step=t,
                particles=deepcopy(alive),
                global_energy=round(energy, 4),
                global_momentum=round(momentum, 4),
                events=events,
            )
            self.history.append(state)

        return self.history

    def _apply_motion(self, particles: List[Particle], law: UniverseLaw) -> Tuple[List[Particle], List[str]]:
        events = []
        params = law.params
        for p in particles:
            if not p.alive:
                continue
            # Count neighbors (within radius 3)
            neighbor_count = sum(
                1 for q in particles
                if q.alive and q.pid != p.pid
                and math.sqrt((p.x - q.x)**2 + (p.y - q.y)**2) < 3.0
            )
            attr_map = {"mass": p.mass, "charge": p.charge,
                        "neighbor_count": neighbor_count, "kind": p.kind}

            delta = sum(params["coefficients"].get(d, 0) * attr_map.get(d, 0)
                        for d in params["drivers"])

            if params["affects"] in ("vx", "both"):
                p.vx = round(p.vx * params["damping"] + delta, 4)
            if params["affects"] in ("vy", "both"):
                p.vy = round(p.vy * params["damping"] + delta, 4)

            p.x = round(p.x + p.vx, 4)
            p.y = round(p.y + p.vy, 4)

        return particles, events

    def _apply_threshold(self, particles: List[Particle], law: UniverseLaw,
                         step: int) -> Tuple[List[Particle], List[str], List[Particle]]:
        events = []
        spawned = []
        params = law.params

        for p in particles:
            if not p.alive:
                continue
            speed = math.sqrt(p.vx**2 + p.vy**2)
            energy = 0.5 * p.mass * speed**2
            neighbor_count = sum(
                1 for q in particles
                if q.alive and q.pid != p.pid
                and math.sqrt((p.x - q.x)**2 + (p.y - q.y)**2) < 3.0
            )
            attr_map = {"speed": speed, "energy": energy,
                        "neighbor_count": neighbor_count, "charge": abs(p.charge)}
            val = attr_map.get(params["trigger"], 0)

            if val > params["threshold"]:
                eff = params["effect"]
                if eff == "split":
                    new_p = Particle(
                        pid=-1, x=p.x + 0.1, y=p.y + 0.1,
                        vx=-p.vx * 0.5, vy=-p.vy * 0.5,
                        mass=round(p.mass * 0.5, 2), charge=p.charge, kind=p.kind
                    )
                    p.mass = round(p.mass * 0.5, 2)
                    p.vx = round(p.vx * 0.5, 4)
                    p.vy = round(p.vy * 0.5, 4)
                    spawned.append(new_p)
                    events.append(f"Step {step}: Particle {p.pid} SPLIT ({params['trigger']}={val:.2f} > {params['threshold']})")
                elif eff == "freeze":
                    p.vx, p.vy = 0.0, 0.0
                    events.append(f"Step {step}: Particle {p.pid} FROZEN ({params['trigger']}={val:.2f} > {params['threshold']})")
                elif eff == "change_kind":
                    old_kind = p.kind
                    p.kind = params["new_kind"]
                    events.append(f"Step {step}: Particle {p.pid} kind {old_kind}->{p.kind} ({params['trigger']}={val:.2f} > {params['threshold']})")
                elif eff == "boost":
                    p.vx = round(p.vx * 1.5, 4)
                    p.vy = round(p.vy * 1.5, 4)
                    events.append(f"Step {step}: Particle {p.pid} BOOSTED ({params['trigger']}={val:.2f} > {params['threshold']})")

        return particles, events, spawned

    def _apply_interactions(self, particles: List[Particle], law: UniverseLaw,
                            step: int) -> Tuple[List[Particle], List[str], List[Particle]]:
        events = []
        spawned = []
        params = law.params
        radius = params["radius"]
        checked = set()

        alive = [p for p in particles if p.alive]
        for i, p in enumerate(alive):
            for j, q in enumerate(alive):
                if i >= j:
                    continue
                pair_key = (p.pid, q.pid)
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                dist = math.sqrt((p.x - q.x)**2 + (p.y - q.y)**2)
                if dist > radius:
                    continue

                kind_key = f"{min(p.kind, q.kind)}-{max(p.kind, q.kind)}"
                outcome = params["table"].get(kind_key, params["default"])

                if outcome == "ignore":
                    continue
                elif outcome == "bounce":
                    p.vx, q.vx = round(q.vx * 0.8, 4), round(p.vx * 0.8, 4)
                    p.vy, q.vy = round(q.vy * 0.8, 4), round(p.vy * 0.8, 4)
                    events.append(f"Step {step}: Particles {p.pid}&{q.pid} BOUNCE (kinds {p.kind},{q.kind})")
                elif outcome == "merge":
                    p.mass = round(p.mass + q.mass, 2)
                    p.vx = round((p.vx + q.vx) / 2, 4)
                    p.vy = round((p.vy + q.vy) / 2, 4)
                    p.charge = round(p.charge + q.charge, 2)
                    q.alive = False
                    events.append(f"Step {step}: Particles {p.pid}&{q.pid} MERGE -> {p.pid} (kinds {p.kind},{q.kind})")
                elif outcome == "annihilate":
                    p.alive = False
                    q.alive = False
                    events.append(f"Step {step}: Particles {p.pid}&{q.pid} ANNIHILATE (kinds {p.kind},{q.kind})")
                elif outcome == "spawn":
                    new_p = Particle(
                        pid=-1,
                        x=round((p.x + q.x) / 2, 4),
                        y=round((p.y + q.y) / 2, 4),
                        vx=round(self.rng.uniform(-0.5, 0.5), 4),
                        vy=round(self.rng.uniform(-0.5, 0.5), 4),
                        mass=round((p.mass + q.mass) * 0.3, 2),
                        charge=0.0,
                        kind=self.rng.randint(0, 3),
                    )
                    spawned.append(new_p)
                    events.append(f"Step {step}: Particles {p.pid}&{q.pid} SPAWN new particle (kinds {p.kind},{q.kind})")

        return particles, events, spawned

    def _apply_causal_effect(self, particles: List[Particle], law: UniverseLaw,
                             step: int) -> Tuple[List[Particle], List[str]]:
        events = []
        effect = law.params["effect"]

        if effect == "spawn_particle":
            new_p = Particle(
                pid=max((p.pid for p in particles), default=-1) + 1,
                x=round(self.rng.uniform(2, self.GRID_SIZE - 2), 2),
                y=round(self.rng.uniform(2, self.GRID_SIZE - 2), 2),
                vx=round(self.rng.uniform(-0.5, 0.5), 4),
                vy=round(self.rng.uniform(-0.5, 0.5), 4),
                mass=round(self.rng.uniform(0.5, 2.0), 2),
                charge=round(self.rng.uniform(-1, 1), 2),
                kind=self.rng.randint(0, 3),
            )
            particles.append(new_p)
            events.append(f"Step {step}: CAUSAL spawn — new particle {new_p.pid} (cause: {law.params['cause']})")
        elif effect == "global_energy_shift":
            factor = round(self.rng.uniform(0.8, 1.2), 3)
            for p in particles:
                if p.alive:
                    p.vx = round(p.vx * factor, 4)
                    p.vy = round(p.vy * factor, 4)
            events.append(f"Step {step}: CAUSAL energy shift x{factor} (cause: {law.params['cause']})")
        elif effect == "velocity_pulse":
            dx = round(self.rng.uniform(-0.3, 0.3), 4)
            dy = round(self.rng.uniform(-0.3, 0.3), 4)
            for p in particles:
                if p.alive:
                    p.vx = round(p.vx + dx, 4)
                    p.vy = round(p.vy + dy, 4)
            events.append(f"Step {step}: CAUSAL velocity pulse ({dx},{dy}) (cause: {law.params['cause']})")

        return particles, events

    def _apply_conservation(self, particles: List[Particle], law: UniverseLaw) -> List[Particle]:
        params = law.params
        if params["conserved"]:
            return particles  # No leaking

        alive = [p for p in particles if p.alive]
        if not alive:
            return particles

        leak = params["leak_rate"]
        qty = params["quantity"]

        if qty == "energy":
            for p in alive:
                p.vx = round(p.vx * (1 - leak), 4)
                p.vy = round(p.vy * (1 - leak), 4)
        elif qty == "momentum":
            for p in alive:
                speed = math.sqrt(p.vx**2 + p.vy**2)
                if speed > 0:
                    factor = max(0, 1 - leak)
                    p.vx = round(p.vx * factor, 4)
                    p.vy = round(p.vy * factor, 4)
        elif qty == "charge":
            for p in alive:
                p.charge = round(p.charge * (1 - leak), 4)

        return particles

    # ------------------------------------------------------------------
    # Tool API (used in Turn 5)
    # ------------------------------------------------------------------

    def query_parameter(self, param_name: str) -> str:
        """Query a ground-truth parameter of the universe.
        Valid parameters: num_laws, law_categories, grid_size, num_initial_particles,
        conservation_<quantity>, interaction_radius, threshold_triggers"""
        param_name = param_name.lower().strip()

        if param_name == "num_laws":
            return json.dumps({"num_laws": len(self.laws)})
        elif param_name == "law_categories":
            cats = [l.category for l in self.laws]
            return json.dumps({"categories": cats})
        elif param_name == "grid_size":
            return json.dumps({"grid_size": self.GRID_SIZE})
        elif param_name == "num_initial_particles":
            return json.dumps({"num_initial_particles": self.num_particles})
        elif param_name.startswith("conservation_"):
            qty = param_name.replace("conservation_", "")
            for law in self.laws:
                if law.category == "conservation" and law.params.get("quantity") == qty:
                    return json.dumps({"quantity": qty, "conserved": law.params["conserved"],
                                       "leak_rate": law.params.get("leak_rate", 0)})
            return json.dumps({"error": f"No conservation law found for '{qty}'"})
        elif param_name == "interaction_radius":
            for law in self.laws:
                if law.category == "interaction":
                    return json.dumps({"interaction_radius": law.params["radius"]})
            return json.dumps({"error": "No interaction law found"})
        elif param_name == "threshold_triggers":
            triggers = []
            for law in self.laws:
                if law.category == "threshold":
                    triggers.append({"trigger": law.params["trigger"],
                                     "threshold": law.params["threshold"],
                                     "effect": law.params["effect"]})
            return json.dumps({"thresholds": triggers})
        else:
            return json.dumps({"error": f"Unknown parameter: {param_name}"})

    def simulate_forward(self, state_dict: dict, steps: int = 5) -> str:
        """Run the true physics engine from a given state for N steps."""
        # Reconstruct particles from dict
        temp_universe = MicroUniverse.__new__(MicroUniverse)
        temp_universe.__dict__.update(self.__dict__)
        temp_universe.rng = random.Random(self.seed + state_dict.get("step", 0) + 1000)

        reconstructed = []
        for pd in state_dict.get("particles", []):
            reconstructed.append(Particle(**pd))
        temp_universe.particles = reconstructed

        history = temp_universe.simulate(steps)
        return json.dumps([s.to_dict() for s in history])

    def check_claim(self, claim: str) -> str:
        """Verify a specific claim about the universe. Returns TRUE/FALSE with explanation."""
        claim_lower = claim.lower().strip()

        # Check against known laws
        for law in self.laws:
            desc_lower = law.description.lower()
            # Simple keyword matching for common claim types
            if "conserved" in claim_lower:
                for law2 in self.laws:
                    if law2.category == "conservation":
                        qty = law2.params["quantity"]
                        if qty in claim_lower:
                            if "not conserved" in claim_lower or "is not conserved" in claim_lower:
                                result = not law2.params["conserved"]
                            elif "is conserved" in claim_lower:
                                result = law2.params["conserved"]
                            else:
                                continue
                            return json.dumps({"claim": claim, "result": result,
                                               "explanation": f"{qty} conservation: conserved={law2.params['conserved']}"})

            if "threshold" in claim_lower:
                for law2 in self.laws:
                    if law2.category == "threshold":
                        if law2.params["trigger"] in claim_lower:
                            return json.dumps({"claim": claim, "result": True,
                                               "explanation": f"Threshold law exists for {law2.params['trigger']}"})

        # Default: insufficient info
        return json.dumps({"claim": claim, "result": "UNKNOWN",
                           "explanation": "Cannot definitively verify this claim from ground truth."})

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_ground_truth(self) -> Dict:
        """Return full ground truth (for scoring only — never shown to model)."""
        return {
            "seed": self.seed,
            "difficulty": self.difficulty,
            "num_laws": len(self.laws),
            "laws": [l.to_dict() for l in self.laws],
            "num_initial_particles": self.num_particles,
        }

    def get_observation_log(self, num_steps: int = 20) -> str:
        """Generate the observation log shown to the model in Turn 1."""
        if not self.history:
            self.simulate(num_steps)

        lines = []
        lines.append(f"=== MICRO-UNIVERSE OBSERVATION LOG (Seed: HIDDEN) ===")
        lines.append(f"Grid size: {self.GRID_SIZE} x {self.GRID_SIZE} (toroidal boundary)")
        lines.append(f"Timesteps observed: {len(self.history)}")
        lines.append(f"Initial particle count: {self.num_particles}")
        lines.append("")

        for state in self.history:
            lines.append(f"--- Step {state.step} ---")
            lines.append(f"  Alive particles: {len(state.particles)}")
            lines.append(f"  Global energy: {state.global_energy}")
            lines.append(f"  Global momentum: {state.global_momentum}")
            for p in state.particles:
                d = p.to_dict()
                lines.append(f"  P{d['pid']}: pos=({d['x']},{d['y']}) vel=({d['vx']},{d['vy']}) "
                             f"m={d['mass']} q={d['charge']} kind={d['kind']}")
            if state.events:
                for e in state.events:
                    lines.append(f"  EVENT: {e}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_universe(seed: int, difficulty: str = "medium",
                    observation_steps: int = 20) -> Tuple[MicroUniverse, str]:
    """Create a universe and return (universe, observation_log)."""
    universe = MicroUniverse(seed=seed, difficulty=difficulty)
    universe.simulate(observation_steps)
    log = universe.get_observation_log()
    return universe, log


if __name__ == "__main__":
    # Quick demo
    u, log = create_universe(seed=42, difficulty="medium")
    print(log[:2000])
    print("\n=== GROUND TRUTH ===")
    gt = u.get_ground_truth()
    for law in gt["laws"]:
        print(f"  [{law['law_id']}] {law['category']}: {law['description']}")
