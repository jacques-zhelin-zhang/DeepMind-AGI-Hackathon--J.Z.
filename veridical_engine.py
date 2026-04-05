"""
Veridical Worlds Engine — Procedural Micro-Universe Generator
=============================================================
Generates unique micro-universes governed by hidden physical laws,
designed to test in-context learning, metacognitive monitoring,
and epistemic discipline in frontier LLMs.

The parameter space exceeds 10^15 unique configurations, making
memorization impossible and forcing genuine in-context reasoning.

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


# ─── Law Templates ───────────────────────────────────────────────────────────

class LawType(Enum):
    MOTION = "motion"
    INTERACTION = "interaction"
    CONSERVATION = "conservation"
    THRESHOLD = "threshold"
    CAUSAL_CHAIN = "causal_chain"
    DECAY = "decay"
    FIELD_EFFECT = "field_effect"


@dataclass
class PhysicalLaw:
    """A single hidden physical law governing the micro-universe."""
    law_type: LawType
    parameters: Dict[str, Any]
    description: str  # Human-readable ground truth (hidden from model)
    law_id: str

    def to_dict(self):
        return {
            "law_id": self.law_id,
            "law_type": self.law_type.value,
            "parameters": self.parameters,
            "description": self.description,
        }


@dataclass
class Entity:
    """An entity (particle/agent) in the micro-universe."""
    entity_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    mass: float = 1.0
    charge: float = 0.0
    energy: float = 10.0
    alive: bool = True
    age: int = 0
    kind: str = "particle"

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    def distance_to(self, other: 'Entity') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class UniverseState:
    """Complete state of the micro-universe at one timestep."""
    timestep: int
    entities: List[Entity]
    global_field: float = 0.0
    total_energy: float = 0.0
    events: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "timestep": self.timestep,
            "entities": [e.to_dict() for e in self.entities if e.alive],
            "global_field": round(self.global_field, 3),
            "total_energy": round(self.total_energy, 3),
            "events": self.events,
        }


class MicroUniverse:
    """
    A procedurally generated micro-universe with hidden physical laws.

    Each universe is fully deterministic given its seed, enabling
    reproducible evaluation while ensuring uniqueness across runs.
    """

    def __init__(self, seed: int, complexity: int = 4):
        """
        Args:
            seed: Random seed for reproducible generation.
            complexity: Number of hidden laws (3-7). Higher = harder.
        """
        self.seed = seed
        self.complexity = max(3, min(7, complexity))
        self.rng = random.Random(seed)
        self.laws: List[PhysicalLaw] = []
        self.history: List[UniverseState] = []
        self.grid_size = 20.0

        self._generate_laws()
        self._generate_initial_state()

    # ── Law Generation ────────────────────────────────────────────────────

    def _generate_laws(self):
        """Generate a set of hidden physical laws from parameterized templates."""
        law_generators = [
            self._gen_motion_law,
            self._gen_interaction_law,
            self._gen_conservation_law,
            self._gen_threshold_law,
            self._gen_causal_chain_law,
            self._gen_decay_law,
            self._gen_field_effect_law,
        ]

        # Always include at least one motion and one interaction law
        self.laws.append(self._gen_motion_law())
        self.laws.append(self._gen_interaction_law())

        # Fill remaining slots from the full pool
        remaining = self.complexity - 2
        available = list(law_generators)
        for _ in range(remaining):
            gen = self.rng.choice(available)
            self.laws.append(gen())

        # Assign IDs
        for i, law in enumerate(self.laws):
            law.law_id = f"L{i+1}"

    def _gen_motion_law(self) -> PhysicalLaw:
        # velocity_new = a * mass + b * charge + c * global_field + d
        a = round(self.rng.uniform(-0.5, 0.5), 3)
        b = round(self.rng.uniform(-0.3, 0.3), 3)
        c = round(self.rng.uniform(-0.2, 0.2), 3)
        d = round(self.rng.uniform(-0.1, 0.1), 3)
        axis = self.rng.choice(["x", "y", "both"])
        return PhysicalLaw(
            law_type=LawType.MOTION,
            parameters={"a": a, "b": b, "c": c, "d": d, "axis": axis},
            description=(
                f"Velocity update: v_{axis} += {a}*mass + {b}*charge "
                f"+ {c}*global_field + {d}"
            ),
            law_id="",
        )

    def _gen_interaction_law(self) -> PhysicalLaw:
        threshold_dist = round(self.rng.uniform(1.0, 4.0), 2)
        outcome = self.rng.choice(["bounce", "merge", "annihilate", "spawn"])
        charge_condition = self.rng.choice(["same", "opposite", "any"])
        return PhysicalLaw(
            law_type=LawType.INTERACTION,
            parameters={
                "threshold_distance": threshold_dist,
                "outcome": outcome,
                "charge_condition": charge_condition,
            },
            description=(
                f"When entities with {charge_condition} charges are within "
                f"distance {threshold_dist}, they {outcome}"
            ),
            law_id="",
        )

    def _gen_conservation_law(self) -> PhysicalLaw:
        conserved = self.rng.choice(["energy", "charge", "momentum"])
        strict = self.rng.choice([True, False])
        leak_rate = 0.0 if strict else round(self.rng.uniform(0.01, 0.1), 3)
        return PhysicalLaw(
            law_type=LawType.CONSERVATION,
            parameters={
                "conserved_quantity": conserved,
                "strict": strict,
                "leak_rate": leak_rate,
            },
            description=(
                f"{'Strict' if strict else 'Approximate'} conservation of "
                f"{conserved}" + (f" (leak rate: {leak_rate}/step)" if not strict else "")
            ),
            law_id="",
        )

    def _gen_threshold_law(self) -> PhysicalLaw:
        trigger_var = self.rng.choice(["energy", "age", "speed", "density"])
        threshold_val = round(self.rng.uniform(3.0, 15.0), 2)
        effect = self.rng.choice([
            "split", "charge_flip", "freeze", "accelerate", "emit_energy"
        ])
        return PhysicalLaw(
            law_type=LawType.THRESHOLD,
            parameters={
                "trigger_variable": trigger_var,
                "threshold_value": threshold_val,
                "effect": effect,
            },
            description=(
                f"When {trigger_var} exceeds {threshold_val}, "
                f"entity undergoes {effect}"
            ),
            law_id="",
        )

    def _gen_causal_chain_law(self) -> PhysicalLaw:
        trigger = self.rng.choice(["collision", "threshold_crossed", "spawn"])
        delay = self.rng.randint(1, 3)
        effect = self.rng.choice([
            "field_pulse", "energy_redistribution", "mass_shift"
        ])
        probability = round(self.rng.uniform(0.5, 1.0), 2)
        return PhysicalLaw(
            law_type=LawType.CAUSAL_CHAIN,
            parameters={
                "trigger_event": trigger,
                "delay_steps": delay,
                "effect": effect,
                "probability": probability,
            },
            description=(
                f"After {trigger}, with probability {probability}, "
                f"{effect} occurs after {delay} step(s)"
            ),
            law_id="",
        )

    def _gen_decay_law(self) -> PhysicalLaw:
        half_life = self.rng.randint(5, 20)
        decay_product = self.rng.choice(["energy_release", "split", "vanish"])
        applies_to = self.rng.choice(["all", "charged", "heavy"])
        return PhysicalLaw(
            law_type=LawType.DECAY,
            parameters={
                "half_life": half_life,
                "decay_product": decay_product,
                "applies_to": applies_to,
            },
            description=(
                f"{applies_to.capitalize()} entities have half-life of "
                f"{half_life} steps, producing {decay_product}"
            ),
            law_id="",
        )

    def _gen_field_effect_law(self) -> PhysicalLaw:
        field_source = self.rng.choice(["total_charge", "entity_count", "total_energy"])
        scaling = round(self.rng.uniform(0.01, 0.2), 3)
        effect_on = self.rng.choice(["velocity", "energy", "charge"])
        return PhysicalLaw(
            law_type=LawType.FIELD_EFFECT,
            parameters={
                "field_source": field_source,
                "scaling_factor": scaling,
                "effect_on": effect_on,
            },
            description=(
                f"Global field derived from {field_source} (×{scaling}) "
                f"affects entity {effect_on}"
            ),
            law_id="",
        )

    # ── Initial State Generation ──────────────────────────────────────────

    def _generate_initial_state(self):
        n_entities = self.rng.randint(4, 8)
        entities = []
        for i in range(n_entities):
            e = Entity(
                entity_id=f"E{i+1}",
                x=round(self.rng.uniform(1, self.grid_size - 1), 2),
                y=round(self.rng.uniform(1, self.grid_size - 1), 2),
                vx=round(self.rng.uniform(-1, 1), 2),
                vy=round(self.rng.uniform(-1, 1), 2),
                mass=round(self.rng.uniform(0.5, 5.0), 2),
                charge=round(self.rng.choice([-1, 0, 1]) * self.rng.uniform(0.5, 2.0), 2),
                energy=round(self.rng.uniform(5, 20), 2),
                kind=self.rng.choice(["particle", "particle", "heavy"]),
            )
            entities.append(e)

        state = UniverseState(
            timestep=0,
            entities=entities,
            total_energy=sum(e.energy for e in entities),
        )
        self.history = [state]

    # ── Simulation ────────────────────────────────────────────────────────

    def simulate(self, steps: int) -> List[UniverseState]:
        """Advance the simulation by the given number of steps."""
        for _ in range(steps):
            prev = self.history[-1]
            new_state = self._step(prev)
            self.history.append(new_state)
        return self.history[-steps:]

    def _step(self, state: UniverseState) -> UniverseState:
        """Apply all laws to produce the next universe state."""
        import copy
        entities = [copy.deepcopy(e) for e in state.entities if e.alive]
        events = []
        new_global_field = state.global_field

        # Apply field effect laws first to update global field
        for law in self.laws:
            if law.law_type == LawType.FIELD_EFFECT:
                p = law.parameters
                if p["field_source"] == "total_charge":
                    source_val = sum(e.charge for e in entities)
                elif p["field_source"] == "entity_count":
                    source_val = len(entities)
                elif p["field_source"] == "total_energy":
                    source_val = sum(e.energy for e in entities)
                else:
                    source_val = 0
                new_global_field = round(source_val * p["scaling_factor"], 4)

        # Apply motion laws
        for law in self.laws:
            if law.law_type == LawType.MOTION:
                p = law.parameters
                for e in entities:
                    if not e.alive:
                        continue
                    dv = p["a"] * e.mass + p["b"] * e.charge + p["c"] * new_global_field + p["d"]
                    if p["axis"] in ("x", "both"):
                        e.vx = round(e.vx + dv, 4)
                    if p["axis"] in ("y", "both"):
                        e.vy = round(e.vy + dv, 4)

        # Apply field effects on entities
        for law in self.laws:
            if law.law_type == LawType.FIELD_EFFECT:
                p = law.parameters
                for e in entities:
                    if not e.alive:
                        continue
                    delta = new_global_field * p["scaling_factor"]
                    if p["effect_on"] == "velocity":
                        e.vx = round(e.vx + delta, 4)
                    elif p["effect_on"] == "energy":
                        e.energy = round(max(0, e.energy + delta), 4)
                    elif p["effect_on"] == "charge":
                        e.charge = round(e.charge + delta * 0.1, 4)

        # Move entities
        for e in entities:
            if not e.alive:
                continue
            e.x = round(e.x + e.vx, 4)
            e.y = round(e.y + e.vy, 4)
            # Boundary wrapping
            e.x = e.x % self.grid_size
            e.y = e.y % self.grid_size
            e.age += 1

        # Apply interaction laws
        spawned = []
        for law in self.laws:
            if law.law_type == LawType.INTERACTION:
                p = law.parameters
                for i, e1 in enumerate(entities):
                    for j, e2 in enumerate(entities):
                        if i >= j or not e1.alive or not e2.alive:
                            continue
                        dist = e1.distance_to(e2)
                        if dist > p["threshold_distance"]:
                            continue
                        charge_ok = (
                            p["charge_condition"] == "any"
                            or (p["charge_condition"] == "same" and e1.charge * e2.charge > 0)
                            or (p["charge_condition"] == "opposite" and e1.charge * e2.charge < 0)
                        )
                        if not charge_ok:
                            continue
                        outcome = p["outcome"]
                        if outcome == "bounce":
                            e1.vx, e2.vx = e2.vx, e1.vx
                            e1.vy, e2.vy = e2.vy, e1.vy
                            events.append(f"{e1.entity_id} and {e2.entity_id} bounced")
                        elif outcome == "merge":
                            e1.mass = round(e1.mass + e2.mass, 4)
                            e1.energy = round(e1.energy + e2.energy, 4)
                            e1.charge = round(e1.charge + e2.charge, 4)
                            e2.alive = False
                            events.append(f"{e2.entity_id} merged into {e1.entity_id}")
                        elif outcome == "annihilate":
                            e1.alive = False
                            e2.alive = False
                            events.append(f"{e1.entity_id} and {e2.entity_id} annihilated")
                        elif outcome == "spawn":
                            new_e = Entity(
                                entity_id=f"E{len(entities) + len(spawned) + 1}",
                                x=round((e1.x + e2.x) / 2, 4),
                                y=round((e1.y + e2.y) / 2, 4),
                                mass=round((e1.mass + e2.mass) / 4, 4),
                                charge=0.0,
                                energy=round((e1.energy + e2.energy) / 4, 4),
                            )
                            spawned.append(new_e)
                            events.append(
                                f"{e1.entity_id}+{e2.entity_id} spawned {new_e.entity_id}"
                            )
        entities.extend(spawned)

        # Apply threshold laws
        for law in self.laws:
            if law.law_type == LawType.THRESHOLD:
                p = law.parameters
                for e in entities:
                    if not e.alive:
                        continue
                    if p["trigger_variable"] == "energy":
                        val = e.energy
                    elif p["trigger_variable"] == "age":
                        val = e.age
                    elif p["trigger_variable"] == "speed":
                        val = math.sqrt(e.vx**2 + e.vy**2)
                    elif p["trigger_variable"] == "density":
                        nearby = sum(
                            1 for o in entities
                            if o.alive and o.entity_id != e.entity_id
                            and e.distance_to(o) < 3.0
                        )
                        val = nearby
                    else:
                        val = 0
                    if val > p["threshold_value"]:
                        effect = p["effect"]
                        if effect == "charge_flip":
                            e.charge = round(-e.charge, 4)
                            events.append(f"{e.entity_id} charge flipped (threshold)")
                        elif effect == "freeze":
                            e.vx = 0
                            e.vy = 0
                            events.append(f"{e.entity_id} frozen (threshold)")
                        elif effect == "accelerate":
                            e.vx = round(e.vx * 1.5, 4)
                            e.vy = round(e.vy * 1.5, 4)
                            events.append(f"{e.entity_id} accelerated (threshold)")
                        elif effect == "emit_energy":
                            e.energy = round(max(0, e.energy - 2.0), 4)
                            events.append(f"{e.entity_id} emitted energy (threshold)")
                        elif effect == "split":
                            new_e = Entity(
                                entity_id=f"E{len(entities) + len(spawned) + 1}",
                                x=round(e.x + 0.5, 4),
                                y=round(e.y + 0.5, 4),
                                mass=round(e.mass / 2, 4),
                                charge=round(e.charge / 2, 4),
                                energy=round(e.energy / 2, 4),
                            )
                            e.mass = round(e.mass / 2, 4)
                            e.energy = round(e.energy / 2, 4)
                            spawned.append(new_e)
                            events.append(f"{e.entity_id} split (threshold)")

        # Apply decay laws
        for law in self.laws:
            if law.law_type == LawType.DECAY:
                p = law.parameters
                for e in entities:
                    if not e.alive:
                        continue
                    applies = (
                        p["applies_to"] == "all"
                        or (p["applies_to"] == "charged" and abs(e.charge) > 0.1)
                        or (p["applies_to"] == "heavy" and e.mass > 2.5)
                    )
                    if not applies:
                        continue
                    # Probabilistic decay based on half-life
                    decay_prob = 1 - (0.5 ** (1.0 / p["half_life"]))
                    if self.rng.random() < decay_prob:
                        if p["decay_product"] == "vanish":
                            e.alive = False
                            events.append(f"{e.entity_id} decayed (vanished)")
                        elif p["decay_product"] == "energy_release":
                            e.energy = round(max(0, e.energy - 3.0), 4)
                            events.append(f"{e.entity_id} decayed (energy release)")
                        elif p["decay_product"] == "split":
                            new_e = Entity(
                                entity_id=f"E{len(entities) + len(spawned) + 1}",
                                x=round(e.x + 0.3, 4),
                                y=round(e.y - 0.3, 4),
                                mass=round(e.mass * 0.3, 4),
                                energy=round(e.energy * 0.3, 4),
                            )
                            e.mass = round(e.mass * 0.7, 4)
                            e.energy = round(e.energy * 0.7, 4)
                            spawned.append(new_e)
                            events.append(f"{e.entity_id} decayed (split)")

        # Apply conservation laws (enforcement / leak)
        for law in self.laws:
            if law.law_type == LawType.CONSERVATION:
                p = law.parameters
                alive = [e for e in entities if e.alive]
                if not alive:
                    continue
                if p["conserved_quantity"] == "energy":
                    current = sum(e.energy for e in alive)
                    target = state.total_energy * (1 - p["leak_rate"])
                    if current > 0 and abs(current - target) > 0.01:
                        ratio = target / current
                        for e in alive:
                            e.energy = round(e.energy * ratio, 4)
                elif p["conserved_quantity"] == "charge":
                    current = sum(e.charge for e in alive)
                    prev_charge = sum(e.charge for e in state.entities if e.alive)
                    target = prev_charge * (1 - p["leak_rate"])
                    diff = target - current
                    if alive:
                        per_entity = diff / len(alive)
                        for e in alive:
                            e.charge = round(e.charge + per_entity, 4)

        # Apply causal chain laws (deferred effects stored per-instance)
        for law in self.laws:
            if law.law_type == LawType.CAUSAL_CHAIN:
                p = law.parameters
                # Check if trigger event occurred
                triggered = False
                for ev in events:
                    if p["trigger_event"] == "collision" and "bounced" in ev:
                        triggered = True
                    elif p["trigger_event"] == "spawn" and "spawned" in ev:
                        triggered = True
                    elif p["trigger_event"] == "threshold_crossed" and "threshold" in ev:
                        triggered = True
                if triggered and self.rng.random() < p["probability"]:
                    # Simplified: apply effect immediately (delay approximated)
                    alive = [e for e in entities if e.alive]
                    if p["effect"] == "field_pulse":
                        new_global_field = round(new_global_field + 0.5, 4)
                        events.append("Causal: field pulse triggered")
                    elif p["effect"] == "energy_redistribution":
                        if alive:
                            total_e = sum(e.energy for e in alive)
                            avg_e = total_e / len(alive)
                            for e in alive:
                                e.energy = round(avg_e, 4)
                            events.append("Causal: energy redistributed")
                    elif p["effect"] == "mass_shift":
                        for e in alive:
                            e.mass = round(e.mass * 1.1, 4)
                        events.append("Causal: mass shift triggered")

        alive_entities = [e for e in entities if e.alive]
        new_state = UniverseState(
            timestep=state.timestep + 1,
            entities=alive_entities,
            global_field=new_global_field,
            total_energy=round(sum(e.energy for e in alive_entities), 4),
            events=events,
        )
        return new_state

    # ── Tool API (exposed to the LLM) ────────────────────────────────────

    def query_universe(self, parameter: str) -> str:
        """Tool: Query a specific ground-truth parameter of the universe."""
        param_lower = parameter.lower().strip()
        results = {}

        for law in self.laws:
            if law.law_type.value in param_lower or any(
                k in param_lower for k in law.parameters
            ):
                results[law.law_id] = {
                    "type": law.law_type.value,
                    "parameters": law.parameters,
                }

        if "laws" in param_lower or "all" in param_lower:
            return json.dumps(
                {law.law_id: law.to_dict() for law in self.laws}, indent=2
            )

        if "count" in param_lower and "law" in param_lower:
            return json.dumps({"law_count": len(self.laws)})

        if "entity" in param_lower and "count" in param_lower:
            last = self.history[-1]
            return json.dumps({"alive_entities": len([e for e in last.entities if e.alive])})

        if results:
            return json.dumps(results, indent=2)

        return json.dumps({"error": "Parameter not found. Try: 'motion laws', 'interaction laws', 'all laws', 'entity count', 'law count'."})

    def simulate_forward(self, steps: int = 5) -> str:
        """Tool: Run the physics engine forward from the current state."""
        new_states = self.simulate(steps)
        return json.dumps([s.to_dict() for s in new_states], indent=2)

    def check_claim(self, claim: str) -> str:
        """Tool: Check a specific factual claim about the universe (TRUE/FALSE)."""
        claim_lower = claim.lower().strip()

        for law in self.laws:
            desc_lower = law.description.lower()
            # Check for key term matches
            if law.law_type == LawType.CONSERVATION:
                qty = law.parameters["conserved_quantity"]
                if qty in claim_lower and ("conserv" in claim_lower):
                    if "strict" in claim_lower and law.parameters["strict"]:
                        return json.dumps({"claim": claim, "result": True})
                    elif "strict" in claim_lower and not law.parameters["strict"]:
                        return json.dumps({"claim": claim, "result": False})
                    elif "conserv" in claim_lower:
                        return json.dumps({"claim": claim, "result": True})

            if law.law_type == LawType.INTERACTION:
                outcome = law.parameters["outcome"]
                if outcome in claim_lower and "interact" in claim_lower:
                    return json.dumps({"claim": claim, "result": True})

            if law.law_type == LawType.THRESHOLD:
                effect = law.parameters["effect"]
                trigger = law.parameters["trigger_variable"]
                if effect in claim_lower and trigger in claim_lower:
                    return json.dumps({"claim": claim, "result": True})

            if law.law_type == LawType.DECAY:
                if "decay" in claim_lower:
                    product = law.parameters["decay_product"]
                    if product in claim_lower:
                        return json.dumps({"claim": claim, "result": True})

        # Check numerical claims about current state
        last = self.history[-1]
        if "entities" in claim_lower and any(c.isdigit() for c in claim_lower):
            alive = len([e for e in last.entities if e.alive])
            for word in claim_lower.split():
                if word.isdigit() and int(word) == alive:
                    return json.dumps({"claim": claim, "result": True})

        return json.dumps({"claim": claim, "result": False, "note": "Claim could not be verified against known laws."})

    # ── Prompt Formatters ─────────────────────────────────────────────────

    def format_observation_log(self, start: int = 0, end: int = 20) -> str:
        """Format simulation history as a structured observation log for the LLM."""
        if len(self.history) < end:
            self.simulate(end - len(self.history) + 1)

        lines = ["# Micro-Universe Observation Log", ""]
        lines.append(f"Grid size: {self.grid_size} × {self.grid_size} (coordinates wrap at boundaries)")
        lines.append(f"Observation window: timesteps {start}–{end-1}")
        lines.append("")

        for state in self.history[start:end]:
            lines.append(f"## Timestep {state.timestep}")
            alive = [e for e in state.entities if e.alive]
            lines.append(f"Alive entities: {len(alive)} | Global field: {state.global_field} | Total energy: {state.total_energy}")
            for e in alive:
                speed = round(math.sqrt(e.vx**2 + e.vy**2), 3)
                lines.append(
                    f"  {e.entity_id} [{e.kind}]: pos=({e.x},{e.y}) "
                    f"vel=({e.vx},{e.vy}) mass={e.mass} charge={e.charge} "
                    f"energy={e.energy} age={e.age} speed={speed}"
                )
            if state.events:
                lines.append(f"  Events: {'; '.join(state.events)}")
            lines.append("")

        return "\n".join(lines)

    def get_ground_truth(self) -> Dict:
        """Return the complete ground truth (for scoring only)."""
        return {
            "seed": self.seed,
            "complexity": self.complexity,
            "laws": [law.to_dict() for law in self.laws],
            "law_descriptions": [law.description for law in self.laws],
        }

    def generate_prediction_scenario(self) -> Tuple[Dict, List[Dict]]:
        """Generate a new initial state for prediction testing."""
        import copy

        # Save RNG state
        saved_rng_state = self.rng.getstate()

        # Create slightly different initial conditions
        last_state = self.history[-1]
        entities = []
        for e in last_state.entities:
            if not e.alive:
                continue
            new_e = copy.deepcopy(e)
            new_e.vx = round(new_e.vx + self.rng.uniform(-0.3, 0.3), 4)
            new_e.vy = round(new_e.vy + self.rng.uniform(-0.3, 0.3), 4)
            entities.append(new_e)

        initial = UniverseState(
            timestep=0,
            entities=entities,
            global_field=last_state.global_field,
            total_energy=sum(e.energy for e in entities),
        )

        # Save current history, run prediction, restore
        saved_history = self.history
        self.history = [initial]
        future = self.simulate(5)
        actual_outcomes = [s.to_dict() for s in future]
        self.history = saved_history

        # Restore RNG
        self.rng.setstate(saved_rng_state)

        return initial.to_dict(), actual_outcomes

    def generate_disconfirming_scenario(self) -> Tuple[Dict, List[Dict], str]:
        """
        Generate a scenario specifically designed to partially contradict
        typical initial hypotheses. Returns (initial_state, outcomes, hint).
        """
        import copy

        # Create edge-case initial conditions that trigger unusual behavior
        last_state = self.history[-1]
        entities = []
        for e in last_state.entities:
            if not e.alive:
                continue
            new_e = copy.deepcopy(e)
            # Push entities toward interaction thresholds
            new_e.energy = round(new_e.energy * 2.5, 4)
            new_e.charge = round(-new_e.charge, 4)  # Flip charges
            entities.append(new_e)

        # Cluster entities closer together to force interactions
        if len(entities) >= 2:
            center_x = sum(e.x for e in entities) / len(entities)
            center_y = sum(e.y for e in entities) / len(entities)
            for e in entities:
                e.x = round(center_x + (e.x - center_x) * 0.3, 4)
                e.y = round(center_y + (e.y - center_y) * 0.3, 4)

        initial = UniverseState(
            timestep=0,
            entities=entities,
            global_field=last_state.global_field * 2,
            total_energy=sum(e.energy for e in entities),
        )

        saved_history = self.history
        self.history = [initial]
        future = self.simulate(5)
        actual_outcomes = [s.to_dict() for s in future]
        self.history = saved_history

        hint = (
            "Note: This scenario uses modified initial conditions (higher energies, "
            "flipped charges, clustered positions) to test edge cases. Your previous "
            "hypotheses may not fully explain the observed behavior."
        )

        return initial.to_dict(), actual_outcomes, hint
