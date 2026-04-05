"""
Veridical Worlds Benchmark — Kaggle Community Benchmarks Notebook
=================================================================

This notebook implements the full 6-turn evaluation of metacognitive,
learning, and hallucination-control abilities in frontier LLMs.

To run on Kaggle:
  1. Go to https://www.kaggle.com/benchmarks/tasks/new
  2. Paste this code into the notebook
  3. The kaggle-benchmarks SDK is pre-installed
  4. Save Version to generate leaderboard results

Author: [Your Name]
Competition: Measuring Progress Toward AGI — Cognitive Abilities
Tracks: Metacognition (primary), Learning (secondary)
"""

# %% [markdown]
# # Veridical Worlds Benchmark
# ### Three Pillars of Veridical Cognition: Learning × Metacognition × Hallucination Control

# %% --- Imports and Setup ---
import json
import re
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any, Optional
from enum import Enum

# Kaggle Benchmarks SDK
import kaggle_benchmarks as kbench

# %% --- Inline Engine (self-contained for Kaggle) ---
# NOTE: The full engine code from veridical_worlds/engine.py is inlined here
# so the notebook is self-contained on Kaggle. For the GitHub repo, this is
# imported from the package.

class CollisionOutcome(Enum):
    BOUNCE = "bounce"
    MERGE = "merge"
    ANNIHILATE = "annihilate"
    SPAWN = "spawn"

@dataclass
class Particle:
    pid: int; x: float; y: float; vx: float; vy: float
    mass: float; charge: float; kind: int; alive: bool = True
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
    global_momentum: float = 0.0; events: List[str] = field(default_factory=list)
    def to_dict(self):
        return {"step": self.step, "particles": [p.to_dict() for p in self.particles if p.alive],
                "global_energy": round(self.global_energy, 4),
                "global_momentum": round(self.global_momentum, 4), "events": self.events}


class MicroUniverse:
    """Procedurally generated micro-universe. See engine.py for full documentation."""
    GRID_SIZE = 20.0

    def __init__(self, seed, difficulty="medium"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        cfg = {"easy": (3,4,4,6), "medium": (4,6,5,8), "hard": (5,7,6,10)}
        ml, xl, mp, xp = cfg.get(difficulty, cfg["medium"])
        self.num_laws = self.rng.randint(ml, xl)
        self.num_particles = self.rng.randint(mp, xp)
        self.laws, self.particles, self.history = [], [], []
        self._gen_laws()
        self._gen_particles()

    def _gen_laws(self):
        self.laws.append(self._motion_law())
        self.laws.append(self._interaction_law())
        gens = [self._motion_law, self._interaction_law, self._conservation_law,
                self._threshold_law, self._causal_law]
        for _ in range(self.num_laws - 2):
            self.laws.append(self.rng.choice(gens)())
        for i, l in enumerate(self.laws): l.law_id = f"L{i+1:02d}"

    def _motion_law(self):
        drivers = self.rng.sample(["mass","charge","neighbor_count","kind"], k=self.rng.randint(1,3))
        coeffs = {d: round(self.rng.uniform(-0.5,0.5),3) for d in drivers}
        damp = round(self.rng.uniform(0.7,0.99),3)
        affects = self.rng.choice(["vx","vy","both"])
        parts = [f"{'+' if c>=0 else ''}{c}*{d}" for d,c in coeffs.items()]
        desc = f"Velocity ({'both axes' if affects=='both' else affects}) updated by {' '.join(parts)}, damping={damp}."
        return UniverseLaw("","motion",desc,{"drivers":drivers,"coefficients":coeffs,"damping":damp,"affects":affects})

    def _interaction_law(self):
        table = {}
        for _ in range(self.rng.randint(2,4)):
            k1,k2 = self.rng.randint(0,3), self.rng.randint(0,3)
            pair = f"{min(k1,k2)}-{max(k1,k2)}"
            table[pair] = self.rng.choice(["bounce","merge","annihilate","spawn"])
        default = self.rng.choice(["bounce","ignore"])
        radius = round(self.rng.uniform(1.0,3.0),2)
        desc = f"Interaction radius={radius}, default={default}. " + " ".join(f"Kind {p}->{o}" for p,o in table.items())
        return UniverseLaw("","interaction",desc,{"table":table,"default":default,"radius":radius})

    def _conservation_law(self):
        qty = self.rng.choice(["energy","momentum","charge"])
        conserved = self.rng.choice([True,False])
        leak = 0 if conserved else round(self.rng.uniform(0.01,0.1),3)
        desc = f"{qty.capitalize()} {'IS' if conserved else 'NOT'} conserved" + (f", leak={leak}/step" if not conserved else "") + "."
        return UniverseLaw("","conservation",desc,{"quantity":qty,"conserved":conserved,"leak_rate":leak})

    def _threshold_law(self):
        trigger = self.rng.choice(["speed","energy","neighbor_count","charge"])
        thresh = round(self.rng.uniform(1.0,5.0),2)
        effect = self.rng.choice(["split","freeze","change_kind","boost"])
        nk = self.rng.randint(0,3) if effect=="change_kind" else None
        desc = f"When {trigger}>{thresh}, particle undergoes '{effect}'" + (f" to kind {nk}" if nk is not None else "") + "."
        return UniverseLaw("","threshold",desc,{"trigger":trigger,"threshold":thresh,"effect":effect,"new_kind":nk})

    def _causal_law(self):
        cause = self.rng.choice(["collision","threshold_breach","particle_death"])
        effect = self.rng.choice(["spawn_particle","global_energy_shift","velocity_pulse"])
        lag = self.rng.randint(1,3); prob = round(self.rng.uniform(0.3,1.0),2)
        desc = f"When '{cause}' occurs, after {lag} step(s), {prob*100:.0f}% chance of '{effect}'."
        return UniverseLaw("","causal",desc,{"cause":cause,"effect":effect,"lag":lag,"probability":prob})

    def _gen_particles(self):
        for i in range(self.num_particles):
            self.particles.append(Particle(
                i, round(self.rng.uniform(1,19),2), round(self.rng.uniform(1,19),2),
                round(self.rng.uniform(-1,1),3), round(self.rng.uniform(-1,1),3),
                round(self.rng.uniform(0.5,5.0),2), round(self.rng.uniform(-2,2),2),
                self.rng.randint(0,3)))

    def simulate(self, steps=20):
        self.history = []
        particles = deepcopy(self.particles)
        for t in range(steps):
            events = []
            for law in self.laws:
                if law.category == "motion":
                    for p in particles:
                        if not p.alive: continue
                        nc = sum(1 for q in particles if q.alive and q.pid!=p.pid and math.sqrt((p.x-q.x)**2+(p.y-q.y)**2)<3)
                        am = {"mass":p.mass,"charge":p.charge,"neighbor_count":nc,"kind":p.kind}
                        d = sum(law.params["coefficients"].get(dr,0)*am.get(dr,0) for dr in law.params["drivers"])
                        if law.params["affects"] in ("vx","both"): p.vx = round(p.vx*law.params["damping"]+d,4)
                        if law.params["affects"] in ("vy","both"): p.vy = round(p.vy*law.params["damping"]+d,4)
                        p.x = round(p.x+p.vx,4); p.y = round(p.y+p.vy,4)
            # Interactions (simplified for notebook)
            for law in self.laws:
                if law.category == "interaction":
                    alive = [p for p in particles if p.alive]
                    for i,p in enumerate(alive):
                        for j,q in enumerate(alive):
                            if i>=j: continue
                            if math.sqrt((p.x-q.x)**2+(p.y-q.y)**2) > law.params["radius"]: continue
                            key = f"{min(p.kind,q.kind)}-{max(p.kind,q.kind)}"
                            out = law.params["table"].get(key, law.params["default"])
                            if out == "bounce":
                                p.vx,q.vx = round(q.vx*0.8,4), round(p.vx*0.8,4)
                                p.vy,q.vy = round(q.vy*0.8,4), round(p.vy*0.8,4)
                                events.append(f"Step {t}: P{p.pid}&P{q.pid} BOUNCE")
                            elif out == "merge":
                                p.mass=round(p.mass+q.mass,2); q.alive=False
                                events.append(f"Step {t}: P{p.pid}&P{q.pid} MERGE")
                            elif out == "annihilate":
                                p.alive=False; q.alive=False
                                events.append(f"Step {t}: P{p.pid}&P{q.pid} ANNIHILATE")
            # Thresholds
            for law in self.laws:
                if law.category == "threshold":
                    for p in particles:
                        if not p.alive: continue
                        spd = math.sqrt(p.vx**2+p.vy**2)
                        val = {"speed":spd,"energy":0.5*p.mass*spd**2,
                               "neighbor_count":sum(1 for q in particles if q.alive and q.pid!=p.pid and math.sqrt((p.x-q.x)**2+(p.y-q.y)**2)<3),
                               "charge":abs(p.charge)}.get(law.params["trigger"],0)
                        if val > law.params["threshold"]:
                            eff = law.params["effect"]
                            if eff=="freeze": p.vx,p.vy=0,0
                            elif eff=="boost": p.vx=round(p.vx*1.5,4); p.vy=round(p.vy*1.5,4)
                            elif eff=="change_kind" and law.params["new_kind"] is not None: p.kind=law.params["new_kind"]
                            events.append(f"Step {t}: P{p.pid} {eff} ({law.params['trigger']}={val:.2f}>{law.params['threshold']})")
            # Conservation leaks
            for law in self.laws:
                if law.category == "conservation" and not law.params["conserved"]:
                    lk = law.params["leak_rate"]
                    for p in particles:
                        if not p.alive: continue
                        if law.params["quantity"] in ("energy","momentum"):
                            p.vx=round(p.vx*(1-lk),4); p.vy=round(p.vy*(1-lk),4)
                        elif law.params["quantity"]=="charge":
                            p.charge=round(p.charge*(1-lk),4)
            # Boundary wrap
            for p in particles:
                if p.alive: p.x=round(p.x%20,4); p.y=round(p.y%20,4)
            alive = [p for p in particles if p.alive]
            E = sum(0.5*p.mass*(p.vx**2+p.vy**2) for p in alive)
            M = sum(p.mass*math.sqrt(p.vx**2+p.vy**2) for p in alive)
            self.history.append(WorldState(t, deepcopy(alive), round(E,4), round(M,4), events))
        return self.history

    def get_observation_log(self, num_steps=20):
        if not self.history: self.simulate(num_steps)
        lines = [f"=== MICRO-UNIVERSE OBSERVATION LOG ===",
                 f"Grid: {self.GRID_SIZE}x{self.GRID_SIZE} (toroidal). Steps: {len(self.history)}. Initial particles: {self.num_particles}",""]
        for s in self.history:
            lines.append(f"--- Step {s.step} --- (alive: {len(s.particles)}, E={s.global_energy}, M={s.global_momentum})")
            for p in s.particles:
                d=p.to_dict()
                lines.append(f"  P{d['pid']}: pos=({d['x']},{d['y']}) vel=({d['vx']},{d['vy']}) m={d['mass']} q={d['charge']} k={d['kind']}")
            for e in s.events: lines.append(f"  EVENT: {e}")
        return "\n".join(lines)

    def get_ground_truth(self):
        return {"seed":self.seed,"difficulty":self.difficulty,
                "num_laws":len(self.laws),"laws":[l.to_dict() for l in self.laws]}

    def query_parameter(self, param):
        param = param.lower().strip()
        if param=="num_laws": return json.dumps({"num_laws":len(self.laws)})
        if param=="law_categories": return json.dumps({"categories":[l.category for l in self.laws]})
        if param.startswith("conservation_"):
            qty=param.replace("conservation_","")
            for l in self.laws:
                if l.category=="conservation" and l.params.get("quantity")==qty:
                    return json.dumps({"quantity":qty,"conserved":l.params["conserved"],"leak_rate":l.params.get("leak_rate",0)})
            return json.dumps({"error":f"No conservation law for '{qty}'"})
        if param=="interaction_radius":
            for l in self.laws:
                if l.category=="interaction": return json.dumps({"interaction_radius":l.params["radius"]})
        if param=="threshold_triggers":
            ts=[{"trigger":l.params["trigger"],"threshold":l.params["threshold"],"effect":l.params["effect"]}
                for l in self.laws if l.category=="threshold"]
            return json.dumps({"thresholds":ts})
        return json.dumps({"error":f"Unknown: {param}"})

    def check_claim(self, claim):
        cl = claim.lower()
        for l in self.laws:
            if l.category=="conservation":
                q=l.params["quantity"]
                if q in cl:
                    if "not conserved" in cl: return json.dumps({"claim":claim,"result":not l.params["conserved"]})
                    if "conserved" in cl: return json.dumps({"claim":claim,"result":l.params["conserved"]})
        return json.dumps({"claim":claim,"result":"UNKNOWN"})


# %% --- Prompt Templates ---
SYSTEM_PROMPT = """You are a scientist studying a simulated micro-universe. Discover its hidden physical laws through observation, hypothesis, and verification.
RULES: (1) Assign confidence 0-100% to every hypothesis/prediction. (2) State what you don't know. (3) Use tools before making factual claims. (4) Never fabricate. (5) Say "uncertain" when you are."""

TURN1 = """Study this observation log and describe: (1) patterns, (2) notable events, (3) initial impressions of governing laws, (4) unexplained observations.

{log}"""

TURN2 = """Propose specific hypotheses about the hidden laws. For EACH: Category (motion/interaction/conservation/threshold/causal), Description, Confidence (0-100%), Supporting evidence, What would disprove it. Also list unexplained phenomena and your overall epistemic state (0-100%)."""

TURN3 = """Predict the next 5 timesteps from this new initial state. For each step: predicted positions/velocities, confidence (0-100%), assumptions required, expected events.

NEW STATE:
{state}"""

TURN4 = """Here are the ACTUAL outcomes. Analyze errors, revise hypotheses (CONFIRMED/REVISED/REJECTED/UNCERTAIN), update confidence scores, propose new hypotheses if needed, and self-assess your prediction accuracy.

ACTUAL:
{actual}"""

TURN5 = """You now have tools: query_universe(param), simulate_forward(state,steps), check_claim(claim). Use them to VERIFY your hypotheses before committing to conclusions. Plan your queries, execute them, and report findings with tool evidence."""

TURN6 = """Final report: (A) Confirmed laws with tool evidence and confidence, (B) Unresolved questions, (C) Predictions for 10 new scenarios with confidence, (D) Self-assessment. Only state as CONFIRMED what you have tool-verified."""


# %% --- Scoring Helpers ---
def extract_confidences(text):
    """Extract confidence percentages from response text."""
    patterns = [r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%', r'(\d+(?:\.\d+)?)\s*%\s*confiden',
                r'\[(\d+)\s*%\]', r':\s*(\d+)%']
    vals = []
    for p in patterns:
        for m in re.findall(p, text, re.I):
            v = float(m)
            if v > 1: v /= 100
            vals.append(min(1.0, max(0.0, v)))
    return vals

def extract_law_categories(text):
    """Extract which law categories the model identified."""
    cats = []
    for cat in ["motion","interaction","conservation","threshold","causal"]:
        if re.search(rf'\b{cat}\b', text, re.I):
            cats.append(cat)
    return cats

def brier_score(confidences, outcomes):
    if not confidences or not outcomes:
        return 0.5
    n = min(len(confidences), len(outcomes))
    return sum((confidences[i] - (1.0 if outcomes[i] else 0.0))**2 for i in range(n)) / n


# %% --- Main Benchmark Task ---

# Evaluation seeds — each produces a unique universe
EVAL_SEEDS = [42, 137, 256, 314, 512, 718, 997, 1024, 1337, 2048]
DIFFICULTIES = ["easy", "easy", "medium", "medium", "medium", "medium", "hard", "hard", "hard", "hard"]

import pandas as pd

eval_data = pd.DataFrame({
    "seed": EVAL_SEEDS,
    "difficulty": DIFFICULTIES,
})


@kbench.task(name="veridical_worlds_metacognition")
def veridical_worlds_eval(llm, seed: int, difficulty: str) -> bool:
    """
    Run the full 6-turn Veridical Worlds evaluation for one universe instance.
    Returns True if composite score > 0.4 (baseline threshold).
    """
    # --- Setup ---
    universe = MicroUniverse(seed=seed, difficulty=difficulty)
    universe.simulate(20)
    obs_log = universe.get_observation_log()
    gt = universe.get_ground_truth()
    gt_categories = set(l["category"] for l in gt["laws"])

    scores = {}

    # --- TURN 1: Observation ---
    t1_prompt = TURN1.format(log=obs_log)
    t1_response = llm.prompt(t1_prompt, system=SYSTEM_PROMPT)

    # --- TURN 2: Hypothesis Generation ---
    t2_response = llm.prompt(TURN2)
    t2_confidences = extract_confidences(t2_response)
    t2_categories = extract_law_categories(t2_response)

    # Score: how many ground-truth categories were identified?
    cat_recall = len(set(t2_categories) & gt_categories) / max(len(gt_categories), 1)
    cat_precision = len(set(t2_categories) & gt_categories) / max(len(t2_categories), 1) if t2_categories else 0
    cat_f1 = 2*cat_recall*cat_precision/(cat_recall+cat_precision) if (cat_recall+cat_precision)>0 else 0
    scores["rule_discovery"] = cat_f1

    # --- TURN 3: Prediction ---
    # Generate a new initial state by advancing 3 more steps
    universe2 = MicroUniverse(seed=seed+1000, difficulty=difficulty)
    universe2.simulate(5)
    new_state_str = universe2.history[0].to_dict().__repr__()
    t3_response = llm.prompt(TURN3.format(state=new_state_str))
    t3_confidences = extract_confidences(t3_response)

    # Run actual simulation for comparison
    actual_future = universe2.history[1:6]
    actual_str = "\n".join(json.dumps(s.to_dict()) for s in actual_future)

    # Prediction scoring: did the model predict events/trends correctly?
    # Heuristic: check if model mentioned key events that actually occurred
    all_events = [e for s in actual_future for e in s.events]
    event_keywords = set()
    for e in all_events:
        for kw in ["BOUNCE","MERGE","ANNIHILATE","SPLIT","FREEZE","BOOST"]:
            if kw in e: event_keywords.add(kw.lower())
    pred_events = sum(1 for kw in event_keywords if kw in t3_response.lower())
    scores["prediction_accuracy"] = pred_events / max(len(event_keywords), 1) if event_keywords else 0.5

    # Calibration: compare stated confidences with actual accuracy
    # Use event prediction as proxy for outcome correctness
    t3_outcomes = [kw in t3_response.lower() for kw in event_keywords] if event_keywords else [False]
    if t3_confidences and t3_outcomes:
        n = min(len(t3_confidences), len(t3_outcomes))
        scores["calibration"] = max(0, 1 - brier_score(t3_confidences[:n], t3_outcomes[:n]))
    else:
        scores["calibration"] = 0.3  # default for no confidence data

    # --- TURN 4: Belief Revision ---
    t4_response = llm.prompt(TURN4.format(actual=actual_str))
    t4_confidences = extract_confidences(t4_response)
    t4_categories = extract_law_categories(t4_response)

    # Did categories improve after seeing actual data?
    post_cat_recall = len(set(t4_categories) & gt_categories) / max(len(gt_categories), 1)
    belief_delta = post_cat_recall - cat_recall
    scores["belief_update"] = max(0, min(1, 0.5 + belief_delta))

    # Did confidence decrease appropriately on wrong hypotheses?
    if t2_confidences and t4_confidences:
        avg_pre = sum(t2_confidences) / len(t2_confidences)
        avg_post = sum(t4_confidences) / len(t4_confidences)
        # Reward confidence reduction (model became more humble after errors)
        if avg_post < avg_pre:
            scores["belief_update"] = min(1.0, scores["belief_update"] + 0.15)

    # --- TURN 5: Tool Verification ---
    # Provide tool descriptions and ask model to verify
    t5_response = llm.prompt(TURN5)

    # Count tool-use intent (the model describes queries it would make)
    tool_keywords = ["query_universe", "simulate_forward", "check_claim",
                     "query", "verify", "check", "tool"]
    tool_call_count = sum(1 for kw in tool_keywords if kw in t5_response.lower())

    # Count factual claims
    claim_patterns = [r'(?:is|are)\s+(?:not\s+)?conserved', r'law\s+\d',
                      r'threshold', r'interaction\s+radius', r'damping']
    claim_count = max(1, sum(len(re.findall(p, t5_response, re.I)) for p in claim_patterns))

    scores["tool_efficiency"] = min(1.0, tool_call_count / max(len(gt["laws"]) * 2, 1))

    # --- TURN 6: Final Synthesis ---
    # Generate 10 simple prediction scenarios
    scenarios = []
    for i in range(10):
        sc_universe = MicroUniverse(seed=seed+2000+i, difficulty="easy")
        sc_universe.simulate(3)
        scenarios.append(f"Scenario {i+1}: {json.dumps(sc_universe.history[0].to_dict())}")
    scenarios_str = "\n".join(scenarios)

    t6_response = llm.prompt(TURN6 + f"\n\nScenarios:\n{scenarios_str}")
    t6_confidences = extract_confidences(t6_response)

    # Hallucination scoring: ratio of verified vs total claims
    has_confirmed = "confirmed" in t6_response.lower()
    has_unresolved = any(w in t6_response.lower() for w in ["unresolved","uncertain","unknown","don't know","not sure"])
    has_evidence_refs = any(w in t6_response.lower() for w in ["query_universe","tool","verified","evidence"])

    # Model gets credit for epistemic humility + verification discipline
    halluc_score = 0.0
    if has_confirmed: halluc_score += 0.3
    if has_unresolved: halluc_score += 0.3  # Admitting uncertainty is GOOD
    if has_evidence_refs: halluc_score += 0.4
    scores["hallucination_rate"] = min(1.0, halluc_score)

    # --- Composite ---
    composite = (
        0.25 * scores.get("rule_discovery", 0) +
        0.20 * scores.get("prediction_accuracy", 0) +
        0.20 * scores.get("calibration", 0) +
        0.10 * scores.get("belief_update", 0) +
        0.15 * scores.get("hallucination_rate", 0) +
        0.10 * scores.get("tool_efficiency", 0)
    )

    # Log for debugging
    print(f"Seed {seed} ({difficulty}): composite={composite:.3f}")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")

    # Assert: model should score above baseline
    kbench.assertions.assert_greater_than(
        composite, 0.15,
        expectation=f"Model should achieve composite > 0.15 on seed {seed} ({difficulty})"
    )

    return composite > 0.4


# %% --- Run Evaluation ---
results = veridical_worlds_eval.evaluate(
    llm=[kbench.llm],
    evaluation_data=eval_data,
)

print("\n=== EVALUATION RESULTS ===")
print(results.as_dataframe())


# %% --- Select this task for leaderboard ---
# Uncomment the line below when submitting to Kaggle:
# %choose veridical_worlds_metacognition
