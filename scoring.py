"""
Veridical Worlds Scoring Engine
================================
Automated scoring for the Veridical Worlds Benchmark.
Evaluates rule discovery, prediction accuracy, confidence calibration,
belief revision, hallucination control, tool-use discipline, and
numerical accuracy.

Features:
- Structural parameter matching (not just keyword detection)
- Claim-evidence linking for hallucination scoring
- Quantitative fabrication detection
- Tolerance-based numerical scoring

Author: Zhelin Zhang
License: Apache 2.0
"""

import re
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TurnScore:
    """Score for a single turn in the evaluation."""
    turn_name: str
    raw_score: float  # 0.0 - 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkScore:
    """Complete benchmark score across all dimensions."""
    rule_discovery: float = 0.0
    prediction_accuracy: float = 0.0
    calibration_brier: float = 0.0
    belief_update: float = 0.0
    hallucination_rate: float = 0.0
    tool_efficiency: float = 0.0
    numerical_accuracy: float = 0.0
    turn_scores: List[TurnScore] = field(default_factory=list)

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        return (
            0.20 * self.rule_discovery
            + 0.15 * self.prediction_accuracy
            + 0.15 * self.calibration_brier
            + 0.10 * self.belief_update
            + 0.15 * self.hallucination_rate
            + 0.10 * self.tool_efficiency
            + 0.15 * self.numerical_accuracy
        )

    def to_dict(self) -> Dict:
        return {
            "composite_score": round(self.composite, 4),
            "rule_discovery_f1": round(self.rule_discovery, 4),
            "prediction_accuracy": round(self.prediction_accuracy, 4),
            "calibration_brier": round(self.calibration_brier, 4),
            "belief_update_quality": round(self.belief_update, 4),
            "hallucination_control": round(self.hallucination_rate, 4),
            "tool_efficiency": round(self.tool_efficiency, 4),
            "numerical_accuracy": round(self.numerical_accuracy, 4),
            "turns": [
                {"name": t.turn_name, "score": round(t.raw_score, 4), "details": t.details}
                for t in self.turn_scores
            ],
        }


class VeridicalScorer:
    """
    Scores model responses against ground truth for the Veridical Worlds Benchmark.
    All scoring is fully automated.
    """

    def __init__(self, ground_truth: Dict):
        self.ground_truth = ground_truth
        self.gt_laws = ground_truth["laws"]
        self.gt_descriptions = ground_truth["law_descriptions"]

    # ── Pillar 1: Rule Discovery (Turn 2) — Structural Matching ──────────

    def score_rule_discovery(self, model_hypotheses: str) -> TurnScore:
        """
        Score how well the model's proposed laws match ground truth.
        Uses structural parameter matching, not just keyword detection.
        Returns F1 score.
        """
        model_text = model_hypotheses.lower()
        model_claims = self._extract_hypotheses(model_text)
        gt_matched = set()
        true_positives = 0

        for i, law in enumerate(self.gt_laws):
            law_type = law["law_type"]
            params = law["parameters"]
            match_score = self._structural_match_law(model_text, law_type, params)
            if match_score > 0.4:
                true_positives += 1
                gt_matched.add(i)

        n_gt = len(self.gt_laws)
        n_model = max(len(model_claims), 1)

        precision = true_positives / n_model if n_model > 0 else 0
        recall = true_positives / n_gt if n_gt > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        return TurnScore(
            turn_name="rule_discovery",
            raw_score=f1,
            details={
                "true_positives": true_positives,
                "ground_truth_laws": n_gt,
                "model_hypotheses": n_model,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "matched_law_indices": list(gt_matched),
            }
        )

    def _structural_match_law(self, text: str, law_type: str, params: Dict) -> float:
        """Score how well text matches a law using structural parameter matching.

        Scoring levels:
          - Category + exact parameter (within 10%): 1.0
          - Category + approximate parameter (within 25%): 0.7
          - Category mentioned + qualitative description: 0.5
          - Category mentioned only: 0.3
          - No match: 0.0
        """
        # Step 1: Check if law category/type is mentioned
        type_keywords = {
            "motion": ["velocity", "motion", "acceleration", "speed", "movement", "move"],
            "interaction": ["interaction", "collide", "collision", "merge", "bounce", "annihilat"],
            "conservation": ["conserv", "preserved", "constant", "maintained"],
            "threshold": ["threshold", "exceed", "trigger", "phase", "transition", "when.*exceed"],
            "causal_chain": ["cause", "causal", "chain", "trigger", "leads to", "results in", "after.*then"],
            "decay": ["decay", "half-life", "degrade", "deteriorat", "vanish"],
            "field_effect": ["field", "global", "ambient", "background"],
            "inverse_square": ["inverse.?square", "1/r", "gravity", "gravitational", "coulomb",
                               "electromagnetic", "r.?squared", "distance.?squared"],
            "oscillation": ["oscillat", "spring", "restor", "harmonic", "equilibrium", "periodic",
                           "back.?and.?forth"],
            "nonlinear_motion": ["nonlinear", "quadratic", "logarithm", "exponential",
                                 "mass.?squared", "charge.?squared"],
        }

        keywords = type_keywords.get(law_type, [])
        category_found = any(re.search(kw, text) for kw in keywords)
        if not category_found:
            return 0.0

        # Step 2: Extract numbers from text near relevant keywords
        score = 0.3  # base score for category match
        param_matches = 0
        param_total = 0

        for key, val in params.items():
            if isinstance(val, (int, float)) and key not in ("axis", "applies_to", "strict",
                                                              "attractive", "conserved", "formula"):
                param_total += 1
                # Search for this number in the text
                numbers_in_text = [float(n) for n in re.findall(r'-?\d+\.?\d*', text)
                                   if self._is_number(n)]
                for num in numbers_in_text:
                    if val == 0:
                        if abs(num) < 0.05:
                            param_matches += 1
                            break
                    elif abs(num - float(val)) / max(abs(float(val)), 0.001) < 0.10:
                        param_matches += 1  # exact (within 10%)
                        break
                    elif abs(num - float(val)) / max(abs(float(val)), 0.001) < 0.25:
                        param_matches += 0.5  # approximate (within 25%)
                        break

            elif isinstance(val, str) and key not in ("axis",):
                param_total += 1
                if val.lower() in text:
                    param_matches += 1

            elif isinstance(val, bool):
                param_total += 1
                key_lower = key.lower().replace("_", " ")
                if val and key_lower in text:
                    param_matches += 1
                elif not val and ("not " + key_lower in text or "no " + key_lower in text):
                    param_matches += 1

        if param_total > 0:
            param_ratio = param_matches / param_total
            score = 0.3 + 0.7 * param_ratio  # Scale from 0.3 (category only) to 1.0 (all params)

        return min(1.0, score)

    def _extract_hypotheses(self, text: str) -> List[str]:
        """Extract individual hypothesis statements from model output."""
        hypotheses = []
        patterns = [
            r'(?:hypothesis|law|rule|observation)\s*\d*\s*[:\-\.]\s*(.+?)(?=\n|$)',
            r'(?:\d+[\.\)]\s*)(.+?)(?=\n\d+[\.\)]|\n\n|$)',
            r'(?:[\-\*]\s*)(.+?)(?=\n[\-\*]|\n\n|$)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            hypotheses.extend(matches)
        if not hypotheses:
            sentences = text.split('.')
            hypotheses = [s.strip() for s in sentences if len(s.strip()) > 20]
        return hypotheses

    @staticmethod
    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    # ── Pillar 1b: Prediction Accuracy (Turn 3) — Quantitative ───────────

    def score_predictions(
        self, model_predictions: str, actual_outcomes: List[Dict]
    ) -> TurnScore:
        """
        Score prediction accuracy using both quantitative (position MAE) and
        qualitative (event prediction) measures.
        """
        # Try quantitative scoring first
        quant_score = self._score_quantitative_predictions(model_predictions, actual_outcomes)

        # Also do qualitative event scoring
        qual_score = self._score_qualitative_predictions(model_predictions, actual_outcomes)

        # Use the better of the two (reward models that give structured output)
        if quant_score is not None:
            combined = 0.6 * quant_score + 0.4 * qual_score
        else:
            combined = qual_score

        return TurnScore(
            turn_name="prediction_accuracy",
            raw_score=combined,
            details={
                "quantitative_score": round(quant_score, 4) if quant_score is not None else None,
                "qualitative_score": round(qual_score, 4),
                "combined": round(combined, 4),
            }
        )

    def _score_quantitative_predictions(self, text: str, actual: List[Dict]) -> Optional[float]:
        """Extract structured predictions and compute position/energy MAE."""
        predicted_states = self._extract_predictions(text)
        if not predicted_states or not actual:
            return None

        total_error = 0.0
        total_comparisons = 0

        for i, actual_state in enumerate(actual):
            if i >= len(predicted_states):
                break
            actual_entities = {e["entity_id"]: e for e in actual_state.get("entities", [])}
            pred = predicted_states[i] if isinstance(predicted_states[i], dict) else {}

            for eid, actual_e in actual_entities.items():
                if eid in pred:
                    pred_e = pred[eid]
                    if "x" in pred_e and "y" in pred_e:
                        dx = abs(float(pred_e.get("x", 0)) - float(actual_e.get("x", 0)))
                        dy = abs(float(pred_e.get("y", 0)) - float(actual_e.get("y", 0)))
                        total_error += (dx + dy) / 40.0
                        total_comparisons += 1
                    if "energy" in pred_e:
                        de = abs(float(pred_e.get("energy", 0)) - float(actual_e.get("energy", 0)))
                        total_error += de / 50.0
                        total_comparisons += 1
                else:
                    total_error += 1.0
                    total_comparisons += 1

        if total_comparisons == 0:
            return None
        mae = total_error / total_comparisons
        return max(0, 1.0 - mae)

    def _score_qualitative_predictions(self, text: str, actual: List[Dict]) -> float:
        """Score qualitative event predictions and trend accuracy."""
        all_events = []
        for state in actual:
            all_events.extend(state.get("events", []))

        # Check event type predictions
        event_keywords = set()
        for e in all_events:
            for kw in ["bounce", "merge", "annihilat", "spawn", "split", "decay",
                        "threshold", "freeze", "charge_flip", "field_pulse", "causal"]:
                if kw.lower() in e.lower():
                    event_keywords.add(kw.lower())

        if not event_keywords:
            return 0.5  # No events to predict

        text_lower = text.lower()
        hits = sum(1 for kw in event_keywords if kw in text_lower)
        event_score = hits / len(event_keywords)

        # Check trend predictions (energy increasing/decreasing)
        if len(actual) >= 2:
            first_energy = actual[0].get("total_energy", 0)
            last_energy = actual[-1].get("total_energy", 0)
            trend = "increas" if last_energy > first_energy else "decreas"
            if trend in text_lower:
                event_score = min(1.0, event_score + 0.2)

        # Check alive count predictions
        if actual:
            final_alive = len(actual[-1].get("entities", []))
            if str(final_alive) in text:
                event_score = min(1.0, event_score + 0.15)

        return min(1.0, event_score)

    def _extract_predictions(self, text: str) -> List[Any]:
        """Extract structured predictions from model output."""
        # Try JSON blocks
        json_blocks = re.findall(r'```(?:json)?\s*(\{.+?\}|\[.+?\])\s*```', text, re.DOTALL)
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except json.JSONDecodeError:
                continue
        # Try inline JSON
        try:
            match = re.search(r'\[.+\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return []

    # ── Pillar 2: Calibration (Brier Score) ──────────────────────────────

    def score_calibration(self, model_output: str, actual_correct: List[bool]) -> TurnScore:
        """Score confidence calibration using Brier score."""
        confidences = self._extract_confidences(model_output)
        if not confidences or not actual_correct:
            return TurnScore(
                turn_name="calibration", raw_score=0.0,
                details={"error": "Could not extract confidence scores"},
            )

        n = min(len(confidences), len(actual_correct))
        confidences = [c / 100.0 for c in confidences[:n]]
        outcomes = [1.0 if correct else 0.0 for correct in actual_correct[:n]]

        brier = sum((c - o) ** 2 for c, o in zip(confidences, outcomes)) / n
        calibration_score = max(0, 1.0 - brier)

        # ECE in 5 bins
        bins = [[] for _ in range(5)]
        for c, o in zip(confidences, outcomes):
            bin_idx = min(int(c * 5), 4)
            bins[bin_idx].append((c, o))
        ece = 0.0
        for b in bins:
            if b:
                avg_conf = sum(c for c, _ in b) / len(b)
                avg_acc = sum(o for _, o in b) / len(b)
                ece += len(b) / n * abs(avg_conf - avg_acc)

        overconfident_count = sum(
            1 for c, o in zip(confidences, outcomes) if c > 0.8 and o < 0.5
        )

        return TurnScore(
            turn_name="calibration",
            raw_score=max(0, calibration_score),
            details={
                "brier_score": round(brier, 4),
                "calibration_score": round(calibration_score, 4),
                "ece": round(ece, 4),
                "n_items": n,
                "mean_confidence": round(sum(confidences) / len(confidences), 3),
                "mean_accuracy": round(sum(outcomes) / len(outcomes), 3),
                "overconfident_items": overconfident_count,
            }
        )

    def _extract_confidences(self, text: str) -> List[float]:
        """Extract numerical confidence scores (0-100) from model output."""
        confidences = []
        patterns = [
            r'confidence[:\s]*(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*confiden',
            r'confidence[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)/100',
            r'\[(\d+)\s*%\]',
            r':\s*(\d+)%',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                val = float(m)
                if val <= 1.0 and val > 0:
                    val *= 100
                if 0 <= val <= 100:
                    confidences.append(val)
        return confidences

    # ── Pillar 2b: Belief Update Quality (Turn 4) ────────────────────────

    def score_belief_update(
        self, pre_update_text: str, post_update_text: str
    ) -> TurnScore:
        """Score the quality of belief revision after disconfirming evidence."""
        post_lower = post_update_text.lower()

        revision_markers = {
            "error_acknowledgment": [
                "was wrong", "incorrect", "revise", "update", "mistaken",
                "error", "reconsider", "flawed", "inaccurate", "abandon",
                "my prediction was off", "i overestimated", "i underestimated",
            ],
            "confidence_adjustment": [
                "lower confidence", "reduce confidence", "less certain",
                "uncertain", "decrease", "downgrade", "revised confidence",
                "from.*%.*to.*%", "lowering.*confidence",
            ],
            "explicit_uncertainty": [
                "don't know", "uncertain", "unclear", "cannot determine",
                "insufficient", "not enough evidence", "ambiguous",
                "open question", "unresolved", "remain unsure",
            ],
            "hypothesis_specificity": [
                "hypothesis 1", "hypothesis 2", "law 1", "law 2",
                "specifically", "in particular", "the motion law",
                "the interaction", "my earlier claim", "rejected",
                "confirmed", "revised",
            ],
        }

        scores = {}
        for category, markers in revision_markers.items():
            hits = sum(1 for m in markers if re.search(m, post_lower))
            scores[category] = min(1.0, hits / 3.0)

        doubling_down_markers = [
            "still confident", "original hypothesis holds",
            "no revision needed", "confirmed my earlier",
        ]
        doubling_penalty = sum(0.15 for m in doubling_down_markers if m in post_lower)

        raw_score = (
            0.3 * scores["error_acknowledgment"]
            + 0.25 * scores["confidence_adjustment"]
            + 0.25 * scores["explicit_uncertainty"]
            + 0.2 * scores["hypothesis_specificity"]
            - doubling_penalty
        )
        raw_score = max(0, min(1, raw_score))

        return TurnScore(
            turn_name="belief_update",
            raw_score=raw_score,
            details={
                "category_scores": {k: round(v, 3) for k, v in scores.items()},
                "doubling_down_penalty": round(doubling_penalty, 3),
            }
        )

    # ── Pillar 3: Hallucination Control — Claim-Evidence Linking ─────────

    def score_hallucination_control(
        self, model_output: str, tool_calls_made: List[str]
    ) -> TurnScore:
        """
        Score whether claims are backed by tool-call evidence.

        Scoring per claim:
          - Correct + tool-verified: +1.0
          - Correct but unverified: +0.3
          - Incorrect + stated as confirmed: -0.5
          - Flagged as uncertain when genuinely unknown: +0.3
        """
        claims = self._extract_claims(model_output)
        if not claims:
            return TurnScore(
                turn_name="hallucination_control",
                raw_score=0.5,
                details={"note": "No factual claims extracted"},
            )

        tool_text = " ".join(tool_calls_made).lower()
        total_score = 0.0
        verified_count = 0
        unverified_claims = []

        for claim in claims:
            claim_lower = claim.lower()

            # Check if claim references tool evidence
            claim_kws = set(re.findall(r'\b\w{4,}\b', claim_lower))
            tool_kws = set(re.findall(r'\b\w{4,}\b', tool_text))
            has_tool_evidence = (
                len(claim_kws & tool_kws) >= 2
                or any(m in claim_lower for m in ["verified", "confirmed via tool",
                                                   "query showed", "tool result"])
            )

            # Check if claim matches a ground truth law
            matches_gt = False
            for law in self.gt_laws:
                match = self._structural_match_law(claim_lower, law["law_type"], law["parameters"])
                if match > 0.3:
                    matches_gt = True
                    break

            # Check if claim is flagged as uncertain
            is_uncertain = any(m in claim_lower for m in [
                "unresolved", "not verified", "uncertain", "unknown",
                "unclear", "open question", "cannot confirm",
            ])

            if matches_gt and has_tool_evidence:
                total_score += 1.0
                verified_count += 1
            elif matches_gt and not has_tool_evidence:
                total_score += 0.3  # correct but didn't verify
            elif not matches_gt and "confirmed" in claim_lower:
                total_score -= 0.5  # hallucination penalty
                unverified_claims.append(claim[:80])
            elif is_uncertain:
                total_score += 0.3  # honesty bonus
                verified_count += 1

        n = len(claims)
        hallucination_score = max(0, min(1.0, total_score / n))

        return TurnScore(
            turn_name="hallucination_control",
            raw_score=hallucination_score,
            details={
                "total_claims": n,
                "verified_claims": verified_count,
                "unverified_claims": unverified_claims[:5],
                "raw_total": round(total_score, 3),
            }
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual assertion statements from model output."""
        claims = []
        sentences = re.split(r'[.!]', text)
        assertive_markers = [
            "is ", "are ", "was ", "were ", "has ", "have ", "does ", "will ",
            "causes ", "results in", "leads to", "produces ", "equals ",
            "conserved", "decays", "interacts", "triggers", "follows",
            "governs", "obeys", "exhibits",
        ]
        for sentence in sentences:
            s = sentence.strip()
            if len(s) < 15:
                continue
            if any(marker in s.lower() for marker in assertive_markers):
                claims.append(s)
        return claims

    # ── Pillar 3b: Tool Efficiency (Turn 5) ──────────────────────────────

    def score_tool_efficiency(
        self, tool_calls_made: List[str], n_claims: int
    ) -> TurnScore:
        """Score tool-use efficiency."""
        actual = len(tool_calls_made)
        optimal = max(n_claims, 1)
        if actual == 0:
            efficiency = 0.0
        elif actual <= optimal:
            efficiency = actual / optimal
        else:
            efficiency = min(1.0, optimal / actual)

        return TurnScore(
            turn_name="tool_efficiency",
            raw_score=efficiency,
            details={
                "tool_calls_made": actual,
                "optimal_calls": optimal,
                "efficiency": round(efficiency, 3),
            }
        )

    # ── Pillar 4: Numerical Accuracy — Quantitative Verification ─────────

    def score_numerical_accuracy(
        self, model_output: str,
        challenges: List[tuple],
    ) -> TurnScore:
        """
        Score the model's numerical answers against ground-truth values.

        challenges: list of (question_str, ground_truth_float) tuples
        Tolerance: within 10% = full credit, within 25% = half credit
        """
        if not challenges:
            return TurnScore(
                turn_name="numerical_accuracy",
                raw_score=0.5,
                details={"note": "No quantitative challenges provided"},
            )

        text_lower = model_output.lower()
        total_score = 0.0
        results = []

        for question, gt_val in challenges:
            # Find the section of text most likely answering this question
            q_keywords = set(re.findall(r'\b\w{4,}\b', question.lower()))
            best_match_score = 0.0

            # Extract all numbers from the response
            numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', text_lower)
                       if self._is_number(n) and abs(float(n)) < 10000]

            for num in numbers:
                if gt_val == 0:
                    if abs(num) < 0.05:
                        best_match_score = 1.0
                        break
                else:
                    rel_error = abs(num - gt_val) / max(abs(gt_val), 0.001)
                    if rel_error < 0.10:
                        best_match_score = 1.0
                        break
                    elif rel_error < 0.25:
                        best_match_score = max(best_match_score, 0.5)

            total_score += best_match_score
            results.append({
                "question": question[:60],
                "ground_truth": gt_val,
                "score": best_match_score,
            })

        accuracy = total_score / len(challenges)

        return TurnScore(
            turn_name="numerical_accuracy",
            raw_score=accuracy,
            details={
                "challenges": len(challenges),
                "accuracy": round(accuracy, 4),
                "per_question": results,
            }
        )

    # ── Composite Scoring ─────────────────────────────────────────────────

    def compute_composite(
        self,
        rule_discovery_score: TurnScore,
        prediction_score: TurnScore,
        calibration_score: TurnScore,
        belief_update_score: TurnScore,
        hallucination_score: TurnScore,
        tool_score: TurnScore,
        numerical_score: Optional[TurnScore] = None,
    ) -> BenchmarkScore:
        """Combine all sub-scores into a final BenchmarkScore."""
        result = BenchmarkScore(
            rule_discovery=rule_discovery_score.raw_score,
            prediction_accuracy=prediction_score.raw_score,
            calibration_brier=calibration_score.raw_score,
            belief_update=belief_update_score.raw_score,
            hallucination_rate=hallucination_score.raw_score,
            tool_efficiency=tool_score.raw_score,
            numerical_accuracy=numerical_score.raw_score if numerical_score else 0.0,
            turn_scores=[
                rule_discovery_score,
                prediction_score,
                calibration_score,
                belief_update_score,
                hallucination_score,
                tool_score,
            ] + ([numerical_score] if numerical_score else []),
        )
        return result
