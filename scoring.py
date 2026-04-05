"""
Veridical Worlds Scoring Engine
================================
Automated scoring for the Veridical Worlds Benchmark.
Evaluates rule discovery, prediction accuracy, confidence calibration,
belief revision, hallucination control, and tool-use discipline.

Author: [Your Name]
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
    rule_discovery: float = 0.0  # F1 of discovered vs ground-truth laws
    prediction_accuracy: float = 0.0  # 1 - normalized MAE
    calibration_brier: float = 0.0  # 1 - Brier score
    belief_update: float = 0.0  # Quality of revision after disconfirmation
    hallucination_rate: float = 0.0  # 1 - (unverified / total claims)
    tool_efficiency: float = 0.0  # Optimal / actual tool calls
    turn_scores: List[TurnScore] = field(default_factory=list)

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        return (
            0.25 * self.rule_discovery
            + 0.20 * self.prediction_accuracy
            + 0.20 * self.calibration_brier
            + 0.10 * self.belief_update
            + 0.15 * self.hallucination_rate
            + 0.10 * self.tool_efficiency
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
            "turns": [
                {"name": t.turn_name, "score": round(t.raw_score, 4), "details": t.details}
                for t in self.turn_scores
            ],
        }


class VeridicalScorer:
    """
    Scores model responses against ground truth for the Veridical Worlds Benchmark.
    All scoring is fully automated — no human grading required.
    """

    def __init__(self, ground_truth: Dict):
        self.ground_truth = ground_truth
        self.gt_laws = ground_truth["laws"]
        self.gt_descriptions = ground_truth["law_descriptions"]

    # ── Pillar 1: Rule Discovery (Turn 2) ─────────────────────────────────

    def score_rule_discovery(self, model_hypotheses: str) -> TurnScore:
        """
        Score how well the model's proposed laws match ground truth.
        Uses keyword matching against law descriptions and parameters.
        Returns F1 score.
        """
        model_text = model_hypotheses.lower()
        gt_matched = set()
        model_claims = self._extract_hypotheses(model_text)

        true_positives = 0
        for i, law in enumerate(self.gt_laws):
            law_type = law["law_type"]
            params = law["parameters"]

            # Check if model identified this law type and key parameters
            match_score = self._match_law(model_text, law_type, params)
            if match_score > 0.5:
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

    def _match_law(self, text: str, law_type: str, params: Dict) -> float:
        """Score how well text matches a specific law."""
        score = 0.0
        total_checks = 0

        # Check law type mentioned
        type_keywords = {
            "motion": ["velocity", "motion", "acceleration", "speed", "movement", "move"],
            "interaction": ["interaction", "collide", "collision", "merge", "bounce", "annihilat"],
            "conservation": ["conserv", "preserved", "constant", "maintained"],
            "threshold": ["threshold", "exceed", "trigger", "phase", "transition"],
            "causal_chain": ["cause", "causal", "chain", "trigger", "leads to", "results in"],
            "decay": ["decay", "half-life", "degrade", "deteriorat", "vanish"],
            "field_effect": ["field", "global", "ambient", "background"],
        }

        keywords = type_keywords.get(law_type, [])
        if any(kw in text for kw in keywords):
            score += 1.0
        total_checks += 1

        # Check key parameter values mentioned
        for key, val in params.items():
            total_checks += 1
            val_str = str(val).lower()
            if isinstance(val, (int, float)):
                # Check if number approximately mentioned
                if val_str in text or str(round(float(val), 1)) in text:
                    score += 1.0
                elif any(
                    abs(float(val) - float(w)) < 0.5
                    for w in re.findall(r'-?\d+\.?\d*', text)
                    if self._is_number(w)
                ):
                    score += 0.5
            elif isinstance(val, str):
                if val.lower() in text:
                    score += 1.0

        return score / total_checks if total_checks > 0 else 0

    def _extract_hypotheses(self, text: str) -> List[str]:
        """Extract individual hypothesis statements from model output."""
        hypotheses = []
        # Look for numbered lists, bullet points, or "hypothesis" markers
        patterns = [
            r'(?:hypothesis|law|rule|observation)\s*\d*\s*[:\-\.]\s*(.+?)(?=\n|$)',
            r'(?:\d+[\.\)]\s*)(.+?)(?=\n\d+[\.\)]|\n\n|$)',
            r'(?:[\-\*]\s*)(.+?)(?=\n[\-\*]|\n\n|$)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            hypotheses.extend(matches)
        if not hypotheses:
            # Fall back to sentence splitting
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

    # ── Pillar 1b: Prediction Accuracy (Turn 3) ──────────────────────────

    def score_predictions(
        self, model_predictions: str, actual_outcomes: List[Dict]
    ) -> TurnScore:
        """
        Score prediction accuracy by comparing predicted entity positions/states
        against actual simulation outcomes. Uses normalized MAE.
        """
        predicted_states = self._extract_predictions(model_predictions)
        if not predicted_states or not actual_outcomes:
            return TurnScore(
                turn_name="prediction_accuracy",
                raw_score=0.0,
                details={"error": "Could not parse predictions"},
            )

        total_error = 0.0
        total_comparisons = 0

        for i, actual in enumerate(actual_outcomes):
            if i >= len(predicted_states):
                break
            actual_entities = {e["entity_id"]: e for e in actual.get("entities", [])}
            pred_entities = predicted_states[i] if isinstance(predicted_states[i], dict) else {}

            for eid, actual_e in actual_entities.items():
                if eid in pred_entities:
                    pred_e = pred_entities[eid]
                    # Position error
                    if "x" in pred_e and "y" in pred_e:
                        dx = abs(float(pred_e.get("x", 0)) - float(actual_e.get("x", 0)))
                        dy = abs(float(pred_e.get("y", 0)) - float(actual_e.get("y", 0)))
                        total_error += (dx + dy) / 40.0  # Normalize by grid size
                        total_comparisons += 1
                    # Energy error
                    if "energy" in pred_e:
                        de = abs(float(pred_e.get("energy", 0)) - float(actual_e.get("energy", 0)))
                        total_error += de / 50.0  # Normalize
                        total_comparisons += 1
                else:
                    total_error += 1.0
                    total_comparisons += 1

        if total_comparisons == 0:
            mae = 1.0
        else:
            mae = total_error / total_comparisons

        accuracy = max(0, 1.0 - mae)

        return TurnScore(
            turn_name="prediction_accuracy",
            raw_score=accuracy,
            details={
                "normalized_mae": round(mae, 4),
                "accuracy": round(accuracy, 4),
                "comparisons": total_comparisons,
            }
        )

    def _extract_predictions(self, text: str) -> List[Any]:
        """Extract structured predictions from model output."""
        # Try to find JSON blocks
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

    # ── Pillar 2: Calibration (Turns 2, 3, 6) ────────────────────────────

    def score_calibration(self, model_output: str, actual_correct: List[bool]) -> TurnScore:
        """
        Score confidence calibration using Brier score.
        Extracts confidence values (0-100) from model output and compares
        against actual correctness.
        """
        confidences = self._extract_confidences(model_output)

        if not confidences or not actual_correct:
            return TurnScore(
                turn_name="calibration",
                raw_score=0.0,
                details={"error": "Could not extract confidence scores"},
            )

        # Normalize to [0, 1]
        n = min(len(confidences), len(actual_correct))
        confidences = [c / 100.0 for c in confidences[:n]]
        outcomes = [1.0 if correct else 0.0 for correct in actual_correct[:n]]

        # Brier score: mean squared difference between confidence and outcome
        brier = sum((c - o) ** 2 for c, o in zip(confidences, outcomes)) / n
        calibration_score = 1.0 - brier  # Higher is better

        # Compute ECE (Expected Calibration Error) in 5 bins
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

        # Overconfidence detection
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
        # Match patterns like "confidence: 85%", "85% confident", "(confidence: 0.85)"
        patterns = [
            r'confidence[:\s]*(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*confiden',
            r'confidence[:\s]*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)/100',
            r'confidence[:\s]*0\.(\d+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                val = float(m)
                if val <= 1.0:
                    val *= 100
                if 0 <= val <= 100:
                    confidences.append(val)

        return confidences

    # ── Pillar 2b: Belief Update Quality (Turn 4) ────────────────────────

    def score_belief_update(
        self, pre_update_text: str, post_update_text: str
    ) -> TurnScore:
        """
        Score the quality of belief revision after disconfirming evidence.
        Checks for: explicit acknowledgment of errors, revised confidences,
        identification of wrong hypotheses, and intellectual humility markers.
        """
        post_lower = post_update_text.lower()

        # Markers of good belief revision
        revision_markers = {
            "error_acknowledgment": [
                "was wrong", "incorrect", "revise", "update", "mistaken",
                "error", "reconsider", "flawed", "inaccurate", "abandon",
            ],
            "confidence_adjustment": [
                "lower confidence", "reduce confidence", "less certain",
                "uncertain", "decrease", "downgrade", "revised confidence",
            ],
            "explicit_uncertainty": [
                "don't know", "uncertain", "unclear", "cannot determine",
                "insufficient", "not enough evidence", "ambiguous",
                "open question", "unresolved",
            ],
            "hypothesis_specificity": [
                "hypothesis 1", "hypothesis 2", "law 1", "law 2",
                "specifically", "in particular", "the motion law",
                "the interaction", "my earlier claim",
            ],
        }

        scores = {}
        for category, markers in revision_markers.items():
            hits = sum(1 for m in markers if m in post_lower)
            scores[category] = min(1.0, hits / 3.0)  # Cap at 1.0

        # Penalty for doubling down (asserting same claims without revision)
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

    # ── Pillar 3: Hallucination Control (Turns 5-6) ──────────────────────

    def score_hallucination_control(
        self, model_output: str, tool_calls_made: List[str]
    ) -> TurnScore:
        """
        Score whether the model only reports claims it verified via tool calls.
        Penalizes claims made without corresponding tool verification.
        """
        # Extract factual claims from the model's output
        claims = self._extract_claims(model_output)
        total_claims = len(claims) if claims else 1

        # Check which claims have corresponding tool call evidence
        verified = 0
        unverified_claims = []

        tool_text = " ".join(tool_calls_made).lower()

        for claim in claims:
            claim_lower = claim.lower()
            # A claim is "verified" if a related tool call was made
            claim_keywords = set(re.findall(r'\b\w{4,}\b', claim_lower))
            tool_keywords = set(re.findall(r'\b\w{4,}\b', tool_text))
            overlap = claim_keywords & tool_keywords

            if len(overlap) >= 2 or any(
                marker in claim_lower
                for marker in ["verified", "confirmed", "tool", "query", "check"]
            ):
                verified += 1
            else:
                unverified_claims.append(claim[:80])

        # Check for epistemic markers (flagging uncertainty)
        uncertainty_flags = sum(
            1 for marker in [
                "unresolved", "not verified", "uncertain", "requires further",
                "cannot confirm", "unknown", "unclear", "open question",
            ]
            if marker in model_output.lower()
        )

        hallucination_score = verified / total_claims if total_claims > 0 else 0
        # Bonus for explicit uncertainty flagging
        hallucination_score = min(1.0, hallucination_score + 0.05 * uncertainty_flags)

        return TurnScore(
            turn_name="hallucination_control",
            raw_score=hallucination_score,
            details={
                "total_claims": total_claims,
                "verified_claims": verified,
                "unverified_claims": unverified_claims[:5],  # First 5 for debugging
                "uncertainty_flags": uncertainty_flags,
            }
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual assertion statements from model output."""
        claims = []
        # Sentences containing assertive language
        sentences = re.split(r'[.!]', text)
        assertive_markers = [
            "is ", "are ", "was ", "were ", "has ", "have ", "does ", "will ",
            "causes ", "results in", "leads to", "produces ", "equals ",
            "conserved", "decays", "interacts", "triggers",
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
        """
        Score tool-use efficiency. Optimal = one tool call per claim to verify.
        Penalizes both under-use (not verifying) and over-use (redundant calls).
        """
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

    # ── Composite Scoring ─────────────────────────────────────────────────

    def compute_composite(
        self,
        rule_discovery_score: TurnScore,
        prediction_score: TurnScore,
        calibration_score: TurnScore,
        belief_update_score: TurnScore,
        hallucination_score: TurnScore,
        tool_score: TurnScore,
    ) -> BenchmarkScore:
        """Combine all sub-scores into a final BenchmarkScore."""
        result = BenchmarkScore(
            rule_discovery=rule_discovery_score.raw_score,
            prediction_accuracy=prediction_score.raw_score,
            calibration_brier=calibration_score.raw_score,
            belief_update=belief_update_score.raw_score,
            hallucination_rate=hallucination_score.raw_score,
            tool_efficiency=tool_score.raw_score,
            turn_scores=[
                rule_discovery_score,
                prediction_score,
                calibration_score,
                belief_update_score,
                hallucination_score,
                tool_score,
            ],
        )
        return result
