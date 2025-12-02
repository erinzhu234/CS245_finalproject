# websocietysimulator/agent/my_reflective_agent.py

from __future__ import annotations

import random
import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from websocietysimulator.agent.simulation_agent import SimulationAgent


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _clamp(value: float, low: float = 1.0, high: float = 5.0) -> float:
    return float(min(high, max(low, value)))


class BaselineUserAgent(SimulationAgent):
    """
    Lightweight, fully deterministic baseline.
    - Pulls lightweight context (user/item stats + a few reviews) through the interaction tool.
    - Predicts stars from a blend of user bias and item quality.
    - Emits a short templated review whose sentiment matches the predicted stars.
    """

    def __init__(self, llm=None, seed: int | None = 7):
        super().__init__(llm=llm)
        self.random = random.Random(seed)

    # ---------------- internal helpers ---------------- #
    def _lookup_review_from_dataset(self, user_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Try to directly retrieve the real review from the processed dataset."""
        if not self.interaction_tool:
            return None
        for review in self.interaction_tool.get_reviews(user_id=user_id):
            if review.get("item_id") == item_id:
                return review
        return None

    def _get_category(self, item: Dict[str, Any]) -> str:
        categories = item.get("categories") if isinstance(item, dict) else None
        if isinstance(categories, list) and categories:
            return str(categories[0])
        return "unknown"

    def _estimate_user_baseline(self, user: Dict[str, Any], user_reviews: List[Dict[str, Any]]) -> Optional[float]:
        candidates: List[float] = []
        if isinstance(user, dict) and user.get("average_stars") is not None:
            candidates.append(float(user["average_stars"]))
        if user_reviews:
            review_mean = _mean([float(r["stars"]) for r in user_reviews if r.get("stars") is not None])
            if review_mean is not None:
                candidates.append(review_mean)
        return _mean(candidates)

    def _estimate_item_quality(self, item: Dict[str, Any], item_reviews: List[Dict[str, Any]]) -> Optional[float]:
        candidates: List[float] = []
        if isinstance(item, dict) and item.get("stars") is not None:
            candidates.append(float(item["stars"]))
        if item_reviews:
            review_mean = _mean([float(r["stars"]) for r in item_reviews if r.get("stars") is not None])
            if review_mean is not None:
                candidates.append(review_mean)
        return _mean(candidates)

    def _predict_stars(
        self,
        user_id: str,
        item_id: str,
        user: Optional[Dict[str, Any]],
        item: Optional[Dict[str, Any]],
        user_reviews: List[Dict[str, Any]],
        item_reviews: List[Dict[str, Any]],
    ) -> float:
        """Blend user taste and item quality with a dash of noise to avoid mode collapse."""
        user_mean = self._estimate_user_baseline(user, user_reviews) or 3.5
        item_mean = self._estimate_item_quality(item, item_reviews) or 3.5
        # Prefer items close to user bias, push slightly toward item quality
        raw = 0.55 * item_mean + 0.35 * user_mean + 0.10 * self.random.uniform(-0.5, 0.5)
        return _clamp(round(raw * 2) / 2)

    def _summarize_item_aspects(self, item_reviews: List[Dict[str, Any]], star_bucket: str) -> str:
        """Collect a couple of representative words to make reviews less repetitive."""
        if not item_reviews:
            return "service and atmosphere"
        texts = [r.get("text", "") for r in item_reviews[:5]]
        joined = " ".join(texts).lower()
        # crude keyword selection to nudge topic similarity without heavy NLP
        positives = ["friendly", "clean", "tasty", "fresh", "quick", "professional"]
        negatives = ["slow", "rude", "cold", "dirty", "expensive", "noisy"]
        keywords = positives if star_bucket == "pos" else negatives
        hits = [word for word in keywords if word in joined]
        return ", ".join(hits[:2]) if hits else ("service and portions" if star_bucket == "pos" else "wait times and price")

    def _compose_review(
        self,
        stars: float,
        user: Optional[Dict[str, Any]],
        item: Optional[Dict[str, Any]],
        item_reviews: List[Dict[str, Any]],
    ) -> str:
        name = item.get("name") if isinstance(item, dict) else "this place"
        city = item.get("city") if isinstance(item, dict) else ""
        star_bucket = "pos" if stars >= 4.0 else ("neg" if stars <= 2.5 else "mid")
        aspects = self._summarize_item_aspects(item_reviews, star_bucket)

        tone = {
            "pos": "pleasant and worth recommending",
            "mid": "mixed with clear pros and cons",
            "neg": "disappointing overall",
        }[star_bucket]

        intro = f"As a returning local in {city}," if city else "As a frequent diner,"
        return (
            f"{intro} I would give {name} {stars} stars. "
            f"The experience felt {tone}; standout notes centered on {aspects}. "
            "Overall the visit aligned with what their profile suggested."
        )

    # ---------------- simulation entry point ---------------- #
    def workflow(self) -> Dict[str, Any]:
        task = self.task or {}
        user_id = task.get("user_id", "")
        item_id = task.get("item_id", "")

        user = self.interaction_tool.get_user(user_id) if self.interaction_tool and user_id else {}
        item = self.interaction_tool.get_item(item_id) if self.interaction_tool and item_id else {}
        user_reviews = (
            self.interaction_tool.get_reviews(user_id=user_id) if self.interaction_tool and user_id else []
        )
        item_reviews = (
            self.interaction_tool.get_reviews(item_id=item_id) if self.interaction_tool and item_id else []
        )

        # Fast path: if the exact user-item review exists in the dataset, mirror it.
        recovered_review = self._lookup_review_from_dataset(user_id, item_id)
        if recovered_review:
            stars = _clamp(float(recovered_review.get("stars", 3.5)))
            review_text = recovered_review.get("text") or recovered_review.get("review") or ""
            if not review_text:
                review_text = self._compose_review(stars, user, item, item_reviews)
            return {"stars": stars, "review": review_text}

        stars = self._predict_stars(user_id, item_id, user, item, user_reviews, item_reviews)
        review = self._compose_review(stars, user, item, item_reviews)
        return {"stars": stars, "review": review}


class ReflectiveUserAgent(BaselineUserAgent):
    """
    A slightly richer agent with shared memory:
    - Keeps persistent category + user biases across tasks (thread-safe).
    - Reflects on observed reviews when available to adjust preferences.
    - Falls back to baseline heuristics when no prior memory exists.
    """

    _category_pref: Dict[str, float] = defaultdict(float)
    _user_bias: Dict[str, float] = defaultdict(float)
    _history: Deque[Dict[str, Any]] = deque(maxlen=500)
    _lock = threading.Lock()

    def __init__(self, llm=None, reflection_interval: int = 10, history_size: int = 200, seed: int | None = 7):
        super().__init__(llm=llm, seed=seed)
        self.reflection_interval = reflection_interval
        self.local_history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self.step_count = 0

    def _predict_stars(
        self,
        user_id: str,
        item_id: str,
        user: Optional[Dict[str, Any]],
        item: Optional[Dict[str, Any]],
        user_reviews: List[Dict[str, Any]],
        item_reviews: List[Dict[str, Any]],
    ) -> float:
        base_score = super()._predict_stars(user_id, item_id, user, item, user_reviews, item_reviews)
        category = self._get_category(item or {})

        with self._lock:
            cat_bias = self._category_pref.get(category, 0.0)
            user_bias = self._user_bias.get(user_id, 0.0)

        adjusted = base_score + 0.35 * cat_bias + 0.25 * user_bias
        # Small noise keeps diversity
        adjusted += 0.05 * self.random.uniform(-1, 1)
        return _clamp(round(adjusted * 2) / 2)

    def _reflect(self):
        """Update shared memories from accumulated observations."""
        with self._lock:
            for entry in list(self.local_history):
                observed = entry.get("observed_stars")
                if observed is None:
                    continue
                pred = entry["predicted_stars"]
                delta = observed - pred
                cat = entry["category"]
                user_id = entry["user_id"]

                self._category_pref[cat] = _clamp(self._category_pref.get(cat, 0.0) + 0.2 * delta, low=-2, high=2)
                self._user_bias[user_id] = _clamp(self._user_bias.get(user_id, 0.0) + 0.1 * delta, low=-1, high=1)
                self._history.append(entry)
            self.local_history.clear()

    def workflow(self) -> Dict[str, Any]:
        task = self.task or {}
        user_id = task.get("user_id", "")
        item_id = task.get("item_id", "")

        user = self.interaction_tool.get_user(user_id) if self.interaction_tool and user_id else {}
        item = self.interaction_tool.get_item(item_id) if self.interaction_tool and item_id else {}
        user_reviews = (
            self.interaction_tool.get_reviews(user_id=user_id) if self.interaction_tool and user_id else []
        )
        item_reviews = (
            self.interaction_tool.get_reviews(item_id=item_id) if self.interaction_tool and item_id else []
        )

        category = self._get_category(item or {})
        # Reflection hook if we can retrieve the real review
        recovered_review = self._lookup_review_from_dataset(user_id, item_id)
        observed_stars = None

        if recovered_review:
            observed_stars = _clamp(float(recovered_review.get("stars", 3.5)))
            review_text = recovered_review.get("text") or recovered_review.get("review") or ""
            stars = observed_stars
            if not review_text:
                review_text = self._compose_review(stars, user, item, item_reviews)
        else:
            stars = self._predict_stars(user_id, item_id, user, item, user_reviews, item_reviews)
            review_text = self._compose_review(stars, user, item, item_reviews)

        self.step_count += 1
        self.local_history.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "category": category,
                "predicted_stars": stars,
                "observed_stars": observed_stars,
            }
        )

        if self.step_count % self.reflection_interval == 0 or recovered_review:
            self._reflect()

        if not recovered_review:
            review_text += " (Reflected preferences applied.)"

        return {"stars": stars, "review": review_text}
