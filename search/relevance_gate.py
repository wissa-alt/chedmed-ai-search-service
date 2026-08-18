"""Relative, explainable relevance gating after filtering and ranking."""

from __future__ import annotations

from dataclasses import dataclass
import re

from models.search_query import StructuredSearchQuery
from search.product_filter import _normalise_text
from search.product_type_resolver import ProductTypeResolver
from search.ranking_engine import RankingCandidate, RankingEngine


@dataclass(frozen=True, slots=True)
class RelevanceDecision:
    """Explain why one ranked candidate passed or failed the gate."""

    accepted: bool
    reason: str
    ranking_score: float
    relative_drop: float
    lexical_terms: tuple[str, ...]


class RelevanceGate:
    """Keep candidates supported by lexical evidence or relative prominence.

    No absolute FAISS threshold is used. A candidate passes when either:

    - it has focal lexical evidence and remains close to the best ranked score;
    - it is the best candidate and is separated from the runner-up by a
      configurable margin.

    Missing lexical evidence is therefore never sufficient on its own to
    reject a strongly isolated semantic leader.
    """

    def __init__(
        self,
        ranking_engine: RankingEngine,
        product_types: ProductTypeResolver | None = None,
        *,
        leader_margin: float = 0.015,
        max_relative_drop: float = 0.04,
        min_token_length: int = 4,
    ) -> None:
        if leader_margin < 0 or max_relative_drop < 0:
            raise ValueError("Les marges de pertinence doivent être positives ou nulles.")
        if min_token_length < 1:
            raise ValueError("min_token_length doit être positif.")
        self._ranking = ranking_engine
        self._product_types = product_types or ProductTypeResolver()
        self._leader_margin = float(leader_margin)
        self._max_relative_drop = float(max_relative_drop)
        self._min_token_length = min_token_length

    def apply(
        self,
        candidates: list[RankingCandidate],
        query: StructuredSearchQuery,
    ) -> tuple[list[RankingCandidate], tuple[RelevanceDecision, ...]]:
        """Return relevant candidates and an aligned decision audit trail."""
        if not candidates:
            return [], ()
        scores = [self._ranking.score(item, query).total for item in candidates]
        best_score = max(scores)
        ordered_ranking_scores = sorted(scores, reverse=True)
        leader_margin = (
            ordered_ranking_scores[0] - ordered_ranking_scores[1]
            if len(ordered_ranking_scores) > 1 else 0.0
        )
        original_leader = scores.index(best_score)
        focus_terms = self._focus_terms(query)
        accepted: list[RankingCandidate] = []
        decisions: list[RelevanceDecision] = []
        for index, (candidate, score) in enumerate(zip(candidates, scores, strict=True)):
            product_text = _normalise_text(
                f"{candidate.product.title} {candidate.product.description}"
            )
            lexical_terms = tuple(term for term in focus_terms if term in product_text)
            relative_drop = best_score - score
            lexical_evidence = bool(lexical_terms) and (
                relative_drop <= self._max_relative_drop
            )
            isolated_leader = index == original_leader and len(scores) > 1 and (
                leader_margin >= self._leader_margin
            )
            if lexical_evidence:
                reason = "focal_lexical_evidence"
            elif isolated_leader:
                reason = "isolated_semantic_leader"
            else:
                reason = "insufficient_relative_relevance"
            decision = RelevanceDecision(
                accepted=lexical_evidence or isolated_leader,
                reason=reason,
                ranking_score=score,
                relative_drop=relative_drop,
                lexical_terms=lexical_terms,
            )
            decisions.append(decision)
            if decision.accepted:
                accepted.append(candidate)
        return accepted, tuple(decisions)

    def _focus_terms(self, query: StructuredSearchQuery) -> tuple[str, ...]:
        """Extract product-concept terms, excluding audience/category words."""
        sources = [query.product_type_hint, query.original_query]
        if query.product_type:
            sources.extend(self._product_types.aliases_for(query.product_type))
        resolved_category_terms = (
            _tokens(query.category_hint or "", self._min_token_length)
            if query.category else ()
        )
        excluded = {
            *resolved_category_terms,
            *_tokens(query.category or "", self._min_token_length),
            "bghit", "pour", "dial", "dyal", "want", "cherche", "homme",
            "hommes", "femme", "femmes", "women", "woman", "men", "male",
            "رجال", "الرجال", "نساء", "للرجال", "للنساء",
        }
        terms: list[str] = []
        for source in sources:
            for token in _tokens(source or "", self._min_token_length):
                if token not in excluded and token not in terms:
                    terms.append(token)
        return tuple(terms)


def _tokens(value: str, min_length: int) -> tuple[str, ...]:
    normalized = _normalise_text(value)
    return tuple(
        token for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        if len(token) >= min_length
    )
