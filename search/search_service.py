"""Typed orchestration of the multilingual ChedMed search pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from models.product import Product
from models.query_analysis import QueryAnalysis, SupportedLanguage
from models.search_query import SearchSource, StructuredSearchQuery
from search.category_resolver import CategoryResolver
from search.filter_engine import FilterEngine
from search.product_filter import ProductFilterService
from search.product_families import ProductFamilies
from search.product_type_resolver import ProductTypeResolver
from search.ranking_engine import RankingCandidate, RankingEngine
from search.relevance_gate import RelevanceGate
from search.semantic_query_builder import SemanticQueryBuilder

LOGGER = logging.getLogger(__name__)


class SearchServiceError(RuntimeError):
    """Raised when semantic retrieval cannot be completed safely."""


class ProductLookupPort(Protocol):
    def get_product(self, product_id: str) -> Product: ...


class QueryEmbeddingPort(Protocol):
    def embed_query(self, query: str) -> np.ndarray: ...


class ProductSearchPort(Protocol):
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...
    def count(self) -> int: ...


class QueryUnderstandingPort(Protocol):
    def understand(self, query: str) -> QueryAnalysis: ...


class QueryNormalizerPort(Protocol):
    def normalize(self, query: str) -> str: ...


class QueryExpansionPort(Protocol):
    def expand(
        self,
        query: str,
        product_type: str | None = None,
        category: str | None = None,
        include_category_aliases: bool = True,
    ) -> str: ...


class BasicQueryExpansionService:
    def expand(
        self,
        query: str,
        product_type: str | None = None,
        category: str | None = None,
        include_category_aliases: bool = True,
    ) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        result = " ".join(query.strip().split())
        if not result:
            raise ValueError("query cannot be empty.")
        return result


@dataclass(frozen=True, slots=True)
class SearchResultItem:
    product: Product
    score: float
    relevance_reason: str | None = None
    lexical_terms: tuple[str, ...] = ()
    match_type: str = "relevant"


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    items: tuple[SearchResultItem, ...]
    structured_query: StructuredSearchQuery | None = None
    faiss_candidates_count: int = 0
    filtered_products_count: int = 0
    relevant_products_count: int = 0
    primary_results_count: int = 0
    similar_results_count: int = 0
    match_type: str = "none"
    broad_similar_results_count: int = 0
    original_query: str | None = None
    normalized_query: str | None = None
    total_catalog_products: int = 0
    candidate_products_count: int = 0


class SearchService:
    """Resolve, retrieve, fetch, filter, and rank without crossing concerns."""

    _CANDIDATE_MULTIPLIER = 10
    _MIN_CANDIDATES = 50

    def __init__(
        self,
        product_lookup: ProductLookupPort,
        embedder: QueryEmbeddingPort,
        vector_store: ProductSearchPort,
        query_understanding: QueryUnderstandingPort,
        query_expansion: QueryExpansionPort,
        default_top_k: int,
        product_filter: ProductFilterService | None = None,
        *,
        category_resolver: CategoryResolver | None = None,
        product_type_resolver: ProductTypeResolver | None = None,
        filter_engine: FilterEngine | None = None,
        ranking_engine: RankingEngine | None = None,
        relevance_gate: RelevanceGate | None = None,
        semantic_query_builder: SemanticQueryBuilder | None = None,
        product_families: ProductFamilies | None = None,
        query_normalizer: QueryNormalizerPort | None = None,
        relevance_leader_margin: float = 0.015,
        relevance_max_relative_drop: float = 0.04,
        relevance_min_token_length: int = 4,
    ) -> None:
        if not isinstance(default_top_k, int) or isinstance(default_top_k, bool) or default_top_k <= 0:
            raise ValueError("default_top_k doit être un entier positif.")
        self._product_lookup = product_lookup
        self._embedder = embedder
        self._vector_store = vector_store
        self._query_understanding = query_understanding
        self._query_expansion = query_expansion
        self._category_resolver = category_resolver or CategoryResolver()
        self._product_type_resolver = product_type_resolver or ProductTypeResolver()
        self._filter_engine = filter_engine or FilterEngine(product_filter)
        self._ranking_engine = ranking_engine or RankingEngine(self._product_type_resolver)
        self._relevance_gate = relevance_gate or RelevanceGate(
            self._ranking_engine,
            self._product_type_resolver,
            leader_margin=relevance_leader_margin,
            max_relative_drop=relevance_max_relative_drop,
            min_token_length=relevance_min_token_length,
        )
        self._semantic_query_builder = semantic_query_builder or SemanticQueryBuilder(
            query_expansion
        )
        self._product_families = product_families or ProductFamilies()
        self._query_normalizer = query_normalizer
        self._default_top_k = default_top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        source: SearchSource | str = SearchSource.TEXT,
        include_all: bool = False,
    ) -> SearchResult:
        original_query = _normalise_query(query)
        normalized_query = original_query
        if self._query_normalizer is not None:
            normalized_query = _normalise_query(
                self._query_normalizer.normalize(original_query)
            )
        normalized_source = _normalise_source(source)
        limit = self._resolve_top_k(top_k)
        try:
            analysis = self._query_understanding.understand(normalized_query)
            if not isinstance(analysis, QueryAnalysis):
                raise TypeError("La compréhension doit retourner un QueryAnalysis.")
            structured = self._resolve(analysis, normalized_source)
            LOGGER.info(
                "Search query understood: original=%r language=%s "
                "source=%s category_hint=%r product_type_hint=%r category=%r product_type=%r",
                normalized_query,
                structured.language.value,
                structured.source.value,
                structured.category_hint,
                structured.product_type_hint,
                structured.category,
                structured.product_type,
            )
            focused = self._semantic_query_builder.build(structured)
            structured = replace(structured, semantic_query=focused.text)
            LOGGER.debug(
                "Focused semantic query: original_query=%r resolved_category=%r "
                "resolved_product_type=%r removed_constraints=%r semantic_query=%r",
                structured.original_query,
                structured.category,
                structured.product_type,
                focused.removed_constraints,
                structured.semantic_query,
            )
            embedding = self._embedder.embed_query(structured.semantic_query)
            LOGGER.info("Semantic query sent to embedder: %r", structured.semantic_query)
            # Relevance gating needs a comparison population even when no
            # structured filter is active. ``top_k`` is an output ceiling,
            # not the FAISS retrieval breadth.
            catalog_count = self._catalog_count()
            candidate_limit = catalog_count or max(
                (limit or self._default_top_k) * self._CANDIDATE_MULTIPLIER,
                self._MIN_CANDIDATES,
            )
            matches = self._vector_store.search(embedding, candidate_limit)
            LOGGER.info("FAISS returned %d candidate(s).", len(matches))
        except Exception as exc:
            LOGGER.exception("La recherche sémantique a échoué.")
            raise SearchServiceError("Impossible d'effectuer la recherche sémantique.") from exc

        accepted: list[RankingCandidate] = []
        retrieved: list[RankingCandidate] = []
        for faiss_rank, (product_id, score) in enumerate(matches, start=1):
            LOGGER.debug(
                "FAISS candidate: rank=%d id=%s score=%.6f",
                faiss_rank,
                product_id,
                score,
            )
            try:
                instance_full_lookup = getattr(
                    self._product_lookup, "get_product_any_status", None
                )
                full_lookup = (
                    instance_full_lookup
                    if include_all and callable(instance_full_lookup)
                    else getattr(self._product_lookup, "get_product_any_status")
                    if callable(
                        getattr(type(self._product_lookup), "get_product_any_status", None)
                    )
                    else None
                )
                # When the catalogue exposes an all-status lookup, use it for
                # both modes. Buyer availability is enforced just below. This
                # avoids treating indexed SOLD/PENDING products as missing and
                # keeps a single candidate population for diagnostics.
                product = (
                    full_lookup(product_id)
                    if callable(full_lookup)
                    else self._product_lookup.get_product(product_id)
                )
            except Exception:
                LOGGER.warning("Produit %s introuvable dans ChedMed.", product_id, exc_info=True)
                continue
            candidate = RankingCandidate(product, float(score))
            available = _buyer_available(product)
            if include_all or available:
                retrieved.append(candidate)
            if not include_all and not available:
                continue
            if include_all:
                accepted.append(candidate)
                continue
            matches_filters, rejection_reason = self._filter_engine.evaluate(product, structured)
            if not matches_filters:
                LOGGER.debug(
                    "Candidate rejected by strict filter: id=%s field=%s",
                    product.id,
                    rejection_reason,
                )
                continue
            accepted.append(candidate)

        ranked = self._ranking_engine.rank(accepted, structured)
        for ranking_rank, candidate in enumerate(ranked, start=1):
            breakdown = self._ranking_engine.score(candidate, structured)
            LOGGER.debug(
                "Ranking candidate: rank=%d id=%s semantic=%.6f "
                "product_type_bonus=%.3f category_bonus=%.3f total=%.6f",
                ranking_rank,
                candidate.product.id,
                breakdown.semantic,
                breakdown.product_type_bonus,
                breakdown.category_bonus,
                breakdown.total,
            )
        relevant, relevance_decisions = self._relevance_gate.apply(ranked, structured)
        for candidate, decision in zip(ranked, relevance_decisions, strict=True):
            LOGGER.debug(
                "Relevance gate: id=%s accepted=%s reason=%s "
                "ranking_score=%.6f relative_drop=%.6f lexical_terms=%s",
                candidate.product.id,
                decision.accepted,
                decision.reason,
                decision.ranking_score,
                decision.relative_drop,
                decision.lexical_terms,
            )
        if include_all:
            all_items = self._classify_full_catalog(
                ranked, relevance_decisions, structured
            )
            if limit is not None:
                all_items = all_items[:limit]
            return SearchResult(
                query=normalized_query,
                items=all_items,
                structured_query=structured,
                faiss_candidates_count=len(matches),
                filtered_products_count=len(accepted),
                relevant_products_count=sum(
                    item.match_type in {"exact", "relevant"} for item in all_items
                ),
                primary_results_count=sum(
                    item.match_type in {"exact", "relevant"} for item in all_items
                ),
                similar_results_count=sum(item.match_type == "similar" for item in all_items),
                broad_similar_results_count=sum(
                    item.match_type == "broad_similar" for item in all_items
                ),
                match_type=all_items[0].match_type if all_items else "none",
                original_query=original_query,
                normalized_query=normalized_query,
                total_catalog_products=self._total_catalog_count(catalog_count),
                candidate_products_count=len(matches),
            )
        output_limit = limit if limit is not None else len(relevant) + len(retrieved)
        final_candidates = relevant[:output_limit]
        accepted_decisions = [
            decision for decision in relevance_decisions if decision.accepted
        ][:output_limit]
        for candidate in final_candidates:
            breakdown = self._ranking_engine.score(candidate, structured)
            LOGGER.debug(
                "Ranking: id=%s semantic=%.6f product_type_bonus=%.3f "
                "category_bonus=%.3f total=%.6f",
                candidate.product.id,
                breakdown.semantic,
                breakdown.product_type_bonus,
                breakdown.category_bonus,
                breakdown.total,
            )
        primary_items = tuple(sorted((
            SearchResultItem(
                item.product,
                item.semantic_score,
                decision.reason,
                decision.lexical_terms,
                "exact" if _is_exact_match(item.product, structured, decision.lexical_terms) else "relevant",
            )
            for item, decision in zip(
                final_candidates, accepted_decisions, strict=True
            )
        ), key=lambda item: (item.match_type != "exact", -item.score)))
        similar_items = self._similar_fallback(
            retrieved,
            structured,
            primary_items,
            output_limit,
        )
        items = (primary_items + similar_items)[:limit]
        broad_count = sum(item.match_type == "broad_similar" for item in similar_items)
        close_count = len(similar_items) - broad_count
        match_type = (
            "relevant" if primary_items
            else "similar" if close_count
            else "broad_similar" if broad_count
            else "none"
        )
        LOGGER.info(
            "Search completed: fetched_candidates=%d filtered_products=%d "
            "relevant_products=%d primary_results=%d similar_results=%d "
            "match_type=%s final_results=%d",
            len(matches),
            len(accepted),
            len(relevant),
            len(primary_items),
            close_count,
            match_type,
            len(items),
        )
        return SearchResult(
            query=normalized_query,
            items=items,
            structured_query=structured,
            faiss_candidates_count=len(matches),
            filtered_products_count=len(accepted),
            relevant_products_count=len(relevant),
            primary_results_count=len(primary_items),
            similar_results_count=close_count,
            match_type=match_type,
            broad_similar_results_count=broad_count,
            original_query=original_query,
            normalized_query=normalized_query,
            total_catalog_products=self._total_catalog_count(catalog_count),
            candidate_products_count=len(matches),
        )

    def _classify_full_catalog(
        self,
        ranked: list[RankingCandidate],
        decisions: list[object],
        query: StructuredSearchQuery,
    ) -> tuple[SearchResultItem, ...]:
        family = self._product_families.for_query(query)
        items: list[SearchResultItem] = []
        for candidate, decision in zip(ranked, decisions, strict=True):
            accepted = bool(getattr(decision, "accepted", False))
            lexical_terms = tuple(getattr(decision, "lexical_terms", ()))
            reason = getattr(decision, "reason", None)
            if accepted and _is_exact_match(candidate.product, query, lexical_terms):
                match_type = "exact"
            elif family and self._product_families.broad_matches(candidate.product, family):
                # An accessory/adjacent-domain product remains broad even if
                # generic lexical evidence made the global gate accept it.
                match_type, reason = "broad_similar", f"product_domain:{family}"
            elif accepted:
                match_type = "relevant"
            elif family and self._product_families.matches(candidate.product, family):
                match_type, reason = "similar", f"product_family:{family}"
            else:
                match_type, reason = "unrelated", "full_catalog"
            items.append(SearchResultItem(
                candidate.product, candidate.semantic_score, reason, lexical_terms, match_type
            ))
        priority = {"exact": 0, "relevant": 1, "similar": 2, "broad_similar": 3, "unrelated": 4}
        return tuple(sorted(
            items,
            key=lambda item: (priority[item.match_type], -item.score),
        ))

    def _catalog_count(self) -> int:
        count = getattr(self._vector_store, "count", None)
        if callable(count):
            value = count()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return 0

    def _total_catalog_count(self, indexed_count: int) -> int:
        count = getattr(self._product_lookup, "count_all_products", None)
        if callable(count):
            value = count()
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return indexed_count

    def _similar_fallback(
        self,
        retrieved: list[RankingCandidate],
        query: StructuredSearchQuery,
        primary_items: tuple[SearchResultItem, ...],
        limit: int,
    ) -> tuple[SearchResultItem, ...]:
        """Complete with a bounded, protected cohort from the same FAISS run."""
        remaining = limit - len(primary_items)
        if remaining <= 0 or not retrieved:
            return ()
        relaxed_query = replace(
            query,
            # Brand remains strict for primary/relevant matches. In the
            # marketplace fallback it becomes a preference: an alternative
            # brand may be shown only when product-family/domain evidence
            # still proves that the candidate is related.
            brand=None,
            # A category inferred from the product concept (with no explicit
            # category hint from the request) is useful for primary search but
            # must not hide related accessories stored under imperfect
            # catalogue categories. Explicit audience/category remains strict.
            category=query.category if query.category_hint else None,
            color=None,
            city=None,
            condition=None,
            is_new=None,
            is_used=None,
        )
        relaxed = [
            candidate
            for candidate in retrieved
            if self._filter_engine.matches(candidate.product, relaxed_query)
        ]
        if not relaxed:
            return ()
        ranked = self._ranking_engine.rank(relaxed, relaxed_query)
        gated, decisions = self._relevance_gate.apply(ranked, relaxed_query)
        selected_ids = {item.product.id for item in primary_items}
        output: list[SearchResultItem] = []
        family = self._product_families.for_query(query)
        decision_by_id = {
            candidate.product.id: decision
            for candidate, decision in zip(ranked, decisions, strict=True)
        }
        for candidate in gated:
            if candidate.product.id in selected_ids:
                continue
            decision = decision_by_id[candidate.product.id]
            if family is not None:
                if self._product_families.broad_matches(candidate.product, family):
                    # Keep adjacent accessories for the broad phase even when
                    # their title contains the focal term (e.g. device + bag).
                    continue
                candidate_family = self._product_families.for_product(candidate.product)
                if candidate_family is not None and candidate_family != family:
                    continue
            # Attribute relaxation may expose a useful alternative, but a
            # merely isolated vector leader is not enough to call a product
            # "similar". This prevents arbitrary catalogue cohorts from
            # filling the marketplace response for unknown product terms.
            if decision.reason != "focal_lexical_evidence":
                continue
            output.append(
                SearchResultItem(
                    candidate.product,
                    candidate.semantic_score,
                    decision.reason,
                    decision.lexical_terms,
                    "similar",
                )
            )
            selected_ids.add(candidate.product.id)
            if len(output) >= remaining:
                return tuple(output)

        if family is not None:
            family_candidates = [
                candidate
                for candidate in ranked
                if candidate.product.id not in selected_ids
                and self._product_families.matches(candidate.product, family)
            ]
            LOGGER.debug(
                "Family fallback: family=%s candidates=%d limit=%d",
                family,
                len(family_candidates),
                remaining,
            )
            if family_candidates:
                for candidate in family_candidates:
                    output.append(SearchResultItem(
                        candidate.product,
                        candidate.semantic_score,
                        f"product_family:{family}",
                        (),
                        "similar",
                    ))
                    selected_ids.add(candidate.product.id)
                    if len(output) >= remaining:
                        break
        if family is not None and len(output) < remaining:
            broad_candidates = [
                candidate for candidate in ranked
                if candidate.product.id not in selected_ids
                and self._product_families.broad_matches(candidate.product, family)
            ]
            for candidate in broad_candidates:
                output.append(SearchResultItem(
                    candidate.product,
                    candidate.semantic_score,
                    f"product_domain:{family}",
                    (),
                    "broad_similar",
                ))
                selected_ids.add(candidate.product.id)
                if len(output) >= remaining:
                    break
        return tuple(output)

    def _resolve(
        self,
        analysis: QueryAnalysis,
        source: SearchSource,
    ) -> StructuredSearchQuery:
        filters = analysis.filters
        category = self._category_resolver.resolve_query(analysis.original_query)
        product_type = self._product_type_resolver.resolve_query(
            analysis.original_query
        )
        language = analysis.language
        if language is SupportedLanguage.UNKNOWN:
            language = _detect_language(analysis.original_query)
        return StructuredSearchQuery(
            original_query=analysis.original_query,
            # QueryExpansionService owns retrieval text enrichment. Provider
            # rewrites remain diagnostic output and cannot replace the user's
            # original wording or inject hallucinated product concepts.
            semantic_query=analysis.original_query,
            language=language,
            intent=analysis.intent,
            source=source,
            category=category,
            product_type=product_type,
            brand=filters.brand,
            color=filters.color,
            city=filters.city,
            condition=filters.condition,
            min_price=filters.min_price,
            max_price=filters.max_price,
            currency=filters.currency,
            is_new=filters.is_new,
            is_used=filters.is_used,
            is_sold=filters.is_sold,
            confidence=analysis.confidence,
            category_hint=filters.category_raw,
            product_type_hint=filters.product_type_raw,
        )

    def _resolve_top_k(self, top_k: int | None) -> int | None:
        if top_k is None:
            return None
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k doit être un entier positif.")
        return top_k


def _normalise_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("La requête de recherche doit être une chaîne.")
    normalized = " ".join(query.strip().split())
    if not normalized:
        raise ValueError("La requête de recherche ne peut pas être vide.")
    return normalized


def _normalise_source(source: SearchSource | str) -> SearchSource:
    if isinstance(source, SearchSource):
        return source
    try:
        return SearchSource(str(source).strip().lower())
    except ValueError as exc:
        raise ValueError("La provenance de recherche est invalide.") from exc


def _detect_language(query: str) -> SupportedLanguage:
    lowered = query.lower()
    if any(token in lowered for token in ("bghit", "sberdila", "dyal", " dial ")):
        return SupportedLanguage.DARIJA
    if any("\u0600" <= character <= "\u06ff" for character in query):
        return SupportedLanguage.DARIJA if "بغيت" in query else SupportedLanguage.ARABIC
    if any(token in lowered.split() for token in ("want", "men", "shoes")):
        return SupportedLanguage.ENGLISH
    if any(token in lowered.split() for token in ("je", "cherche", "chaussures", "homme")):
        return SupportedLanguage.FRENCH
    return SupportedLanguage.UNKNOWN


def _buyer_available(product: Product) -> bool:
    # Sold listings remain searchable when relevant (and visibly retain their
    # isSold/status fields); pending/rejected listings stay out of buyer mode.
    return product.status.upper() in {"ACCEPTED", "ACTIVE", "PENDING", "SOLD"}


def _is_exact_match(
    product: Product, query: StructuredSearchQuery, lexical_terms: tuple[str, ...]
) -> bool:
    if not lexical_terms:
        return False
    # Exactness is intentionally title/declared-brand focused. A query term
    # mentioned only in a long description (for example the device supported
    # by an accessory) is useful lexical evidence, but not an exact product.
    text = " ".join((product.title, product.brand or "")).casefold()
    if not all(term.casefold() in text for term in lexical_terms):
        return False
    # Requested brand/type hints are defining parts of exactness even when
    # the relevance gate exposes only one of them as focal lexical evidence.
    if query.brand is not None and query.brand.casefold() not in text:
        return False
    type_hint = query.product_type or query.product_type_hint
    if type_hint and type_hint.casefold() != (query.brand or "").casefold():
        return type_hint.casefold() in text
    return True
