"""L3B multi-agent workflow: coordinator + specialist agents over the MCP Evidence Gateway.

Agents are plain async functions sharing a per-case ``CaseContext``. Every MCP result is
fetched through ``CaseContext.fetch`` which enforces case scope, caches per case, bounds
retries and emits ``tool_result_consumed`` with the server-issued ``evidence_ref``.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

from . import OUTPUT_SCHEMA_VERSION
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

COORDINATOR = "coordinator"
ENTITY_AGENT = "entity-agent"
ORDER_AGENT = "order-agent"
SHIPMENT_AGENT = "shipment-agent"
PAYMENT_AGENT = "payment-agent"
POLICY_AGENT = "policy-agent"
VERIFIER = "verifier-agent"

TOOL_PERMISSIONS: dict[str, frozenset[str]] = {
    ENTITY_AGENT: frozenset({"get_customer_history", "get_order"}),
    ORDER_AGENT: frozenset({"get_order", "get_order_items", "get_product_context"}),
    SHIPMENT_AGENT: frozenset({"get_shipment_summary"}),
    PAYMENT_AGENT: frozenset({"get_payment_timeline", "get_order_payments", "get_refund_timeline"}),
    POLICY_AGENT: frozenset({"get_policy"}),
}

TRANSIENT_RETRIES = 1
MONEY_TOLERANCE = 0.01
ORDER_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

PAYMENT_ISSUES = {
    "payment_mismatch",
    "duplicate_charge",
    "refund_pending",
    "refund_failed",
    "valid_split_payment",
}
SHIPMENT_ISSUES = {"late_delivery_seller", "late_delivery_logistics"}
ORDER_ISSUES = {"canceled_order_paid", "unavailable_order_paid"}
KNOWN_ISSUES = PAYMENT_ISSUES | SHIPMENT_ISSUES | ORDER_ISSUES


# --------------------------------------------------------------------------------------
# Generic, schema-agnostic accessors for MCP ``data`` payloads
# --------------------------------------------------------------------------------------


def _walk(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _first(value: Any, *keys: str) -> Any:
    for node in _walk(value):
        for key in keys:
            if node.get(key) not in (None, ""):
                return node[key]
    return None


def _rows(value: Any, required_key: str) -> list[dict[str, Any]]:
    return [node for node in _walk(value) if required_key in node]


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _text(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _money(value: float) -> float:
    return round(max(value, 0.0) + 1e-9, 2)


def _unique(values: Iterable[Any]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value is None or value == "":
            continue
        text = str(value)[:128]
        if text not in seen:
            seen.append(text)
    return seen[:20]


def _event_status(row: dict[str, Any]) -> str:
    return _text(
        row.get("status")
        or row.get("event_type")
        or row.get("event")
        or row.get("lifecycle_status")
        or row.get("type")
    )


def _event_amount(row: dict[str, Any]) -> float | None:
    for key in ("amount_brl", "amount", "value", "payment_value", "refund_amount"):
        amount = _num(row.get(key))
        if amount is not None:
            return amount
    return None


# --------------------------------------------------------------------------------------
# Per-case A2A context
# --------------------------------------------------------------------------------------


@dataclass
class Evidence:
    tool_name: str
    evidence_ref: str
    domain: str
    data: Any
    warnings: list[str]


@dataclass
class CaseContext:
    case: dict[str, Any]
    gateway: EvidenceGateway
    trace: TraceWriter
    cache: dict[tuple[str, tuple[tuple[str, str], ...]], Evidence | None] = field(
        default_factory=dict
    )
    failures: list[str] = field(default_factory=list)

    @property
    def case_id(self) -> str:
        return str(self.case["case_id"])

    def emit(self, event_type: str, actor: str, **kwargs: Any) -> None:
        self.trace.emit(case_id=self.case_id, event_type=event_type, actor=actor, **kwargs)

    def assign(self, target: str, task: str) -> None:
        self.emit("task_assigned", COORDINATOR, target=target, decision_code=task)

    def handoff(self, actor: str, target: str, code: str, refs: list[str] | None = None) -> None:
        self.emit("handoff", actor, target=target, decision_code=code, evidence_refs=refs or None)

    async def fetch(self, actor: str, tool_name: str, **arguments: str) -> Evidence | None:
        if tool_name not in TOOL_PERMISSIONS.get(actor, frozenset()):
            raise PermissionError(f"{actor} is not allowed to call {tool_name}")
        key = (tool_name, tuple(sorted(arguments.items())))
        if key in self.cache:
            return self.cache[key]
        evidence: Evidence | None = None
        for attempt in range(TRANSIENT_RETRIES + 1):
            try:
                raw = await self.gateway.call(tool_name, case_id=self.case_id, **arguments)
            except (TimeoutError, ConnectionError, OSError) as exc:
                if attempt < TRANSIENT_RETRIES:
                    continue
                self.failures.append(f"{tool_name}:{type(exc).__name__}")
                break
            except (RuntimeError, ValueError) as exc:
                self.failures.append(f"{tool_name}:{type(exc).__name__}")
                break
            evidence = Evidence(
                tool_name=tool_name,
                evidence_ref=raw["evidence_ref"],
                domain=raw["domain"],
                data=raw["data"],
                warnings=list(raw.get("warnings") or []),
            )
            self.emit(
                "tool_result_consumed",
                actor,
                tool_name=tool_name,
                evidence_refs=[evidence.evidence_ref],
                attributes={"domain": evidence.domain, "warnings": len(evidence.warnings)},
            )
            break
        self.cache[key] = evidence
        return evidence


# --------------------------------------------------------------------------------------
# Order-row layering
# --------------------------------------------------------------------------------------


@dataclass
class Layers:
    """Assigns timestamped records to the order row they belong to.

    One ``order_id`` can carry several history rows (re-used ids). A record belongs to the
    row with the latest purchase timestamp at or before the record's own timestamp.
    """

    purchases: list[datetime]
    target: datetime | None
    deliveries: dict[datetime, datetime] = field(default_factory=dict)

    def contains_delivery_event(self, value: Any) -> bool:
        at = _time(value)
        owner = self.deliveries.get(at) if at else None
        return owner == self.target if owner else self.contains(value)

    def contains(self, value: Any) -> bool:
        if self.target is None or len(self.purchases) < 2:
            return True
        at = _time(value)
        if at is None:
            return True
        owners = [p for p in self.purchases if p <= at]
        return bool(owners) and max(owners) == self.target


def _order_row_dates(row: dict[str, Any]) -> dict[str, datetime | None]:
    return {
        "purchase": _time(row.get("order_purchase_timestamp")),
        "carrier": _time(row.get("order_delivered_carrier_date")),
        "delivered": _time(row.get("order_delivered_customer_date")),
        "estimated": _time(row.get("order_estimated_delivery_date")),
    }


# --------------------------------------------------------------------------------------
# Specialist results
# --------------------------------------------------------------------------------------


@dataclass
class EntityResult:
    status: str
    resolved_order_ids: list[str]
    rejected_candidates: list[str]
    confidence: float
    customer_unique_id: str | None
    related_order_ids: list[str]
    order_rows: list[dict[str, Any]]
    evidence: list[Evidence]


@dataclass
class OrderResult:
    order_id: str | None
    status: str
    purchase_at: datetime | None
    carrier_at: datetime | None
    delivered_at: datetime | None
    estimated_at: datetime | None
    layers: Layers
    items: list[dict[str, Any]]
    item_ids: list[str]
    seller_ids: list[str]
    items_total: float | None
    freight_total: float
    conflicts: list[dict[str, Any]]
    evidence: list[Evidence]
    product_evidence: Evidence | None
    replicated: bool = False


@dataclass
class ShipmentResult:
    verdict: str
    late_seller_ids: list[str]
    timeline_complete: bool
    shipment_ids: list[str]
    conflicts: list[dict[str, Any]]
    evidence: list[Evidence]


@dataclass
class PaymentResult:
    verdict: str
    captured_total: float | None
    refunded_total: float | None
    refundable_total: float | None
    pending_refund: float
    failed_refund: float
    duplicate_amount: float
    mismatch_amount: float
    split_payment: bool
    payment_references: list[str]
    conflicts: list[dict[str, Any]]
    evidence: list[Evidence]
    refund_evidence: list[Evidence]


@dataclass
class Decision:
    primary_issue: str
    secondary_issues: list[str]
    case_status: str
    responsible_parties: list[dict[str, str | None]]
    ranked_causes: list[str]
    refund_lines: list[dict[str, Any]]
    actions: list[str]
    evidence: list[Evidence]
    policy_evidence: Evidence | None


# --------------------------------------------------------------------------------------
# Investigation plan
# --------------------------------------------------------------------------------------

REFUND_ISSUES = {"refund_pending", "refund_failed"}


def _claimed_topics(case: dict[str, Any]) -> list[str]:
    claims = (case.get("customer_request") or {}).get("claims") or []
    return [str(c.get("topic")) for c in claims if isinstance(c, dict) and c.get("topic")]


def _issue_topics(case: dict[str, Any]) -> set[str]:
    topics = set(_claimed_topics(case))
    if not topics & (KNOWN_ISSUES | {"unsupported_claim"}):
        topics |= KNOWN_ISSUES
    return topics


def _needs_refunds(case: dict[str, Any]) -> bool:
    return bool(_issue_topics(case) & (REFUND_ISSUES | {"canceled_order_paid"}))


# --------------------------------------------------------------------------------------
# Entity / customer agent
# --------------------------------------------------------------------------------------


async def entity_agent(ctx: CaseContext) -> EntityResult:
    request = ctx.case.get("customer_request") or {}
    claimed = request.get("claimed_order_id")
    candidates = _unique([claimed, *(ctx.case.get("candidate_order_ids") or [])])
    hint = ctx.case.get("customer_unique_id_hint")
    evidence: list[Evidence] = []

    history_rows: list[dict[str, Any]] = []
    customer_unique_id: str | None = hint if isinstance(hint, str) else None
    if hint:
        history = await ctx.fetch(ENTITY_AGENT, "get_customer_history", customer_unique_id=hint)
        if history is not None:
            evidence.append(history)
            history_rows = _rows(history.data, "order_id")
            found = _first(history.data, "customer_unique_id")
            if isinstance(found, str):
                customer_unique_id = found
    history_ids = _unique(row.get("order_id") for row in history_rows)

    plausible = [c for c in candidates if ORDER_ID_PATTERN.fullmatch(c)]
    if history_ids:
        resolved = [c for c in candidates if c in history_ids]
    else:
        resolved = []
        for candidate in plausible:
            order = await ctx.fetch(ENTITY_AGENT, "get_order", order_id=candidate)
            if order is None:
                continue
            owner = _first(order.data, "customer_unique_id")
            if owner is None or hint is None or owner == hint:
                resolved.append(candidate)
                evidence.append(order)
    if len(resolved) > 1 and claimed in resolved:
        resolved = [claimed]

    if len(resolved) == 1:
        status = "resolved"
        confidence = 0.95 if history_ids else 0.75
    elif resolved:
        status = "ambiguous"
        confidence = 0.4
    else:
        status = "not_found"
        confidence = 0.2
    rejected = [c for c in candidates if c not in resolved]
    related = [oid for oid in history_ids if oid not in resolved]
    order_rows = [row for row in history_rows if row.get("order_id") in resolved]
    ctx.emit(
        "handoff",
        ENTITY_AGENT,
        target=COORDINATOR,
        decision_code=f"entity_{status}",
        evidence_refs=[e.evidence_ref for e in evidence] or None,
        attributes={"resolved": len(resolved), "rejected": len(rejected)},
    )
    return EntityResult(
        status, resolved, rejected, confidence, customer_unique_id, related, order_rows, evidence
    )


# --------------------------------------------------------------------------------------
# Order / item agent
# --------------------------------------------------------------------------------------


def _select_order_row(
    rows: list[dict[str, Any]], opened_at: datetime | None
) -> dict[str, Any] | None:
    dated = [(d, r) for r in rows if (d := _order_row_dates(r)["purchase"]) is not None]
    if not dated:
        return None
    eligible = [(d, r) for d, r in dated if opened_at is None or d <= opened_at]
    pool = eligible or dated
    return max(pool, key=lambda pair: pair[0])[1] if eligible else min(pool, key=lambda p: p[0])[1]


def _item_id(row: dict[str, Any], order_id: str) -> str | None:
    if row.get("item_id"):
        return str(row["item_id"])
    raw = row.get("order_item_id")
    if raw is None:
        return None
    text = str(raw)
    return text if not text.isdigit() else f"{order_id}:{text}"


def _dedupe(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    for row in rows:
        if row not in seen:
            seen.append(row)
    return seen


def _order_candidate(
    order_id: str,
    row: dict[str, Any],
    purchases: list[datetime],
    deliveries: dict[datetime, datetime],
    all_items: list[dict[str, Any]],
    order_row: dict[str, Any],
    evidence: list[Evidence],
    product: Evidence | None,
) -> OrderResult:
    dates = _order_row_dates(row)
    layers = Layers(purchases=purchases, target=dates["purchase"], deliveries=deliveries)
    conflicts: list[dict[str, Any]] = []
    order_purchase = _order_row_dates(order_row)["purchase"] if order_row else None
    if order_purchase and dates["purchase"] and order_purchase != dates["purchase"]:
        conflicts.append(
            {
                "field": "order_purchase_timestamp",
                "sources": ["get_order", "get_customer_history"],
                "selected_source": "get_customer_history",
                "resolution_code": "ORDER_ROW_MATCHING_CLAIM_TIMELINE",
            }
        )
    items = [r for r in all_items if layers.contains(r.get("shipping_limit_date"))] or all_items
    prices = [_num(r.get("price")) for r in items]
    freights = [_num(r.get("freight_value")) or 0.0 for r in items]
    items_total = (
        sum(p for p in prices if p is not None) + sum(freights)
        if items and all(p is not None for p in prices)
        else None
    )
    return OrderResult(
        order_id=order_id,
        status=_text(row.get("order_status") or row.get("status")),
        purchase_at=dates["purchase"],
        carrier_at=dates["carrier"],
        delivered_at=dates["delivered"],
        estimated_at=dates["estimated"],
        layers=layers,
        items=items,
        item_ids=_unique(_item_id(r, order_id) for r in items),
        seller_ids=_unique(r.get("seller_id") for r in items),
        items_total=items_total,
        freight_total=sum(freights),
        conflicts=conflicts,
        evidence=evidence,
        product_evidence=product,
    )


async def order_agent(ctx: CaseContext, order_id: str, entity: EntityResult) -> list[OrderResult]:
    """Return one candidate per distinct order row, preferred (case-opened) row first."""
    evidence: list[Evidence] = []
    order = await ctx.fetch(ORDER_AGENT, "get_order", order_id=order_id)
    items_ev = await ctx.fetch(ORDER_AGENT, "get_order_items", order_id=order_id)
    evidence.extend(e for e in (order, items_ev) if e is not None)
    if order is None and not entity.order_rows:
        ctx.handoff(ORDER_AGENT, COORDINATOR, "order_unknown")
        return []

    order_row = order.data if order and isinstance(order.data, dict) else {}
    rows = _dedupe([*entity.order_rows, *([order_row] if order_row else [])])
    by_purchase: dict[datetime | None, dict[str, Any]] = {}
    for row in rows:
        by_purchase.setdefault(_order_row_dates(row)["purchase"], row)
    preferred = _select_order_row(rows, _time(ctx.case.get("opened_at"))) or order_row
    ordered = [preferred, *[r for r in by_purchase.values() if r is not preferred]]
    ordered = [
        r
        for i, r in enumerate(ordered)
        if all(
            _order_row_dates(r)["purchase"] != _order_row_dates(o)["purchase"] for o in ordered[:i]
        )
    ]
    purchases = sorted(d for d in by_purchase if d is not None)
    deliveries = {
        dates["delivered"]: dates["purchase"]
        for dates in map(_order_row_dates, by_purchase.values())
        if dates["delivered"] and dates["purchase"]
    }

    statuses = {_text(r.get("order_status")) for r in ordered}
    product = None
    scope = ctx.case.get("investigation_scope") or {}
    if scope.get("include_product_context") and (
        "unavailable_order_paid" in _issue_topics(ctx.case) or "unavailable" in statuses
    ):
        product = await ctx.fetch(ORDER_AGENT, "get_product_context", order_id=order_id)

    all_items = _rows(items_ev.data, "seller_id") if items_ev else []
    all_items = _dedupe(r for r in all_items if r.get("order_id") in (None, order_id))
    candidates = [
        _order_candidate(
            order_id, row, purchases, deliveries, all_items, order_row, evidence, product
        )
        for row in ordered
    ]
    for candidate, row in zip(candidates, ordered, strict=True):
        candidate.replicated = entity.order_rows.count(row) > 1
    ctx.handoff(
        ORDER_AGENT,
        COORDINATOR,
        f"order_{candidates[0].status or 'unknown'}",
        [e.evidence_ref for e in (*evidence, *([product] if product else []))],
    )
    return candidates


# --------------------------------------------------------------------------------------
# Shipment agent
# --------------------------------------------------------------------------------------


async def shipment_agent(ctx: CaseContext, orders: list[OrderResult]) -> list[ShipmentResult]:
    order_id = orders[0].order_id
    assert order_id is not None
    summary = await ctx.fetch(SHIPMENT_AGENT, "get_shipment_summary", order_id=order_id)
    evidence = [summary] if summary else []
    results = [_analyze_shipment(summary.data if summary else {}, o, evidence) for o in orders]
    ctx.handoff(
        SHIPMENT_AGENT,
        COORDINATOR,
        f"shipment_{results[0].verdict}",
        [e.evidence_ref for e in evidence],
    )
    return results


def _analyze_shipment(data: Any, order: OrderResult, evidence: list[Evidence]) -> ShipmentResult:

    carrier, delivered, estimated = order.carrier_at, order.delivered_at, order.estimated_at
    if not order.layers.target or len(order.layers.purchases) < 2:
        carrier = carrier or _time(_first(data, "delivered_carrier_at"))
        delivered = delivered or _time(_first(data, "delivered_customer_at"))
        estimated = estimated or _time(_first(data, "estimated_delivery_at"))

    limits: dict[str, datetime] = {}
    limit_rows = [
        *[
            (r.get("seller_id"), r.get("shipping_limit_at"))
            for r in _rows(data, "shipping_limit_at")
        ],
        *[(r.get("seller_id"), r.get("shipping_limit_date")) for r in order.items],
    ]
    for seller, raw_limit in limit_rows:
        limit = _time(raw_limit)
        if limit and seller and order.layers.contains(raw_limit):
            limits[str(seller)] = min(limit, limits.get(str(seller), limit))

    events = [
        row
        for row in _rows(data, "event_type")
        if order.layers.contains_delivery_event(row.get("event_at"))
    ]
    kinds = {_text(row.get("event_type")) for row in events}
    late_actors = {
        _text(row.get("actor")) for row in events if "late" in _text(row.get("event_type"))
    }
    shipment_ids = _unique(row.get("shipment_id") for row in _rows(data, "shipment_id"))

    late_by_limit = [s for s, limit in limits.items() if carrier and carrier > limit]
    is_late = bool(delivered and estimated and delivered > estimated) or bool(late_actors)
    timeline_complete = bool(carrier and delivered and estimated)
    if any("lost" in k for k in kinds):
        verdict = "lost"
    elif any("return" in k for k in kinds):
        verdict = "returned"
    elif not is_late and (delivered is None or estimated is None):
        verdict = "insufficient_evidence"
    elif not is_late:
        verdict = "on_time"
    elif "seller" in late_actors or (late_by_limit and not late_actors):
        verdict = "seller_delay"
    else:
        verdict = "logistics_delay"
    late_sellers = (late_by_limit or order.seller_ids) if verdict == "seller_delay" else []
    result = ShipmentResult(
        verdict, _unique(late_sellers), timeline_complete, shipment_ids, [], evidence
    )
    return result


# --------------------------------------------------------------------------------------
# Payment / refund agent
# --------------------------------------------------------------------------------------


def _payment_reference(row: dict[str, Any], order_id: str) -> str | None:
    ref = row.get("payment_reference") or row.get("payment_id")
    if ref:
        return str(ref)
    if row.get("payment_sequential") is None:
        return None
    return f"{order_id}:{row['payment_sequential']}"


def _pairs_matching_items(
    pairs: list[dict[str, dict[str, Any]]], expected: float | None
) -> list[dict[str, dict[str, Any]]]:
    """Keep the capture subset that settles the items total when extra captures share a row."""
    amounts = [_event_amount(p["capture"]) or 0.0 for p in pairs]
    if expected is None or len(pairs) < 3 or abs(sum(amounts) - expected) <= MONEY_TOLERANCE:
        return pairs
    for size in range(len(pairs) - 1, 1, -1):
        for combo in combinations(range(len(pairs)), size):
            if abs(sum(amounts[i] for i in combo) - expected) <= MONEY_TOLERANCE:
                return [pairs[i] for i in combo]
    return pairs


async def payment_agent(ctx: CaseContext, orders: list[OrderResult]) -> list[PaymentResult]:
    order_id = orders[0].order_id
    assert order_id is not None
    timeline = await ctx.fetch(PAYMENT_AGENT, "get_payment_timeline", order_id=order_id)
    if timeline is None:
        timeline = await ctx.fetch(PAYMENT_AGENT, "get_order_payments", order_id=order_id)
    refunds = None
    if _needs_refunds(ctx.case):
        refunds = await ctx.fetch(PAYMENT_AGENT, "get_refund_timeline", order_id=order_id)
    results = [_analyze_payment(timeline, refunds, o) for o in orders]
    ctx.handoff(
        PAYMENT_AGENT,
        COORDINATOR,
        f"payment_{results[0].verdict}",
        [e.evidence_ref for e in (timeline, refunds) if e is not None],
    )
    return results


def _analyze_payment(
    timeline: Evidence | None, refunds: Evidence | None, order: OrderResult
) -> PaymentResult:
    assert order.order_id is not None
    layers = order.layers
    evidence = [timeline] if timeline else []
    refund_evidence = [refunds] if refunds else []

    data = timeline.data if timeline else {}
    base_rows = [
        r for r in _rows(data, "payment_value") if _num(r.get("payment_value")) is not None
    ]
    lifecycle = [
        r
        for r in _walk(data)
        if _event_amount(r) is not None and _event_status(r) and "payment_value" not in r
    ]
    all_captures = [
        r for r in lifecycle if "captur" in _text(r.get("event_type") or r.get("status"))
    ]
    if all_captures and len(all_captures) == len(base_rows):
        pairs = [
            {"row": b, "capture": c}
            for b, c in zip(base_rows, all_captures, strict=True)
            if layers.contains(c.get("event_at"))
        ]
        if order.replicated:
            pairs = _dedupe(pairs)
        pairs = _pairs_matching_items(pairs, order.items_total)
        base_rows = [pair["row"] for pair in pairs]
        captures = [pair["capture"] for pair in pairs]
    else:
        captures = [r for r in all_captures if layers.contains(r.get("event_at"))]
        if order.replicated:
            captures = _dedupe(captures)
    lifecycle = [r for r in lifecycle if layers.contains(r.get("event_at"))]

    base_total = sum(_num(r["payment_value"]) or 0.0 for r in base_rows) if base_rows else None
    captured = sum(_event_amount(r) or 0.0 for r in captures) if captures else base_total
    references = _unique(_payment_reference(r, order.order_id) for r in base_rows)

    mismatch_amount = sum(
        _event_amount(r) or 0.0 for r in lifecycle if "mismatch" in _text(r.get("event_type"))
    )
    amounts = [_event_amount(r) or 0.0 for r in captures]
    repeated = [
        a for i, a in enumerate(amounts) if any(abs(a - b) <= MONEY_TOLERANCE for b in amounts[:i])
    ]
    expected = order.items_total
    covers_items = (
        expected is not None
        and captured is not None
        and abs(captured - expected) <= max(MONEY_TOLERANCE, 0.005 * expected)
    )
    duplicate_amount = sum(repeated) if repeated and not covers_items else 0.0
    split = len(base_rows) > 1 and not duplicate_amount

    refunded = pending = failed = 0.0
    refund_rows = [
        r
        for r in _walk(refunds.data if refunds else {})
        if _event_status(r) and _event_amount(r) is not None and layers.contains(r.get("event_at"))
    ]
    latest: dict[Any, dict[str, Any]] = {}
    for index, row in enumerate(refund_rows):
        key = row.get("refund_id") or row.get("refund_reference") or index
        latest[key] = row
    for row in latest.values():
        status = _text(row.get("status") or row.get("event_type"))
        amount = _event_amount(row) or 0.0
        if "fail" in status or "reject" in status or "revers" in status:
            failed += amount
        elif "pend" in status or "request" in status or "process" in status or "initiat" in status:
            pending += amount
        elif "complet" in status or "succe" in status or "refunded" in status or "paid" in status:
            refunded += amount

    conflicts: list[dict[str, Any]] = []
    if mismatch_amount > MONEY_TOLERANCE:
        conflicts.append(
            {
                "field": "captured_total_brl",
                "sources": ["payment_rows", "payment_lifecycle"],
                "selected_source": "payment_lifecycle",
                "resolution_code": "RECONCILIATION_MISMATCH_OPEN",
            }
        )

    if captured is None and not refund_rows:
        verdict = "insufficient_evidence"
    elif duplicate_amount > MONEY_TOLERANCE:
        verdict = "duplicate_capture"
    elif failed > MONEY_TOLERANCE:
        verdict = "refund_failed"
    elif pending > MONEY_TOLERANCE:
        verdict = "refund_pending"
    elif mismatch_amount > MONEY_TOLERANCE:
        verdict = "capture_mismatch"
    elif refunded > MONEY_TOLERANCE:
        verdict = "refunded"
    else:
        verdict = "reconciled"

    result = PaymentResult(
        verdict=verdict,
        captured_total=None if captured is None else _money(captured),
        refunded_total=_money(refunded) if captured is not None or refunds else None,
        refundable_total=None if captured is None else _money(captured - refunded),
        pending_refund=_money(pending),
        failed_refund=_money(failed),
        duplicate_amount=_money(duplicate_amount),
        mismatch_amount=_money(mismatch_amount),
        split_payment=split,
        payment_references=references,
        conflicts=conflicts,
        evidence=evidence,
        refund_evidence=refund_evidence,
    )
    return result


# --------------------------------------------------------------------------------------
# Policy agent
# --------------------------------------------------------------------------------------

DEFAULT_RULES: dict[str, dict[str, Any]] = {
    "canceled_order_paid": {
        "party": "platform",
        "refund": "refundable",
        "actions": ["issue_refund"],
        "status": "action_required",
        "cause": "CANCELED_ORDER_CAPTURED",
    },
    "unavailable_order_paid": {
        "party": "seller",
        "refund": "refundable",
        "actions": ["issue_refund"],
        "status": "action_required",
        "cause": "ITEM_UNAVAILABLE_AFTER_PAYMENT",
    },
    "late_delivery_seller": {
        "party": "seller",
        "refund": "freight",
        "actions": ["refund_freight"],
        "status": "action_required",
        "cause": "SELLER_MISSED_SHIPPING_LIMIT",
    },
    "late_delivery_logistics": {
        "party": "logistics_provider",
        "refund": "freight",
        "actions": ["refund_freight"],
        "status": "action_required",
        "cause": "CARRIER_TRANSIT_DELAY",
    },
    "payment_mismatch": {
        "party": "payment_provider",
        "refund": "mismatch",
        "actions": ["reconcile_payment"],
        "status": "action_required",
        "cause": "CAPTURE_AMOUNT_MISMATCH",
    },
    "duplicate_charge": {
        "party": "payment_provider",
        "refund": "duplicate",
        "actions": ["refund_duplicate_charge"],
        "status": "action_required",
        "cause": "DUPLICATE_CAPTURE",
    },
    "refund_pending": {
        "party": "payment_provider",
        "refund": "none",
        "actions": ["monitor_refund"],
        "status": "needs_investigation",
        "cause": "REFUND_NOT_SETTLED",
    },
    "refund_failed": {
        "party": "payment_provider",
        "refund": "failed",
        "actions": ["retry_refund"],
        "status": "action_required",
        "cause": "REFUND_PROCESSING_FAILED",
    },
    "valid_split_payment": {
        "party": "customer",
        "refund": "none",
        "actions": ["document_no_action"],
        "status": "no_action",
        "cause": "VALID_SPLIT_PAYMENT",
    },
    "unsupported_claim": {
        "party": "customer",
        "refund": "none",
        "actions": ["document_no_action"],
        "status": "no_action",
        "cause": "CLAIM_NOT_SUPPORTED_BY_EVIDENCE",
    },
    "insufficient_evidence": {
        "party": None,
        "refund": "none",
        "actions": ["escalate_manual_review"],
        "status": "needs_investigation",
        "cause": "INSUFFICIENT_EVIDENCE",
    },
}


def _detected_issues(
    order: OrderResult, shipment: ShipmentResult | None, payment: PaymentResult | None
) -> list[str]:
    issues: list[str] = []
    paid = bool(payment and (payment.refundable_total or 0.0) > MONEY_TOLERANCE)
    if order.status == "canceled" and paid:
        issues.append("canceled_order_paid")
    if order.status == "unavailable" and paid:
        issues.append("unavailable_order_paid")
    if shipment and shipment.verdict == "seller_delay":
        issues.append("late_delivery_seller")
    if shipment and shipment.verdict == "logistics_delay":
        issues.append("late_delivery_logistics")
    if payment:
        verdict_issue = {
            "duplicate_capture": "duplicate_charge",
            "refund_failed": "refund_failed",
            "refund_pending": "refund_pending",
            "capture_mismatch": "payment_mismatch",
        }.get(payment.verdict)
        if verdict_issue:
            issues.append(verdict_issue)
        if payment.verdict == "reconciled" and payment.split_payment:
            issues.append("valid_split_payment")
    return issues


def _policy_rule(policy: Evidence | None, issue: str) -> dict[str, Any]:
    rule = dict(DEFAULT_RULES[issue])
    rules = policy.data.get("rules") if policy and isinstance(policy.data, dict) else None
    node = rules.get(issue) if isinstance(rules, dict) else None
    if not isinstance(node, dict):
        return rule
    if isinstance(node.get("case_status"), str):
        rule["status"] = node["case_status"]
    action = node.get("recommended_action")
    if isinstance(action, str):
        rule["actions"] = [action]
    amount = _num(node.get("refund_brl"))
    if amount is not None:
        rule["refund_amount"] = amount
    parties = node.get("responsible_parties")
    if isinstance(parties, list) and parties and isinstance(parties[0], dict):
        party = parties[0].get("party_type")
        if isinstance(party, str):
            rule["party"] = party
    return rule


def _refund_amount(rule: dict[str, Any], order: OrderResult, payment: PaymentResult) -> float:
    basis = rule["refund"]
    if basis == "freight":
        computed = order.freight_total
    else:
        computed = {
            "refundable": payment.refundable_total or 0.0,
            "duplicate": payment.duplicate_amount,
            "mismatch": payment.mismatch_amount,
            "failed": payment.failed_refund,
        }.get(basis, 0.0)
    amount = rule.get("refund_amount", computed)
    return min(amount, payment.refundable_total or 0.0)


async def policy_agent(
    ctx: CaseContext,
    entity: EntityResult,
    order: OrderResult | None,
    shipment: ShipmentResult | None,
    payment: PaymentResult | None,
) -> Decision:
    policy_version = ctx.case.get("policy_version")
    policy = None
    if policy_version:
        policy = await ctx.fetch(POLICY_AGENT, "get_policy", policy_version=str(policy_version))

    topics = [t for t in _claimed_topics(ctx.case) if t in KNOWN_ISSUES]
    if order is None or entity.status != "resolved":
        primary, detected = "insufficient_evidence", []
    else:
        detected = _detected_issues(order, shipment, payment)
        supported = [t for t in topics if t in detected]
        if supported:
            primary = supported[0]
        elif detected:
            primary = detected[0]
        elif payment is None or payment.verdict == "insufficient_evidence":
            primary = "insufficient_evidence"
        else:
            primary = "unsupported_claim"
    secondary = [i for i in detected if i != primary and i in topics]

    rule = _policy_rule(policy, primary)
    lines: list[dict[str, Any]] = []
    if order is not None and payment is not None and rule["refund"] != "none":
        amount = _refund_amount(rule, order, payment)
        if amount > MONEY_TOLERANCE:
            if rule["refund"] == "freight":
                entity_id = (order.item_ids or [order.order_id])[0]
            else:
                entity_id = (payment.payment_references or [order.order_id])[0]
            lines.append(
                {
                    "reason_code": f"{primary.upper()}_REFUND",
                    "amount_brl": _money(amount),
                    "entity_id": entity_id,
                }
            )

    party_type = rule["party"]
    parties: list[dict[str, str | None]] = []
    if party_type == "seller":
        sellers = (shipment.late_seller_ids if shipment else []) or (
            order.seller_ids if order else []
        )
        parties = [{"party_type": "seller", "party_id": s} for s in sellers[:5]] or [
            {"party_type": "seller", "party_id": None}
        ]
    elif party_type:
        parties = [{"party_type": party_type, "party_id": None}]

    status = rule["status"]
    if status == "action_required" and not lines and not rule["actions"]:
        status = "no_action"

    evidence: list[Evidence] = list(entity.evidence)
    if order:
        evidence.extend(order.evidence)
        if order.product_evidence and primary == "unavailable_order_paid":
            evidence.append(order.product_evidence)
    if shipment:
        evidence.extend(shipment.evidence)
    if payment:
        evidence.extend(payment.evidence)
        evidence.extend(payment.refund_evidence)
    if policy:
        evidence.append(policy)

    decision = Decision(
        primary_issue=primary,
        secondary_issues=secondary,
        case_status=status,
        responsible_parties=parties,
        ranked_causes=[rule["cause"]],
        refund_lines=lines,
        actions=list(dict.fromkeys(a[:80] for a in rule["actions"]))[:8],
        evidence=evidence,
        policy_evidence=policy,
    )
    ctx.emit(
        "policy_decided",
        POLICY_AGENT,
        decision_code=primary,
        evidence_refs=[e.evidence_ref for e in evidence][:20] or None,
        attributes={"case_status": status, "refund_lines": len(lines)},
    )
    ctx.handoff(POLICY_AGENT, VERIFIER, "policy_decision_ready")
    return decision


# --------------------------------------------------------------------------------------
# Verifier agent
# --------------------------------------------------------------------------------------


def verifier_agent(
    ctx: CaseContext,
    entity: EntityResult,
    order: OrderResult | None,
    shipment: ShipmentResult | None,
    payment: PaymentResult | None,
    decision: Decision,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    consumed = {e.evidence_ref for e in ctx.cache.values() if e is not None}
    refs = _unique(e.evidence_ref for e in decision.evidence if e.evidence_ref in consumed)
    checks["evidence_owned"] = len(refs) == len({e.evidence_ref for e in decision.evidence})

    lines = decision.refund_lines if decision.case_status == "action_required" else []
    cap = payment.refundable_total if payment and payment.refundable_total is not None else 0.0
    total = _money(sum(line["amount_brl"] for line in lines))
    checks["refund_within_refundable"] = total <= cap + MONEY_TOLERANCE
    if not checks["refund_within_refundable"]:
        lines, total = [], 0.0

    parties = decision.responsible_parties
    if decision.primary_issue == "late_delivery_seller":
        parties = [p for p in parties if p["party_type"] == "seller"] or parties
    if decision.primary_issue == "late_delivery_logistics":
        parties = [p for p in parties if p["party_type"] != "seller"]
    checks["responsibility_consistent"] = parties == decision.responsible_parties

    actions = decision.actions
    status = decision.case_status
    if status == "action_required" and not lines and not actions:
        status = "no_action"

    conflicts = [
        *(order.conflicts if order else []),
        *(shipment.conflicts if shipment else []),
        *(payment.conflicts if payment else []),
    ]
    warnings = sum(len(e.warnings) for e in decision.evidence)

    topics = _claimed_topics(ctx.case)
    if decision.primary_issue == "insufficient_evidence":
        confidence = 0.35
    elif decision.primary_issue in topics:
        confidence = 0.92
    elif decision.primary_issue == "unsupported_claim":
        confidence = 0.8
    else:
        confidence = 0.6
    confidence -= 0.03 * min(warnings, 3) + 0.1 * len(ctx.failures)
    if entity.status != "resolved":
        confidence = min(confidence, 0.4)
    if shipment and not shipment.timeline_complete and decision.primary_issue in SHIPMENT_ISSUES:
        confidence -= 0.1
    confidence = round(min(max(confidence, 0.05), 0.95), 2)

    ctx.emit(
        "verification_completed",
        VERIFIER,
        decision_code="verified" if all(checks.values()) else "adjusted",
        evidence_refs=refs[:20] or None,
        attributes={**checks, "confidence": confidence},
    )
    ctx.handoff(VERIFIER, COORDINATOR, "validated_output")

    claim_assessments = []
    for claim in (ctx.case.get("customer_request") or {}).get("claims") or []:
        topic = claim.get("topic")
        if decision.primary_issue == "insufficient_evidence":
            verdict = "insufficient_evidence"
        elif topic == "requested_full_refund":
            full = payment.refundable_total if payment else None
            verdict = (
                "supported"
                if full and total >= full - MONEY_TOLERANCE
                else "partially_supported"
                if total > 0
                else "unsupported"
            )
        elif topic == "unsupported_claim":
            verdict = (
                "supported" if decision.primary_issue == "unsupported_claim" else "unsupported"
            )
        elif topic == decision.primary_issue or topic in decision.secondary_issues:
            verdict = "supported"
        else:
            verdict = "unsupported"
        claim_assessments.append(
            {
                "claim_id": str(claim.get("claim_id"))[:64],
                "verdict": verdict,
                "confidence": confidence,
                "evidence_refs": refs[:30],
            }
        )

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": ctx.case_id,
        "assessment": {
            "primary_issue": decision.primary_issue,
            "secondary_issues": decision.secondary_issues[:10],
            "case_status": status,
            "confidence": confidence,
        },
        "affected_entities": {
            "order_ids": entity.resolved_order_ids,
            "item_ids": order.item_ids if order else [],
            "seller_ids": order.seller_ids if order else [],
            "payment_references": payment.payment_references if payment else [],
            "shipment_ids": shipment.shipment_ids if shipment else [],
        },
        "claim_assessments": claim_assessments[:5],
        "entity_resolution": {
            "status": entity.status,
            "resolved_order_ids": entity.resolved_order_ids,
            "rejected_candidates": entity.rejected_candidates,
            "confidence": entity.confidence,
        },
        "customer_context": {
            "customer_unique_id": entity.customer_unique_id,
            "related_order_ids": entity.related_order_ids,
        },
        "shipment_analysis": {
            "verdict": shipment.verdict if shipment else "insufficient_evidence",
            "late_seller_ids": shipment.late_seller_ids if shipment else [],
            "timeline_complete": shipment.timeline_complete if shipment else False,
        },
        "payment_analysis": {
            "verdict": payment.verdict if payment else "insufficient_evidence",
            "captured_total_brl": payment.captured_total if payment else None,
            "refunded_total_brl": payment.refunded_total if payment else None,
            "refundable_total_brl": payment.refundable_total if payment else None,
        },
        "root_cause_analysis": {
            "ranked_causes": [
                {"cause_code": code, "rank": rank}
                for rank, code in enumerate(decision.ranked_causes[:5], 1)
            ],
            "responsible_parties": parties[:5],
        },
        "evidence_refs": refs[:30],
        "data_conflicts": conflicts[:5],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": total,
            "refund_lines": lines[:10],
        },
        "resolution_actions": actions,
    }


# --------------------------------------------------------------------------------------
# Coordinator
# --------------------------------------------------------------------------------------


def _select_candidate(
    case: dict[str, Any],
    orders: list[OrderResult],
    shipments: list[ShipmentResult | None],
    payments: list[PaymentResult | None],
) -> int:
    """Pick the order row the claim refers to; fall back to the case-opened row."""
    topics = set(_claimed_topics(case))
    detected = [
        _detected_issues(o, s, p) for o, s, p in zip(orders, shipments, payments, strict=True)
    ]
    for index, issues in enumerate(detected):
        if topics & set(issues):
            return index
    if "unsupported_claim" in topics:
        for index, issues in enumerate(detected):
            if not issues:
                return index
    return 0


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    ctx = CaseContext(case=case, gateway=gateway, trace=trace)

    ctx.assign(ENTITY_AGENT, "resolve_entity")
    entity = await entity_agent(ctx)

    order: OrderResult | None = None
    shipment: ShipmentResult | None = None
    payment: PaymentResult | None = None
    if entity.status == "resolved":
        order_id = entity.resolved_order_ids[0]
        ctx.assign(ORDER_AGENT, "collect_order_items")
        orders = await order_agent(ctx, order_id, entity)
        if orders:
            ctx.assign(PAYMENT_AGENT, "analyze_payment_refund")
            ctx.assign(SHIPMENT_AGENT, "analyze_shipment")
            found, payments = await asyncio.gather(
                shipment_agent(ctx, orders), payment_agent(ctx, orders)
            )
            shipments: list[ShipmentResult | None] = list(found)
            index = _select_candidate(case, orders, shipments, list(payments))
            order, shipment, payment = orders[index], shipments[index], payments[index]

    ctx.assign(POLICY_AGENT, "decide_policy")
    decision = await policy_agent(ctx, entity, order, shipment, payment)
    if not entity.evidence and ctx.failures:
        failures = ", ".join(ctx.failures)
        raise RuntimeError(
            f"{ctx.case_id}: entity resolution has no MCP evidence; failed calls: {failures}"
        )
    ctx.assign(VERIFIER, "verify_output")
    return verifier_agent(ctx, entity, order, shipment, payment, decision)
