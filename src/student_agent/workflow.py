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
            except (TimeoutError, ConnectionError, OSError):
                if attempt < TRANSIENT_RETRIES:
                    continue
                self.failures.append(f"{tool_name}:timeout")
                break
            except (RuntimeError, ValueError):
                self.failures.append(f"{tool_name}:error")
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
    evidence: list[Evidence]


@dataclass
class OrderResult:
    order_id: str | None
    status: str
    purchase_at: datetime | None
    carrier_at: datetime | None
    delivered_at: datetime | None
    estimated_at: datetime | None
    items: list[dict[str, Any]]
    item_ids: list[str]
    seller_ids: list[str]
    items_total: float | None
    freight_total: float
    evidence: list[Evidence]


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
    over_capture: float
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
# Entity / customer agent
# --------------------------------------------------------------------------------------


async def entity_agent(ctx: CaseContext) -> EntityResult:
    request = ctx.case.get("customer_request") or {}
    claimed = request.get("claimed_order_id")
    candidates = _unique([claimed, *(ctx.case.get("candidate_order_ids") or [])])
    hint = ctx.case.get("customer_unique_id_hint")
    evidence: list[Evidence] = []

    history_ids: list[str] = []
    customer_unique_id: str | None = hint if isinstance(hint, str) else None
    if hint:
        history = await ctx.fetch(ENTITY_AGENT, "get_customer_history", customer_unique_id=hint)
        if history is not None:
            evidence.append(history)
            history_ids = _unique(row.get("order_id") for row in _rows(history.data, "order_id"))
            found = _first(history.data, "customer_unique_id")
            if isinstance(found, str):
                customer_unique_id = found

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
    ctx.emit(
        "handoff",
        ENTITY_AGENT,
        target=COORDINATOR,
        decision_code=f"entity_{status}",
        evidence_refs=[e.evidence_ref for e in evidence] or None,
        attributes={"resolved": len(resolved), "rejected": len(rejected)},
    )
    return EntityResult(
        status, resolved, rejected, confidence, customer_unique_id, related, evidence
    )


# --------------------------------------------------------------------------------------
# Order / item agent
# --------------------------------------------------------------------------------------


async def order_agent(ctx: CaseContext, order_id: str) -> OrderResult:
    evidence: list[Evidence] = []
    order = await ctx.fetch(ORDER_AGENT, "get_order", order_id=order_id)
    items_ev = await ctx.fetch(ORDER_AGENT, "get_order_items", order_id=order_id)
    scope = ctx.case.get("investigation_scope") or {}
    product = None
    if scope.get("include_product_context"):
        product = await ctx.fetch(ORDER_AGENT, "get_product_context", order_id=order_id)
    evidence.extend(e for e in (order, items_ev, product) if e is not None)

    data = order.data if order else {}
    items = _rows(items_ev.data, "seller_id") if items_ev else []
    items = [row for row in items if row.get("order_id") in (None, order_id)]
    item_ids = _unique(
        row.get("item_id")
        or (f"{order_id}:{row['order_item_id']}" if row.get("order_item_id") is not None else None)
        for row in items
    )
    seller_ids = _unique(row.get("seller_id") for row in items)
    prices = [_num(row.get("price")) for row in items]
    freights = [_num(row.get("freight_value")) or 0.0 for row in items]
    items_total = (
        sum(p for p in prices if p is not None) + sum(freights)
        if items and all(p is not None for p in prices)
        else None
    )
    result = OrderResult(
        order_id=order_id if order else None,
        status=_text(_first(data, "order_status", "status")),
        purchase_at=_time(_first(data, "order_purchase_timestamp", "purchase_timestamp")),
        carrier_at=_time(_first(data, "order_delivered_carrier_date", "delivered_carrier_date")),
        delivered_at=_time(
            _first(data, "order_delivered_customer_date", "delivered_customer_date")
        ),
        estimated_at=_time(
            _first(data, "order_estimated_delivery_date", "estimated_delivery_date")
        ),
        items=items,
        item_ids=item_ids,
        seller_ids=seller_ids,
        items_total=items_total,
        freight_total=sum(freights),
        evidence=evidence,
    )
    ctx.handoff(
        ORDER_AGENT,
        COORDINATOR,
        f"order_{result.status or 'unknown'}",
        [e.evidence_ref for e in evidence],
    )
    return result


# --------------------------------------------------------------------------------------
# Shipment agent
# --------------------------------------------------------------------------------------


async def shipment_agent(ctx: CaseContext, order: OrderResult) -> ShipmentResult:
    assert order.order_id is not None
    summary = await ctx.fetch(SHIPMENT_AGENT, "get_shipment_summary", order_id=order.order_id)
    evidence = [summary] if summary else []
    data = summary.data if summary else {}
    conflicts: list[dict[str, Any]] = []

    delivered = _time(
        _first(data, "delivered_customer_date", "order_delivered_customer_date", "delivered_at")
    )
    carrier = _time(
        _first(data, "delivered_carrier_date", "order_delivered_carrier_date", "carrier_handoff_at")
    )
    estimated = _time(
        _first(data, "estimated_delivery_date", "order_estimated_delivery_date", "estimated_at")
    )
    if delivered and order.delivered_at and abs((delivered - order.delivered_at).days) >= 1:
        conflicts.append(
            {
                "field": "delivered_customer_date",
                "sources": ["get_order", "get_shipment_summary"],
                "selected_source": "get_shipment_summary",
                "resolution_code": "SHIPMENT_EVENTS_AUTHORITATIVE",
            }
        )
    delivered = delivered or order.delivered_at
    carrier = carrier or order.carrier_at
    estimated = estimated or order.estimated_at

    limits: dict[str, datetime] = {}
    for row in [*_rows(data, "shipping_limit_date"), *order.items]:
        limit = _time(row.get("shipping_limit_date"))
        seller = row.get("seller_id")
        if limit and seller:
            limits[str(seller)] = min(limit, limits.get(str(seller), limit))
    events = {_event_status(row) for row in _rows(data, "status") + _rows(data, "event_type")}
    shipment_ids = _unique(row.get("shipment_id") for row in _rows(data, "shipment_id"))

    late_sellers = (
        [s for s, limit in limits.items() if carrier and carrier > limit] if carrier else []
    )
    timeline_complete = bool(order.purchase_at or summary) and bool(
        carrier and delivered and estimated
    )
    if any("lost" in e for e in events):
        verdict = "lost"
    elif any("return" in e for e in events):
        verdict = "returned"
    elif delivered is None or estimated is None:
        verdict = "insufficient_evidence"
    elif delivered.date() <= estimated.date():
        verdict = "on_time"
    elif late_sellers:
        verdict = "seller_delay"
    else:
        verdict = "logistics_delay"
    result = ShipmentResult(
        verdict, _unique(late_sellers), timeline_complete, shipment_ids, conflicts, evidence
    )
    ctx.handoff(
        SHIPMENT_AGENT, COORDINATOR, f"shipment_{verdict}", [e.evidence_ref for e in evidence]
    )
    return result


# --------------------------------------------------------------------------------------
# Payment / refund agent
# --------------------------------------------------------------------------------------


async def payment_agent(ctx: CaseContext, order: OrderResult) -> PaymentResult:
    assert order.order_id is not None
    timeline = await ctx.fetch(PAYMENT_AGENT, "get_payment_timeline", order_id=order.order_id)
    if timeline is None:
        timeline = await ctx.fetch(PAYMENT_AGENT, "get_order_payments", order_id=order.order_id)
    refunds = await ctx.fetch(PAYMENT_AGENT, "get_refund_timeline", order_id=order.order_id)
    evidence = [timeline] if timeline else []
    refund_evidence = [refunds] if refunds else []
    conflicts: list[dict[str, Any]] = []

    data = timeline.data if timeline else {}
    base_rows = [
        r for r in _rows(data, "payment_value") if _num(r.get("payment_value")) is not None
    ]
    base_total = sum(_num(r["payment_value"]) or 0.0 for r in base_rows) if base_rows else None
    references = _unique(
        r.get("payment_reference")
        or r.get("payment_id")
        or (
            f"{order.order_id}:{r['payment_sequential']}"
            if r.get("payment_sequential") is not None
            else None
        )
        for r in base_rows
    )

    lifecycle = [
        r
        for r in _walk(data)
        if _event_amount(r) is not None and _event_status(r) and "payment_value" not in r
    ]
    captures = [r for r in lifecycle if "captur" in _event_status(r)]
    captured = sum(_event_amount(r) or 0.0 for r in captures) if captures else base_total
    duplicate_amount = 0.0
    seen: set[tuple[Any, ...]] = set()
    for row in captures:
        key = (
            row.get("payment_reference") or row.get("payment_sequential"),
            _event_amount(row),
        )
        if key in seen:
            duplicate_amount += _event_amount(row) or 0.0
        seen.add(key)
    mismatch = base_total is not None and abs((captured or 0.0) - base_total) > MONEY_TOLERANCE
    if captures and mismatch:
        conflicts.append(
            {
                "field": "captured_total_brl",
                "sources": ["payment_rows", "payment_lifecycle"],
                "selected_source": "payment_lifecycle",
                "resolution_code": "LIFECYCLE_EVENTS_AUTHORITATIVE",
            }
        )

    refunded = pending = failed = 0.0
    refund_rows = [
        r
        for r in _walk(refunds.data if refunds else {})
        if _event_status(r) and _event_amount(r) is not None
    ]
    latest: dict[Any, dict[str, Any]] = {}
    for index, row in enumerate(refund_rows):
        key = row.get("refund_id") or row.get("refund_reference") or index
        latest[key] = row
    for row in latest.values():
        status = _event_status(row)
        amount = _event_amount(row) or 0.0
        if "fail" in status or "reject" in status or "revers" in status:
            failed += amount
        elif "pend" in status or "request" in status or "process" in status or "initiat" in status:
            pending += amount
        elif "complet" in status or "succe" in status or "refunded" in status or "paid" in status:
            refunded += amount

    expected = order.items_total
    over_capture = 0.0
    if captured is not None and expected is not None:
        over_capture = max(captured - duplicate_amount - expected, 0.0)
    split = len(base_rows) > 1 or len({_text(r.get("payment_type")) for r in base_rows}) > 1

    if captured is None and not refund_rows:
        verdict = "insufficient_evidence"
    elif duplicate_amount > MONEY_TOLERANCE:
        verdict = "duplicate_capture"
    elif failed > MONEY_TOLERANCE:
        verdict = "refund_failed"
    elif pending > MONEY_TOLERANCE:
        verdict = "refund_pending"
    elif (
        captured is not None
        and expected is not None
        and abs(captured - expected) > max(MONEY_TOLERANCE, 0.005 * expected)
    ):
        verdict = "capture_mismatch"
    elif refunded > MONEY_TOLERANCE:
        verdict = "refunded"
    else:
        verdict = "reconciled"

    refundable = None if captured is None else _money(captured - refunded)
    result = PaymentResult(
        verdict=verdict,
        captured_total=None if captured is None else _money(captured),
        refunded_total=_money(refunded) if refunds else (0.0 if captured is not None else None),
        refundable_total=refundable,
        pending_refund=_money(pending),
        failed_refund=_money(failed),
        duplicate_amount=_money(duplicate_amount),
        over_capture=_money(over_capture),
        split_payment=split,
        payment_references=references,
        conflicts=conflicts,
        evidence=evidence,
        refund_evidence=refund_evidence,
    )
    ctx.handoff(
        PAYMENT_AGENT,
        COORDINATOR,
        f"payment_{verdict}",
        [e.evidence_ref for e in (*evidence, *refund_evidence)],
    )
    return result


# --------------------------------------------------------------------------------------
# Policy agent
# --------------------------------------------------------------------------------------

DEFAULT_RULES: dict[str, dict[str, Any]] = {
    "canceled_order_paid": {
        "party": "platform",
        "refund": "refundable",
        "actions": ["refund_captured_payment", "confirm_order_cancellation"],
        "cause": "CANCELED_ORDER_CAPTURED",
    },
    "unavailable_order_paid": {
        "party": "seller",
        "refund": "refundable",
        "actions": ["refund_captured_payment", "flag_seller_inventory"],
        "cause": "ITEM_UNAVAILABLE_AFTER_PAYMENT",
    },
    "late_delivery_seller": {
        "party": "seller",
        "refund": "freight",
        "actions": ["refund_shipping_fee", "notify_seller_sla_breach"],
        "cause": "SELLER_MISSED_SHIPPING_LIMIT",
    },
    "late_delivery_logistics": {
        "party": "logistics_provider",
        "refund": "freight",
        "actions": ["refund_shipping_fee", "open_carrier_claim"],
        "cause": "CARRIER_TRANSIT_DELAY",
    },
    "payment_mismatch": {
        "party": "payment_provider",
        "refund": "over_capture",
        "actions": ["refund_overcharge", "reconcile_payment_capture"],
        "cause": "CAPTURE_AMOUNT_MISMATCH",
    },
    "duplicate_charge": {
        "party": "payment_provider",
        "refund": "duplicate",
        "actions": ["refund_duplicate_charge", "reconcile_payment_capture"],
        "cause": "DUPLICATE_CAPTURE",
    },
    "refund_pending": {
        "party": "payment_provider",
        "refund": "none",
        "actions": ["monitor_pending_refund", "notify_customer_refund_status"],
        "cause": "REFUND_NOT_SETTLED",
    },
    "refund_failed": {
        "party": "payment_provider",
        "refund": "failed",
        "actions": ["reissue_failed_refund", "notify_customer_refund_status"],
        "cause": "REFUND_PROCESSING_FAILED",
    },
    "valid_split_payment": {
        "party": None,
        "refund": "none",
        "actions": [],
        "cause": "VALID_SPLIT_PAYMENT",
    },
    "unsupported_claim": {
        "party": None,
        "refund": "none",
        "actions": [],
        "cause": "CLAIM_NOT_SUPPORTED_BY_EVIDENCE",
    },
    "insufficient_evidence": {
        "party": None,
        "refund": "none",
        "actions": ["escalate_manual_review"],
        "cause": "INSUFFICIENT_EVIDENCE",
    },
}


def _claimed_topics(case: dict[str, Any]) -> list[str]:
    claims = (case.get("customer_request") or {}).get("claims") or []
    return [str(c.get("topic")) for c in claims if isinstance(c, dict) and c.get("topic")]


def _detected_issues(
    order: OrderResult, shipment: ShipmentResult | None, payment: PaymentResult | None
) -> list[str]:
    issues: list[str] = []
    paid = bool(payment and (payment.refundable_total or 0.0) > MONEY_TOLERANCE)
    if order.status == "canceled" and paid:
        issues.append("canceled_order_paid")
    if order.status == "unavailable" and paid:
        issues.append("unavailable_order_paid")
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
    if shipment and shipment.verdict == "seller_delay":
        issues.append("late_delivery_seller")
    if shipment and shipment.verdict == "logistics_delay":
        issues.append("late_delivery_logistics")
    return issues


def _policy_rule(policy: Evidence | None, issue: str) -> dict[str, Any]:
    rule = dict(DEFAULT_RULES[issue])
    if policy is None:
        return rule
    for node in _walk(policy.data):
        if issue not in (node.get("issue"), node.get("primary_issue"), node.get("issue_code")):
            continue
        party = node.get("responsible_party") or node.get("party_type")
        if isinstance(party, str):
            rule["party"] = party
        actions = node.get("resolution_actions") or node.get("actions")
        if isinstance(actions, list) and all(isinstance(a, str) for a in actions):
            rule["actions"] = actions
        basis = node.get("refund_basis") or node.get("refund")
        if isinstance(basis, str):
            rule["refund"] = basis
        break
    return rule


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
        elif (payment and payment.verdict == "insufficient_evidence") and (
            shipment and shipment.verdict == "insufficient_evidence"
        ):
            primary = "insufficient_evidence"
        else:
            primary = "unsupported_claim"
    secondary = [i for i in detected if i != primary]

    rule = _policy_rule(policy, primary)
    lines: list[dict[str, Any]] = []
    basis = rule["refund"]
    if order is not None and payment is not None and basis != "none":
        amount_by_basis = {
            "refundable": payment.refundable_total or 0.0,
            "full": payment.refundable_total or 0.0,
            "duplicate": payment.duplicate_amount,
            "over_capture": payment.over_capture,
            "failed": payment.failed_refund,
        }
        if basis == "freight":
            late = set(shipment.late_seller_ids if shipment else [])
            for row in order.items:
                if primary == "late_delivery_seller" and late and row.get("seller_id") not in late:
                    continue
                freight = _num(row.get("freight_value")) or 0.0
                if freight > 0:
                    item_id = row.get("item_id") or (
                        f"{order.order_id}:{row['order_item_id']}"
                        if row.get("order_item_id") is not None
                        else order.order_id
                    )
                    lines.append(
                        {
                            "reason_code": "LATE_DELIVERY_FREIGHT_REFUND",
                            "amount_brl": _money(freight),
                            "entity_id": str(item_id)[:128],
                        }
                    )
        else:
            amount = amount_by_basis.get(basis, 0.0)
            if amount > MONEY_TOLERANCE:
                lines.append(
                    {
                        "reason_code": f"{primary.upper()}_REFUND",
                        "amount_brl": _money(amount),
                        "entity_id": (payment.payment_references or [order.order_id])[0],
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

    if primary == "insufficient_evidence":
        status = "needs_investigation"
    elif lines or rule["actions"]:
        status = "action_required"
    else:
        status = "no_action"

    evidence: list[Evidence] = list(entity.evidence)
    if order:
        evidence.extend(e for e in order.evidence if e.tool_name != "get_product_context")
        if primary == "unavailable_order_paid":
            evidence.extend(e for e in order.evidence if e.tool_name == "get_product_context")
    unsupported = primary == "unsupported_claim"
    if shipment and (primary in SHIPMENT_ISSUES or (unsupported and set(topics) & SHIPMENT_ISSUES)):
        evidence.extend(shipment.evidence)
    if payment and primary not in SHIPMENT_ISSUES:
        evidence.extend(payment.evidence)
    refund_topics = {"refund_pending", "refund_failed"}
    if payment and (
        primary in refund_topics | ORDER_ISSUES or (unsupported and set(topics) & refund_topics)
    ):
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
    if status == "no_action":
        actions = []

    conflicts = [*(shipment.conflicts if shipment else []), *(payment.conflicts if payment else [])]
    warnings = sum(len(e.warnings) for e in decision.evidence)

    topics = _claimed_topics(ctx.case)
    if decision.primary_issue in topics:
        confidence = 0.9
    elif decision.primary_issue == "unsupported_claim":
        confidence = 0.7
    elif decision.primary_issue == "insufficient_evidence":
        confidence = 0.35
    else:
        confidence = 0.65
    confidence -= 0.05 * len(conflicts) + 0.03 * min(warnings, 3) + 0.1 * len(ctx.failures)
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
        if topic == "requested_full_refund":
            full = payment.refundable_total if payment else None
            verdict = (
                "supported"
                if full and total >= full - MONEY_TOLERANCE
                else "partially_supported"
                if total > 0
                else "unsupported"
            )
        elif topic == decision.primary_issue or topic in decision.secondary_issues:
            verdict = "supported"
        elif decision.primary_issue == "insufficient_evidence":
            verdict = "insufficient_evidence"
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
        order = await order_agent(ctx, order_id)
        if order.order_id is not None:
            ctx.assign(SHIPMENT_AGENT, "analyze_shipment")
            ctx.assign(PAYMENT_AGENT, "analyze_payment_refund")
            shipment, payment = await asyncio.gather(
                shipment_agent(ctx, order), payment_agent(ctx, order)
            )
        else:
            order = None

    ctx.assign(POLICY_AGENT, "decide_policy")
    decision = await policy_agent(ctx, entity, order, shipment, payment)
    ctx.assign(VERIFIER, "verify_output")
    return verifier_agent(ctx, entity, order, shipment, payment, decision)
