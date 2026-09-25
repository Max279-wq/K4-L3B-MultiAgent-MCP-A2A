from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

import pytest

from student_agent.cases import CaseSet
from student_agent.contracts import Contracts
from student_agent.submission import validate_artifacts
from student_agent.trace import TraceWriter
from student_agent.workflow import solve_case

ROOT = Path(__file__).resolve().parents[1]
ORDER_ID = "a" * 32
SELLER_ID = "s" * 32
CUSTOMER = "customer-test"
DOMAINS = {
    "get_customer_history": "customer",
    "get_order": "order",
    "get_order_items": "item",
    "get_product_context": "product",
    "get_shipment_summary": "shipment",
    "get_payment_timeline": "payment",
    "get_order_payments": "payment",
    "get_refund_timeline": "refund",
    "get_policy": "policy",
}


class FakeGateway:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.calls: list[tuple[str, str]] = []
        self.issued: set[str] = set()

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.calls.append((tool_name, case_id))
        if tool_name not in self.data:
            raise RuntimeError(f"MCP tool {tool_name} failed")
        payload = self.data[tool_name]
        ref = f"ev_{secrets.token_urlsafe(24)}"
        self.issued.add(ref)
        digest = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": ref,
            "result_hash": f"sha256:{digest}",
            "domain": DOMAINS[tool_name],
            "data": payload,
        }


def base_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "get_customer_history": {
            "customer_unique_id": CUSTOMER,
            "orders": [{"order_id": ORDER_ID}, {"order_id": "b" * 32}],
        },
        "get_order": {
            "order_id": ORDER_ID,
            "order_status": "delivered",
            "order_purchase_timestamp": "2018-01-01 10:00:00",
            "order_delivered_carrier_date": "2018-01-03 10:00:00",
            "order_delivered_customer_date": "2018-01-08 10:00:00",
            "order_estimated_delivery_date": "2018-01-10 00:00:00",
        },
        "get_order_items": {
            "items": [
                {
                    "order_id": ORDER_ID,
                    "order_item_id": 1,
                    "seller_id": SELLER_ID,
                    "product_id": "p1",
                    "shipping_limit_date": "2018-01-04 10:00:00",
                    "price": 90.0,
                    "freight_value": 10.0,
                }
            ]
        },
        "get_product_context": {"products": [{"product_id": "p1"}]},
        "get_shipment_summary": {"events": []},
        "get_payment_timeline": {
            "payments": [
                {"payment_sequential": 1, "payment_type": "credit_card", "payment_value": 100.0}
            ],
            "events": [{"event_type": "captured", "amount": 100.0, "payment_sequential": 1}],
        },
        "get_refund_timeline": {"events": []},
        "get_policy": {"policy_version": "EC_POLICY_V2"},
    }
    data.update(overrides)
    return data


def make_case(topic: str) -> dict[str, Any]:
    return {
        "case_id": "L3B_CASE_001",
        "customer_request": {
            "claimed_order_id": ORDER_ID,
            "claims": [
                {"claim_id": "claim-a", "topic": topic},
                {"claim_id": "claim-b", "topic": "requested_full_refund"},
            ],
        },
        "policy_version": "EC_POLICY_V2",
        "candidate_order_ids": [ORDER_ID, "candidate-001"],
        "investigation_scope": {"include_product_context": True},
        "customer_unique_id_hint": CUSTOMER,
    }


def run(tmp_path: Path, topic: str, data: dict[str, Any]) -> tuple[dict, list[dict], FakeGateway]:
    contracts = Contracts(ROOT / "contracts" / "schemas")
    trace = TraceWriter(tmp_path / "trace.jsonl", contracts)
    gateway = FakeGateway(data)
    output = asyncio.run(solve_case(make_case(topic), gateway, trace))  # type: ignore[arg-type]
    contracts.validate_output(output, "output")
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert set(output["evidence_refs"]) <= gateway.issued
    assert all(case_id == "L3B_CASE_001" for _, case_id in gateway.calls)
    kinds = {e["event_type"] for e in events}
    assert {"task_assigned", "handoff", "policy_decided", "verification_completed"} <= kinds
    if gateway.issued:
        assert "tool_result_consumed" in kinds
    return output, events, gateway


def test_late_delivery_seller(tmp_path: Path) -> None:
    data = base_data()
    data["get_order"] = {
        **data["get_order"],
        "order_delivered_carrier_date": "2018-01-06 10:00:00",
        "order_delivered_customer_date": "2018-01-15 10:00:00",
    }
    output, _, gateway = run(tmp_path, "late_delivery_seller", data)
    assert output["assessment"]["primary_issue"] == "late_delivery_seller"
    assert output["shipment_analysis"]["late_seller_ids"] == [SELLER_ID]
    assert output["root_cause_analysis"]["responsible_parties"][0]["party_type"] == "seller"
    assert output["entity_resolution"]["rejected_candidates"] == ["candidate-001"]
    tools = [tool for tool, _ in gateway.calls]
    assert len(tools) == len(set(tools))


def test_duplicate_charge(tmp_path: Path) -> None:
    data = base_data()
    data["get_payment_timeline"] = {
        **data["get_payment_timeline"],
        "events": [
            {"event_type": "captured", "amount": 100.0, "payment_sequential": 1},
            {"event_type": "captured", "amount": 100.0, "payment_sequential": 1},
        ],
    }
    output, _, _ = run(tmp_path, "duplicate_charge", data)
    assert output["assessment"]["primary_issue"] == "duplicate_charge"
    assert output["financial_resolution"]["recommended_refund_brl"] == 100.0


def test_canceled_order_paid(tmp_path: Path) -> None:
    data = base_data()
    data["get_order"] = {**data["get_order"], "order_status": "canceled"}
    output, _, _ = run(tmp_path, "canceled_order_paid", data)
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"
    assert output["financial_resolution"]["recommended_refund_brl"] == 100.0
    assert output["assessment"]["case_status"] == "action_required"


@pytest.mark.parametrize("topic", ["late_delivery_logistics", "refund_failed"])
def test_unsupported_claim_is_no_action(tmp_path: Path, topic: str) -> None:
    output, _, _ = run(tmp_path, topic, base_data())
    assert output["assessment"]["primary_issue"] == "unsupported_claim"
    assert output["assessment"]["case_status"] == "no_action"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0


def test_gateway_failure_stops_before_unscorable_output(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="entity resolution has no MCP evidence"):
        run(tmp_path, "payment_mismatch", {})


def test_submission_rejects_output_without_evidence(tmp_path: Path) -> None:
    output, _, _ = run(tmp_path, "late_delivery_seller", base_data())
    output["evidence_refs"] = []
    output_path = tmp_path / "outputs" / "L3B_CASE_001.json"
    output_path.parent.mkdir()
    output_path.write_text(json.dumps(output), encoding="utf-8")
    trace_path = tmp_path / "traces" / "trace.jsonl"
    trace_path.parent.mkdir()
    trace_path.write_bytes((tmp_path / "trace.jsonl").read_bytes())
    case_set = CaseSet("test-v1", "l3b", ("L3B_CASE_001",), {})

    with pytest.raises(ValueError, match="has no MCP evidence refs"):
        validate_artifacts(tmp_path, case_set, Contracts(ROOT / "contracts" / "schemas"))


def layered_data() -> dict[str, Any]:
    """Two history rows share one order id; only the older one matches the claim."""
    recent = {
        "order_id": ORDER_ID,
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-05-11T09:00:00-03:00",
        "order_delivered_carrier_date": "2018-05-13T09:00:00-03:00",
        "order_delivered_customer_date": "2018-05-20T09:00:00-03:00",
        "order_estimated_delivery_date": "2018-05-21T09:00:00-03:00",
    }
    older = {
        "order_id": ORDER_ID,
        "order_status": "delivered",
        "order_purchase_timestamp": "2017-12-20T09:00:00-03:00",
        "order_delivered_carrier_date": "2017-12-22T09:00:00-03:00",
        "order_delivered_customer_date": "2018-01-04T09:00:00-03:00",
        "order_estimated_delivery_date": "2017-12-30T09:00:00-03:00",
    }
    item = {
        "order_id": ORDER_ID,
        "order_item_id": "item-1",
        "seller_id": SELLER_ID,
        "price": "79.00",
    }
    return base_data(
        get_customer_history={"customer_unique_id": CUSTOMER, "orders": [recent, older]},
        get_order=recent,
        get_order_items=[
            {**item, "shipping_limit_date": "2018-05-14T09:00:00-03:00", "freight_value": "10.00"},
            {**item, "shipping_limit_date": "2017-12-23T09:00:00-03:00", "freight_value": "18.00"},
        ],
        get_shipment_summary={
            "delivered_carrier_at": recent["order_delivered_carrier_date"],
            "delivered_customer_at": recent["order_delivered_customer_date"],
            "estimated_delivery_at": recent["order_estimated_delivery_date"],
            "events": [
                {
                    "event_at": "2018-01-04T09:00:00-03:00",
                    "event_type": "delivered_late",
                    "actor": "logistics_provider",
                    "status": "confirmed",
                }
            ],
        },
        get_payment_timeline={
            "payments": [
                {
                    "payment_sequential": "1",
                    "payment_type": "credit_card",
                    "payment_value": "89.00",
                },
                {
                    "payment_sequential": "1",
                    "payment_type": "credit_card",
                    "payment_value": "16.00",
                },
            ],
            "events": [
                {
                    "event_at": "2018-05-11T10:00:00-03:00",
                    "event_type": "captured",
                    "amount_brl": "89.00",
                },
                {
                    "event_at": "2017-12-20T10:00:00-03:00",
                    "event_type": "captured",
                    "amount_brl": "16.00",
                },
            ],
        },
        get_policy={
            "policy_version": "EC_POLICY_V2",
            "rules": {
                "late_delivery_logistics": {
                    "case_status": "action_required",
                    "recommended_action": "refund_freight",
                    "refund_brl": 16.0,
                    "responsible_parties": [{"party_id": None, "party_type": "logistics_provider"}],
                }
            },
        },
    )


def test_layered_order_rows_follow_claimed_timeline(tmp_path: Path) -> None:
    output, _, gateway = run(tmp_path, "late_delivery_logistics", layered_data())
    assert output["assessment"]["primary_issue"] == "late_delivery_logistics"
    assert output["shipment_analysis"]["verdict"] == "logistics_delay"
    assert output["payment_analysis"]["verdict"] == "reconciled"
    assert output["payment_analysis"]["captured_total_brl"] == 16.0
    assert output["financial_resolution"]["recommended_refund_brl"] == 16.0
    assert output["resolution_actions"] == ["refund_freight"]
    assert output["data_conflicts"][0]["field"] == "order_purchase_timestamp"
    tools = {tool for tool, _ in gateway.calls}
    assert "get_product_context" not in tools
    assert "get_refund_timeline" not in tools


def test_every_case_analyzes_shipment(tmp_path: Path) -> None:
    output, _, gateway = run(tmp_path, "payment_mismatch", base_data())
    assert "get_shipment_summary" in {tool for tool, _ in gateway.calls}
    assert output["shipment_analysis"]["verdict"] != "insufficient_evidence"


def test_canceled_order_fetches_refund_timeline(tmp_path: Path) -> None:
    _, _, gateway = run(tmp_path, "canceled_order_paid", base_data())
    assert "get_refund_timeline" in {tool for tool, _ in gateway.calls}
