# L3B Architecture Record

Implementation: `src/student_agent/workflow.py` (pure Python async state machine, no LLM, no
framework). All decisions are deterministic functions of MCP evidence + case input.

## 1. System overview

```text
                          ┌──────────────────────────┐
  case_received ────────▶ │   Coordinator / Router   │  (solve_case)
                          └─────────────┬────────────┘
                     task_assigned      │
                                        ▼
                          ┌──────────────────────────┐
                          │ Entity / Customer Agent  │  get_customer_history (+get_order fallback)
                          └─────────────┬────────────┘
                     handoff(entity_*)  │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼ (asyncio.gather)             ▼
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│ Order/Item Agent │ ────────▶ │  Payment Agent   │           │  Shipment Agent  │
└────────┬─────────┘           └────────┬─────────┘           └────────┬─────────┘
         └──────────────────────────────┼──────────────────────────────┘
                                        │ handoff + tool_result_consumed (MCP evidence)
                                        ▼
                               ┌──────────────────┐
                               │   Policy Agent   │  get_policy → policy_decided
                               └────────┬─────────┘
                                        ▼ handoff(policy_decision_ready)
                               ┌──────────────────┐
                               │  Verifier Agent  │  verification_completed
                               └────────┬─────────┘
                                        ▼ handoff(validated_output)
                                   [END OUTPUT]  → cli validates schema → case_finalized
```

Order agent runs first because shipment and payment need its items (seller shipping limits,
item totals). Shipment and payment agents then run concurrently.

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| Entity/customer (`entity-agent`) | claimed order, candidates, customer hint | Resolve order within customer scope, reject candidates, related orders | `get_customer_history`, `get_order` (only if history missing) | `EntityResult` → coordinator |
| Coordinator (`coordinator`) | case | Route tasks, gate specialists on `entity.status == resolved` | none | `task_assigned` to each agent |
| Order/product (`order-agent`) | resolved order id | Order status, timestamps, items, sellers, totals | `get_order`, `get_order_items`, `get_product_context` (only if scope requests) | `OrderResult` → coordinator |
| Shipment (`shipment-agent`) | `OrderResult` | Delivery vs estimate, seller handoff vs `shipping_limit_date`, lost/returned | `get_shipment_summary` | `ShipmentResult` → coordinator |
| Payment/refund (`payment-agent`) | `OrderResult` | Captured/refunded/refundable totals, duplicate capture, mismatch, refund state | `get_payment_timeline` (fallback `get_order_payments`), `get_refund_timeline` | `PaymentResult` → coordinator |
| Policy (`policy-agent`) | all specialist results + claims | Primary/secondary issue, responsible party, refund lines, actions, relevant evidence | `get_policy` | `policy_decided` → verifier |
| Conflict resolver | (inside shipment/payment agents) | Emit `data_conflicts` with selected source | — | part of specialist results |
| Verifier (`verifier-agent`) | `Decision` + specialist results | Cross-field consistency, evidence ownership, refund cap, confidence | none | `verification_completed` → coordinator |

Least privilege is enforced by `TOOL_PERMISSIONS`; `CaseContext.fetch` raises `PermissionError`
if an actor calls a tool outside its allow-list.

## 3. Entity resolution và A2A protocol

- Candidates = `claimed_order_id` ∪ `candidate_order_ids` (deduplicated, claimed first).
- One `get_customer_history(customer_unique_id_hint)` call; candidates present in the
  customer's history are resolved; others (e.g. `candidate-NNN`) are `rejected_candidates`.
- If history is unavailable, only well-formed (32-hex) candidates are probed with `get_order`
  and accepted if the owner matches the hint.
- `status`: exactly one → `resolved` (0.95 with history, 0.75 otherwise); several →
  `ambiguous` (0.4); none → `not_found` (0.2). Specialists only run when `resolved`.
- `related_order_ids` = other orders in the customer history.
- Envelope: a per-case `CaseContext` (case, gateway, trace, cache). Every event is correlated by
  `case_id`; agents return typed dataclasses, never free text. The flow is a fixed DAG, so no
  loops are possible.

## 4. Evidence và conflict lifecycle

- `EvidenceGateway.call` validates every response against `mcp-evidence-response-v1`.
- `CaseContext.fetch` always passes `case_id=case["case_id"]`, stores the server-issued
  `evidence_ref` unchanged, and emits `tool_result_consumed` (actor, tool, ref, domain).
- The cache lives in the per-case context, so evidence is never reused across cases.
- Output `evidence_refs` contain only refs consumed in this case and relevant to the decision:
  entity + order/items + policy always; shipment for shipment issues; payment for non-shipment
  issues; refund timeline for refund/cancel/unavailable issues; product context only for
  `unavailable_order_paid`.
- Conflicts: order row vs shipment summary delivery date → select `get_shipment_summary`;
  payment rows vs lifecycle capture total → select lifecycle events.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| MCP timeout / connection | 1 | treat as missing evidence | none (recorded in `ctx.failures`, lowers confidence) |
| MCP tool error | 0 | treat as missing evidence | none |
| Entity not found/ambiguous | 0 | skip specialists → `insufficient_evidence`, `needs_investigation` | `handoff` `entity_not_found` / `entity_ambiguous` |
| Source conflict | 0 | policy source precedence | `data_conflicts[]` + lower confidence |
| Invalid specialist result | 0 | verifier drops inconsistent lines/parties | `verification_completed` `adjusted` |

Budget: ~8 calls per case (history, order, items, product, shipment, payment timeline, refund
timeline, policy). `get_order_payments` and `get_sellers` are not called because the timeline
and items already contain that data. Identical calls are cached per case.

## 6. Verification invariants

- Output is validated against `l3b-output-v2.schema.json` by the CLI before writing.
- Every output evidence ref was issued in this case (`evidence_owned`).
- `recommended_refund_brl` = Σ `refund_lines` ≤ `refundable_total_brl` (else lines dropped).
- `no_action` ⇒ no refund lines and no actions; `action_required` ⇒ refund or action.
- `late_delivery_seller` ⇒ only seller parties; `late_delivery_logistics` ⇒ no seller party.
- Confidence ∈ [0.05, 0.95]: base by claim support (0.9 supported, 0.7 unsupported, 0.65
  issue differs from claim, 0.35 insufficient), minus penalties for conflicts, warnings, MCP
  failures, incomplete timeline; capped at 0.4 when entity is not resolved.

## 7. Reproducibility

- Python ≥ 3.11, dependencies pinned by range in `pyproject.toml`; no model, no randomness in
  decisions (trace `event_id`s are random by design).
- Cases are processed sequentially by `day09 run`; within a case shipment + payment run
  concurrently.
- Commands: `day09 validate-inputs && day09 run && day09 validate && day09 package`.
- Tests: `pytest -q` (includes a fake-gateway test of the workflow). API keys only in `.env`.
