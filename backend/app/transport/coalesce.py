"""Per-tick event coalescing (PROTOCOL.md §3).

Only two event kinds collapse, and both for the same reason: within a 100ms
tick they carry no information the operator can act on separately. Everything
else — moves, task transitions, file changes, alerts — is a distinct decision
and is never merged.
"""

from __future__ import annotations

from app.protocol.events import ServerEvent


def coalesce(events: list[ServerEvent]) -> list[ServerEvent]:
    """Collapse a tick's events, preserving relative order.

    - `agent.status` for one agent keeps only the last; intermediate statuses
      inside a tick are not observable and must not be relied on.
    - `log.append` for one (agent, stream) concatenates into the first
      occurrence's position, so interleaving with other agents is preserved.
    """
    if len(events) < 2:
        return list(events)

    last_status_at: dict[str, int] = {}
    for i, event in enumerate(events):
        if event.type == "agent.status":
            last_status_at[event.data.agent_id] = i

    out: list[ServerEvent] = []
    log_slot: dict[tuple[str, str], int] = {}

    for i, event in enumerate(events):
        if event.type == "agent.status" and last_status_at[event.data.agent_id] != i:
            continue

        if event.type == "log.append":
            key = (event.data.agent_id, event.data.stream.value)
            slot = log_slot.get(key)
            if slot is not None:
                held = out[slot]
                assert held.type == "log.append"
                held.data.chunk += event.data.chunk
                continue
            # Copy before mutating: the caller's event objects are not ours.
            event = event.model_copy(deep=True)
            log_slot[key] = len(out)

        out.append(event)

    return out
