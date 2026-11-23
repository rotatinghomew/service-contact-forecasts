"""Service-level promises: caps, the compliant share, and what the record proves.

Two rules are enforced here, both directly from the data dictionary:

* the wait clock starts at **first entry**, not at a station's own claim;
* a consult without an evening answer is still demand.  Dropping it (the common
  "among answered" shortcut) silently removes exactly the travellers a take-
  over hurts most, so every share below is reported next to its demand
  denominator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .common import Consult, Dataset, mean, wilson


@dataclass(frozen=True)
class CapAssessment:
    scope: str
    skill: str
    slot: int
    cap: int
    goal: float
    demand: int
    answered: int
    compliant_answered: int
    share_among_answered: float | None
    share_of_demand: float | None
    share_of_demand_ci: tuple[float, float]
    definite_failures: int
    undecided: int
    unproven_at_goal: bool
    proven_at_goal: bool
    verdict: str


def scope_consults(ds: Dataset, scope: dict) -> list[Consult]:
    site = scope["site"]
    return [
        c
        for c in ds.consults
        if c.skill == scope["skill"]
        and c.slot == scope["slot"]
        and (site == "全区" or c.origin == site)
    ]


def assess(scope: dict, consults: Iterable[Consult]) -> CapAssessment:
    rows = list(consults)
    cap = int(scope["wait_cap"])
    goal = float(scope["coverage"])
    answered = [c for c in rows if c.wait is not None]
    ok = [c for c in answered if c.wait <= cap]
    unresolved = [c for c in rows if c.wait is None]
    definite = sum(1 for c in unresolved if c.censored_lower > cap)
    within_floor = sum(1 for c in unresolved if c.censored_upper <= cap)
    undecided = len(unresolved) - definite - within_floor
    lo, hi = wilson(len(ok), len(rows))
    if hi < goal:
        verdict = "已证不达标"
    elif lo >= goal:
        verdict = "已证达标"
    else:
        verdict = "未能判定"
    return CapAssessment(
        scope=scope["site"],
        skill=scope["skill"],
        slot=scope["slot"],
        cap=cap,
        goal=goal,
        demand=len(rows),
        answered=len(answered),
        compliant_answered=len(ok),
        share_among_answered=(len(ok) / len(answered)) if answered else None,
        share_of_demand=(len(ok) / len(rows)) if rows else None,
        share_of_demand_ci=(lo, hi),
        definite_failures=definite,
        undecided=undecided,
        unproven_at_goal=hi < goal,
        proven_at_goal=lo >= goal,
        verdict=verdict,
    )


def baseline_cell_wait_decomposition(consults: Iterable[Consult]) -> dict:
    """Within one (day, slot, site, skill) cell, split observed service phase.

    Short waits (3-5 minutes) are service right after arrival; long waits
    (14-29) are arrivals queued behind an occupied seat.  Transfer arrivals and
    direct arrivals are separated so a takeover is never credited with a wait
    its origin already absorbed.
    """
    rows = [c for c in consults if c.answer is not None]
    cells: dict[tuple, list[Consult]] = defaultdict(list)
    for c in rows:
        cells[(c.day, c.slot, c.origin, c.skill)].append(c)
    short = {"transfer": 0, "direct": 0}
    long = {"transfer": 0, "direct": 0}
    n = {"transfer": 0, "direct": 0}
    first_wait: list[int] = []
    for cell in cells.values():
        ordered = sorted(cell, key=lambda c: (c.first, c.thread))
        first_wait.append(ordered[0].wait)  # type: ignore[arg-type]
        for c in ordered:
            kind = "transfer" if c.is_transfer else "direct"
            n[kind] += 1
            if c.wait <= 5:  # type: ignore[operator]
                short[kind] += 1
            elif c.wait >= 14:  # type: ignore[operator]
                long[kind] += 1
    return {
        "cells": len(cells),
        "short_wait": short,
        "long_wait": long,
        "n": n,
        "first_arrival_wait_mean": mean(first_wait),
        "first_arrival_wait_median": sorted(first_wait)[len(first_wait) // 2] if first_wait else None,
    }


def build(ds: Dataset | None = None) -> dict:
    ds = ds or Dataset.read()
    assessments = [assess(scope, scope_consults(ds, scope)) for scope in ds.guarantees]
    return {
        "assessments": assessments,
        "cell_wait_decomposition": baseline_cell_wait_decomposition(ds.consults),
        "note": (
            "share_of_demand counts every first entry (answered, departed, parked "
            "and next-morning callback); share_among_answered is shown only to "
            "expose the survivor bias it creates."
        ),
    }
