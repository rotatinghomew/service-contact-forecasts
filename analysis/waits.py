"""Wait accounting along the real consult chain.

Three views are deliberately kept apart:

``chain``
    One traveller, one clock: ``answer - first``.  It is the only wait that can
    be compared with a first-entry promise, and it is the only wait that is
    counted once even when two sites are busy at the same time.
``station``
    The same elapsed minutes sliced by who held the consult
    (origin before the offer / transfer dwell / destination after the claim /
    resume at the origin after an unclaimed offer).  The slices add up to the
    chain wait exactly.
``dual``
    Minutes in which *both* sites are busy on one consult.  They are real load
    on both sides but must never be added to the traveller's clock.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .common import Consult, Dataset, band, mean, quantile


@dataclass(frozen=True)
class WaitSummary:
    n: int
    n_unseen: int
    mean: float | None
    median: float | None
    p90: float | None
    minimum: float | None
    maximum: float | None


def summarise(consults: Iterable[Consult], *, use_chain: bool = True) -> WaitSummary:
    rows = list(consults)
    waits = [c.wait for c in rows if c.wait is not None]
    return WaitSummary(
        n=len(rows),
        n_unseen=len(rows) - len(waits),
        mean=mean(waits),
        median=quantile(waits, 0.5),
        p90=quantile(waits, 0.9),
        minimum=min(waits) if waits else None,
        maximum=max(waits) if waits else None,
    )


def group_waits(consults: Iterable[Consult], key) -> dict[str, WaitSummary]:
    groups: dict[str, list[Consult]] = defaultdict(list)
    for c in consults:
        groups[key(c)].append(c)
    return {k: summarise(v) for k, v in sorted(groups.items())}


def chain_slices(consults: Iterable[Consult]) -> dict[str, float | int]:
    """Exact decomposition of answered chain wait into station-held minutes.

    Identity::

        chain = origin_before_offer + dwell + destination_after_claim + resume
                (+ direct, i.e. consults that never left the origin)

    ``dwell`` is counted once for the traveller; the same minutes are also the
    dual-occupancy load (reported separately in :func:`dual_occupancy`).
    """
    direct = 0.0
    origin_before_offer = 0.0
    dwell = 0.0
    destination = 0.0
    resume = 0.0
    chain = 0.0
    counted = 0
    for c in consults:
        if c.wait is None:
            continue
        counted += 1
        chain += c.wait
        if not c.is_transfer:
            direct += c.wait
            continue
        assert c.pre_offer is not None
        origin_before_offer += c.pre_offer
        if c.claimed_asof:
            dwell += c.dwell_asof or 0
            destination += c.dest_service_asof or 0
        else:
            resume += c.origin_resume or 0
    return {
        "answered": counted,
        "chain": chain,
        "direct_at_origin": direct,
        "origin_before_offer": origin_before_offer,
        "dwell_dual": dwell,
        "destination_after_claim": destination,
        "origin_resume": resume,
        "components_sum": direct + origin_before_offer + dwell + destination + resume,
    }


def dual_occupancy(consults: Iterable[Consult]) -> dict[str, float | int]:
    """Dwell minutes during which both sites carried the same consult."""
    rows = [c for c in consults if c.dwell_asof is not None]
    minutes = [c.dwell_asof for c in rows]
    return {
        "consults": len(rows),
        "seat_minutes_per_side": sum(minutes),
        "mean": mean(minutes),
        "matched_both_sites": len(rows),
        "max": max(minutes) if minutes else 0,
        # the traveller's own clock counts the same dwell once; the two sites
        # hold it concurrently, so site-side load is twice this figure
        "dual_seat_minutes_all_sites": 2 * sum(minutes),
    }


def region_reconciliation(consults: Iterable[Consult]) -> dict:
    """Every answered minute attributed to a site layer, and the remainder.

    The regional metric must be explainable from the same consult facts as the
    site metrics.  For answered consults::

        region = sum(site-held: direct + pre-offer + origin resume)
               + sum(destination-held: after claim)
               + sum(jointly-held: dwell, shared by both sites)
               + unattributed (bounded interval for consults with no answer)

    Each site row is computed from the same consults, so the site rows sum to
    the regional figure.  The unattributed bucket is never merged into the
    observed total: it is reported as an interval.
    """
    rows = list(consults)
    sites = sorted({c.origin for c in rows} | {c.destination for c in rows if c.destination})
    per_site: dict[str, dict] = {}
    for site in sites:
        own_direct = [c for c in rows if c.origin == site and not c.is_transfer and c.wait is not None]
        offered = [c for c in rows if c.origin == site and c.is_transfer and c.wait is not None]
        inbound = [
            c
            for c in rows
            if c.destination == site and c.claimed_asof and c.wait is not None
        ]
        per_site[site] = {
            "held_direct": sum(c.wait for c in own_direct),
            "held_before_offer": sum(c.pre_offer or 0 for c in offered),
            "held_resume": sum(c.origin_resume or 0 for c in offered),
            "held_at_own_site": (
                sum(c.wait for c in own_direct)
                + sum(c.pre_offer or 0 for c in offered)
                + sum(c.origin_resume or 0 for c in offered)
            ),
            "held_after_claim": sum(c.dest_service_asof or 0 for c in inbound),
            "shared_dwell": sum(c.dwell_asof or 0 for c in offered),
            "consults_entered": sum(1 for c in rows if c.origin == site),
            "consults_answered_here": sum(1 for c in rows if c.answer_site == site),
        }
    held_total = sum(v["held_at_own_site"] for v in per_site.values())
    after_claim = sum(v["held_after_claim"] for v in per_site.values())
    dwell = sum(v["shared_dwell"] for v in per_site.values())
    observed = sum(c.wait for c in rows if c.wait is not None)
    unresolved = [c for c in rows if c.wait is None]
    return {
        "per_site": per_site,
        "site_held_total": held_total,
        "destination_after_claim_total": after_claim,
        "shared_dwell_total": dwell,
        "sum_of_layers": held_total + after_claim + dwell,
        "observed_chain_wait": observed,
        "balances": abs(held_total + after_claim + dwell - observed) < 1e-6,
        "unattributed": {
            "consults": len(unresolved),
            "floor_minutes": sum(c.censored_lower for c in unresolved),
            "ceiling_minutes": sum(c.censored_upper for c in unresolved),
            "note": "未答复部分只给区间，不并入上面的已观察总额",
        },
    }


def station_table(consults: Iterable[Consult]) -> list[dict]:
    """Per-site and per-flow load, using the same consult facts everywhere."""
    rows = list(consults)
    out: list[dict] = []
    for site in sorted({c.origin for c in rows} | {c.answer_site for c in rows if c.answer_site}):
        direct = [c for c in rows if c.origin == site and not c.is_transfer]
        outbound = [c for c in rows if c.origin == site and c.is_transfer]
        inbound = [c for c in rows if c.destination == site and c.claimed_asof]
        answered_here = [c for c in rows if c.answer_site == site]
        out.append(
            {
                "site": site,
                "consults_entered": len([c for c in rows if c.origin == site]),
                "direct": len(direct),
                "offered_out": len(outbound),
                "claimed_in": len(inbound),
                "answered_here": len(answered_here),
                "own_wait_mean": mean([c.wait for c in direct if c.wait is not None]),
                "inbound_service_mean": mean(
                    [c.dest_service_asof for c in inbound if c.dest_service_asof is not None and c.answer is not None]
                ),
                "dwell_held_mean": mean([c.dwell_asof for c in outbound if c.dwell_asof is not None]),
            }
        )
    return out


def flow_table(consults: Iterable[Consult]) -> list[dict]:
    rows = [c for c in consults if c.is_transfer]
    groups: dict[tuple[str, str], list[Consult]] = defaultdict(list)
    for c in rows:
        groups[(c.origin, c.destination or "")].append(c)
    out = []
    for (origin, dest), group in sorted(groups.items()):
        answered = [c for c in group if c.wait is not None]
        out.append(
            {
                "origin": origin,
                "destination": dest,
                "consults": len(group),
                "claimed": sum(1 for c in group if c.claimed_asof),
                "answered": len(answered),
                "answer_rate": len(answered) / len(group) if group else None,
                "chain_wait_mean": mean([c.wait for c in answered]),
                "chain_wait_median": quantile([c.wait for c in answered], 0.5),
                "dwell_mean": mean([c.dwell_asof for c in group if c.dwell_asof is not None]),
                "dest_service_mean": mean([c.dest_service_asof for c in answered if c.claimed_asof]),
            }
        )
    return out


def queue_placement(consults: Iterable[Consult]) -> dict:
    """How often a later first entry was answered before an earlier one.

    Cells are (day, slot, origin site, skill): the same four sites share the
    language, so a single queue is the natural comparison.  Only answered
    consults enter the pair count; unanswered ones are reported separately as
    censored, never silently dropped.
    """
    rows = [c for c in consults if c.answer is not None]
    cells: dict[tuple, list[Consult]] = defaultdict(list)
    for c in rows:
        cells[(c.day, c.slot, c.origin, c.skill)].append(c)
    pairs = 0
    inversions = 0
    by_rank: dict[int, list[int]] = defaultdict(list)
    untouched = 0
    first_wait: list[int] = []
    for cell in cells.values():
        ordered = sorted(cell, key=lambda c: (c.first, c.thread))
        first_wait.append(ordered[0].wait)  # type: ignore[arg-type]
        for i, a in enumerate(ordered):
            later_before = 0
            for b in ordered[i + 1 :]:
                pairs += 1
                if b.answer < a.answer:  # type: ignore[operator]
                    inversions += 1
                    later_before += 1
            by_rank[i].append(later_before)
            if i > 0 and ordered[i - 1].answer < a.answer:  # type: ignore[operator]
                untouched += 1
    return {
        "cells": len(cells),
        "answered_pairs": pairs,
        "inversions": inversions,
        "inversion_rate": inversions / pairs if pairs else None,
        "answered_in_first_entry_order": untouched,
        "first_arrival_wait_mean": mean(first_wait),
        "first_arrival_wait_median": quantile(first_wait, 0.5),
        "rank_overtaken_mean": {r: mean(v) for r, v in sorted(by_rank.items()) if r < 6},
    }


def censoring(consults: Iterable[Consult], caps: Iterable[int] = (18, 22, 25, 27)) -> dict:
    """What the record proves about consults without an in-window answer.

    There is no such thing as "no demand" here: a departure, a parked consult
    and a next-morning callback all remain demand.  For each unresolved consult
    the chain wait is only bounded: the last recorded mark (departure/park) is a
    floor, and the end of the service slot is the ceiling.  The floor alone is
    enough to prove a cap breach; the ceiling alone is never enough to prove
    compliance.
    """
    rows = list(consults)
    out: dict = {"total": len(rows), "by_outcome": Counter(c.outcome for c in rows)}
    unresolved = [c for c in rows if c.answer is None]
    out["unresolved"] = len(unresolved)
    out["unresolved_by_outcome"] = Counter(c.outcome for c in unresolved)
    out["floors"] = {
        "mean": mean([c.censored_lower for c in unresolved]),
        "max": max((c.censored_lower for c in unresolved), default=None),
        "with_recorded_mark": sum(
            1 for c in unresolved if c.departure is not None or c.park is not None
        ),
        "without_recorded_mark": sum(
            1 for c in unresolved if c.departure is None and c.park is None
        ),
    }
    out["ceiling_min"] = min((c.censored_upper for c in unresolved), default=None)
    out["ceiling_max"] = max((c.censored_upper for c in unresolved), default=None)
    out["per_cap"] = {}
    for cap in caps:
        definitely_over = sum(1 for c in unresolved if c.censored_lower > cap)
        maybe_over = sum(
            1 for c in unresolved if c.censored_lower <= cap < c.censored_upper
        )
        provably_within = sum(1 for c in unresolved if c.censored_upper <= cap)
        out["per_cap"][str(cap)] = {
            "definitely_over": definitely_over,
            "undecided": maybe_over,
            "provably_within": provably_within,
        }
    # next-morning callbacks: they restart the day base, so they cannot be
    # booked back into the previous evening's answered share.
    cb = [c for c in rows if c.outcome == "callback"]
    out["callback_consults"] = len(cb)
    # all callbacks restart on a next-morning day base, so their raw minute is
    # below the previous evening's window end; none of them can be booked back
    # into the previous evening's answered share.
    out["callback_next_morning_base"] = sum(
        1
        for c in cb
        if c.callback is not None and c.callback < c.window_end
    )
    out["callback_with_evening_answer"] = sum(
        1 for c in cb if c.answer is not None
    )
    out["departed_still_demand"] = sum(1 for c in rows if c.outcome == "departed")
    out["departed_with_unrecorded_time"] = sum(
        1 for c in rows if c.outcome == "departed" and c.departure is None
    )
    out["parked_still_demand"] = sum(1 for c in rows if c.outcome == "parked")
    return out


def cross_tab(consults: Iterable[Consult], caps: Iterable[int] = (18, 22, 25, 27)) -> list[dict]:
    """Site x language table that the duty manager can trace back to intervals.

    Each row is one (site, skill) cell with the same consult facts used by the
    regional table: demand, answered, mean/median/P90, the wait bands, and how
    many answered consults are inside each cap.  Nothing is weighted by cell
    size here - the row counts are the weights.
    """
    rows = list(consults)
    groups: dict[tuple[str, str], list[Consult]] = defaultdict(list)
    for c in rows:
        groups[(c.origin, c.skill)].append(c)
    out = []
    for (site, skill), group in sorted(groups.items()):
        answered = [c for c in group if c.wait is not None]
        bands_here = Counter(band(c.wait) for c in answered)
        out.append(
            {
                "site": site,
                "skill": skill,
                "demand": len(group),
                "answered": len(answered),
                "mean": mean([c.wait for c in answered]),
                "median": quantile([c.wait for c in answered], 0.5),
                "p90": quantile([c.wait for c in answered], 0.9),
                "max": max((c.wait for c in answered), default=None),
                "transfers": sum(1 for c in group if c.is_transfer),
                "bands": {b: bands_here.get(b, 0) for b in ("0-5", "6-10", "11-15", "16-20", "21-25", "26-29", "30+")},
        "within_cap_answered": {str(cap): sum(1 for c in answered if c.wait <= cap) for cap in caps},
                "share_of_demand": {str(cap): (sum(1 for c in answered if c.wait <= cap) / len(group)) if group else None for cap in caps},
                "share_of_answered": {str(cap): (sum(1 for c in answered if c.wait <= cap) / len(answered)) if answered else None for cap in caps},
                "definite_failures_unseen": {
                    str(cap): sum(1 for c in group if c.wait is None and c.censored_lower > cap) for cap in caps
                },
                "unseen": len(group) - len(answered),
            }
        )
    return out


def bands(consults: Iterable[Consult]) -> Counter[str]:
    return Counter(band(c.wait) for c in consults)


def build(ds: Dataset | None = None) -> dict:
    ds = ds or Dataset.read()
    consults = ds.consults
    answered = [c for c in consults if c.answer is not None]
    out = {
        "chain_summary": summarise(consults),
        "by_skill": group_waits(consults, lambda c: c.skill),
        "by_site": group_waits(consults, lambda c: c.origin),
        "by_slot": group_waits(consults, lambda c: str(c.slot)),
        "by_site_skill": cross_tab(consults),
        "by_transfer": {
            "transfer": summarise([c for c in consults if c.is_transfer]),
            "direct": summarise([c for c in consults if not c.is_transfer]),
        },
        "chain_slices": chain_slices(consults),
        "dual_occupancy": dual_occupancy(consults),
        "stations": station_table(consults),
        "region_reconciliation": region_reconciliation(consults),
        "flows": flow_table(consults),
        "queue": queue_placement(consults),
        "censoring": censoring(consults),
        "bands": bands(consults),
        "answered": len(answered),
    }
    return out
