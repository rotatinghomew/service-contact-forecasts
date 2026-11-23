"""Paired replay of two service rules on the same recorded service episodes.

Design: the *set of recorded service episodes* is the capacity witness.  Each
answered consult leaves one episode at a station cell (day, slot, answer site,
skill) with a start minute and a duration.  Replaying on the same day/slot
reuses exactly those episodes - same count, same starts, same durations - and
only re-decides **who** gets each episode.  Both rules therefore see the same
real arrivals, the same handoff timings and the same information cut; the
difference in wait is attributable to the rule.

Why episodes and not the published roster: the roster is a reference resource
with fixed published rosters for replay days, but on the replay days observed
concurrent service exceeds the qualified seats on the roster in 136 of 602
site/skill/slot cells (and 85 slot-3 answers occur after the nominal 110-minute
seat minute end).  A literal seat-minute replay would therefore *invent*
unmet demand; an episode replay does not.

Rule A - station queue + temporary lending
    A consult is queued at its own station: direct consults from first entry,
    claimed consults at the destination from their claim.  Each station's
    episodes are handed out in start order to its queue in station-arrival
    order.  A station with skill demand but no episode of its own borrows a
    lendable member - but a loan only adds capacity if the roster actually has
    a lendable seat for that skill in that slot, and never for ``本地专用``.
    Staffing is fixed per day, so the loan is recorded as a booking, not
    silently converted into an extra episode.

Rule B - regional first-entry order
    Episodes for a skill are pooled across the region and assigned in first-
    entry order.  ``本地专用`` episodes still only serve consults that entered
    or were claimed at that site.  Serving a consult away from its first site
    pays the recorded handoff cost (8 minutes to offer, 3 minutes to answer),
    so the traveller never gains time the physical handoff cannot produce.

Censoring is preserved in both arms: a consult that gets no episode is not
re-labelled as departed, answered or "no need".  It keeps a provable floor (the
last recorded mark) and a ceiling (end of the service slot), and the cap table
reports how much of the class is undecided.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .common import (
    Consult,
    Dataset,
    SITES,
    SLOT_START,
    mean,
    quantile,
    wilson,
)

OFFER_DELAY = 8  # recorded on every handoff in the file
READY_DELAY = 3  # recorded minimum answer delay for a locally served consult
HANDOFF_CYCLE = ("港口", "滨海馆", "芦湾酒店", "老城馆")
CROSS_SITE_READY = OFFER_DELAY + READY_DELAY  # 11 minutes


@dataclass
class Episode:
    day: str
    slot: int
    site: str
    skill: str
    start: int
    duration: float
    origin_thread: str
    policy: str = "unrecorded"
    taken_by: str | None = None


@dataclass
class Case:
    consult: Consult
    ready: int
    served: bool = False
    episode: Episode | None = None
    wait: int | None = None
    served_site: str | None = None
    reached_by: str | None = None
    service_tie: bool = False

    @property
    def floor(self) -> int:
        if self.served:
            return self.wait or 0
        return self.consult.censored_lower

    @property
    def ceiling(self) -> int:
        if self.served:
            return self.wait or 0
        return self.consult.censored_upper


@dataclass
class Arm:
    rule: str
    cases: list[Case]
    loans: list[dict] = field(default_factory=list)

    def by(self, key) -> dict[str, dict]:
        groups: dict[str, list[Case]] = defaultdict(list)
        for c in self.cases:
            groups[key(c)].append(c)
        return {k: summarise(v) for k, v in sorted(groups.items())}

    @property
    def overall(self) -> dict:
        return summarise(self.cases)


def summarise(cases: Iterable[Case]) -> dict:
    rows = list(cases)
    waits = [c.wait for c in rows if c.wait is not None]
    unmet = [c for c in rows if not c.served]
    return {
        "consults": len(rows),
        "served": len(waits),
        "unmet": len(unmet),
        "mean": mean(waits),
        "median": quantile(waits, 0.5),
        "p90": quantile(waits, 0.9),
        "max": max(waits) if waits else None,
        "unmet_floor_mean": mean([c.floor for c in unmet]),
        "unmet_ceiling_mean": mean([c.ceiling for c in unmet]),
    }


def build_episodes(ds: Dataset) -> dict[tuple, list[Episode]]:
    """One episode per recorded answer, keyed by day/slot/site/skill."""
    out: dict[tuple, list[Episode]] = defaultdict(list)
    for c in ds.consults:
        if c.answer is None:
            continue
        ep = Episode(
            day=c.day,
            slot=c.slot,
            site=c.answer_site or c.origin,
            skill=c.skill,
            start=c.answer,
            duration=float(c.duration) if c.duration is not None else 6.0,
            origin_thread=c.thread,
        )
        out[(c.day, c.slot, ep.site, c.skill)].append(ep)
    for eps in out.values():
        eps.sort(key=lambda e: (e.start, e.origin_thread))
    return out


def claimed_ready(c: Consult) -> int | None:
    """Claim minute as known on the consult day, not after later corrections.

    ``handoff_ack`` rows for two thirds of the transfers only became visible on
    the second day, and the 32 correction rows became visible two to four days
    later.  Using them in a same-day replay would leak future information into
    the past.  The replay therefore accepts a claim only when the handoff row is
    known on or before the consult day and is not a correction.
    """
    if c.handoff_known_asof is not None and c.handoff_known_asof > c.day:
        return None
    if c.state_asof == "更正已认领":
        return None
    if c.state_asof == "已认领" and c.accepted_asof is not None:
        return c.accepted_asof
    return None


def _make_cases(consults: Iterable[Consult]) -> list[Case]:
    cases = []
    for c in consults:
        claim = claimed_ready(c)
        ready = (claim + READY_DELAY) if claim is not None else (c.first + READY_DELAY)
        # unclaimed handoffs: the origin keeps holding the consult, so it waits
        # where it entered rather than disappearing from the residual load
        via = c.state_asof if c.is_transfer else None
        cases.append(Case(consult=c, ready=ready, reached_by=via))
    return cases


def _queue_key(c: Case) -> tuple[int, int, str]:
    return (c.ready, c.consult.first, c.consult.thread)


def _first_key(c: Case) -> tuple[int, str]:
    return (c.consult.first, c.consult.thread)


def _can_use_episode(c: Case, ep: Episode) -> bool:
    return c.consult.skill == ep.skill


def _episode_seat_policy(ep: Episode, ds: Dataset) -> str:
    """Which consults an episode's seats may serve.

    ``local_only``  the site's roster for this skill holds no 可借 seat, so only
                    consults attached to that site may use the episode;
    ``lendable``    a 可借 seat exists, so anyone qualified by skill may use it;
    ``unrecorded``  the roster has no seat for this skill at all (the published
                    roster already under-counts the recorded service), so no
                    seat restriction can be imposed without inventing one.
    """
    seats = ds.qualified_seats(ep.day, ep.slot, ep.site, ep.skill)
    if not seats:
        return "unrecorded"
    if ds.lendable_seats(ep.day, ep.slot, ep.site, ep.skill):
        return "lendable"
    return "local_only"


def _policy_ok(c: Case, ep: Episode, policy: str) -> bool:
    if policy in ("lendable", "unrecorded"):
        return True
    if c.consult.origin == ep.site:
        return True
    claim = claimed_ready(c.consult)
    return claim is not None and c.consult.destination == ep.site


def _ready_for(c: Case, ep: Episode) -> int:
    if c.consult.origin == ep.site:
        return c.ready
    claim = claimed_ready(c.consult)
    if claim is not None and c.consult.destination == ep.site:
        return c.ready
    # a new regional routing still pays the physical handoff
    return max(c.ready, c.consult.first + CROSS_SITE_READY)


def _run_rule_a(
    ds: Dataset,
    cases: list[Case],
    episodes: dict[tuple, list[Episode]],
) -> Arm:
    loans: list[dict] = []
    # station queues: the consult waits where it currently is
    queues: dict[tuple, list[Case]] = defaultdict(list)
    for c in cases:
        claim = claimed_ready(c.consult)
        site = c.consult.destination if claim is not None else c.consult.origin
        queues[(c.consult.day, c.consult.slot, site, c.consult.skill)].append(c)
    for q in queues.values():
        q.sort(key=_queue_key)
    for key, q in sorted(queues.items()):
        day, slot, site, skill = key
        eps = list(episodes.get(key, []))
        if not eps:
            # Borrowed capacity is *booked*, not conjured: the loan is recorded
            # only if the roster really has a lendable (可借) seat for that
            # skill, and it consumes that seat for the whole slot.  The episode
            # replay has no start minute for borrowed capacity, so the loan is
            # reported as an unserved residual here rather than silently
            # creating service.
            idx = HANDOFF_CYCLE.index(site) if site in HANDOFF_CYCLE else 0
            for step in range(1, len(HANDOFF_CYCLE)):
                donor = HANDOFF_CYCLE[(idx + step) % len(HANDOFF_CYCLE)]
                seats = ds.lendable_seats(day, slot, donor, skill)
                if seats:
                    loans.append(
                        {
                            "day": day,
                            "slot": slot,
                            "recipient": site,
                            "donor": donor,
                            "member": seats[0]["member"],
                            "skills": list(seats[0]["skills"]),
                            "skill": skill,
                            "consults_waiting": len(q),
                            "capacity_source": "donor roster seat, booked for the slot",
                        }
                    )
                    break
            continue
        for ep in eps:
            candidates = [
                c for c in q if not c.served and _can_use_episode(c, ep) and c.ready <= ep.start
            ]
            if not candidates:
                continue
            c = candidates[0]
            c.served = True
            c.episode = ep
            c.wait = ep.start - c.consult.first
            c.served_site = ep.site
            c.service_tie = c.ready == ep.start
            c.reached_by = "station"
            ep.taken_by = c.consult.thread
    return Arm(rule="A", cases=cases, loans=loans)


def _run_rule_b(
    ds: Dataset,
    cases: list[Case],
    episodes: dict[tuple, list[Episode]],
) -> Arm:
    pools: dict[tuple, list[Episode]] = defaultdict(list)
    for (day, slot, site, skill), eps in episodes.items():
        for ep in eps:
            ep.policy = _episode_seat_policy(ep, ds)
            pools[(day, slot, skill)].append(ep)
    for pool in pools.values():
        pool.sort(key=lambda e: (e.start, e.site, e.origin_thread))
    regional: dict[tuple, list[Case]] = defaultdict(list)
    for c in cases:
        regional[(c.consult.day, c.consult.slot, c.consult.skill)].append(c)
    for q in regional.values():
        q.sort(key=lambda c: _first_key(c))
    for key, q in sorted(regional.items()):
        pool = pools.get(key, [])
        for ep in pool:
            candidates = [
                c
                for c in q
                if not c.served
                and _can_use_episode(c, ep)
                and _policy_ok(c, ep, ep.policy)
                and _ready_for(c, ep) <= ep.start
            ]
            if not candidates:
                continue
            c = candidates[0]
            c.served = True
            c.episode = ep
            c.wait = ep.start - c.consult.first
            c.served_site = ep.site
            c.service_tie = c.ready == ep.start
            c.reached_by = "origin" if c.consult.origin == ep.site else "regional"
            ep.taken_by = c.consult.thread
    return Arm(rule="B", cases=cases)


def replay(ds: Dataset | None = None, *, days: Iterable[str] | None = None) -> dict[str, Arm]:
    ds = ds or Dataset.read()
    day_list = list(days) if days is not None else ds.candidate_days()
    episodes = build_episodes(ds)
    day_set = set(day_list)
    consults = [c for c in ds.consults if c.day in day_set]
    arms: dict[str, Arm] = {}
    for rule in ("A", "B"):
        aggregated: list[Case] = []
        loans: list[dict] = []
        for day in day_list:
            day_consults = [c for c in consults if c.day == day]
            cases = _make_cases(day_consults)
            ep = {k: v for k, v in episodes.items() if k[0] == day}
            arm = _run_rule_a(ds, cases, ep) if rule == "A" else _run_rule_b(ds, cases, ep)
            aggregated.extend(arm.cases)
            loans.extend(arm.loans)
        arms[rule] = Arm(rule=rule, cases=aggregated, loans=loans)
    return arms


def paired(arms: dict[str, Arm]) -> dict:
    a = {c.consult.thread: c for c in arms["A"].cases}
    b = {c.consult.thread: c for c in arms["B"].cases}
    rows = []
    for thread in sorted(set(a) | set(b)):
        ca, cb = a.get(thread), b.get(thread)
        ref = ca or cb
        assert ref is not None
        rows.append(
            {
                "thread": thread,
                "skill": ref.consult.skill,
                "origin": ref.consult.origin,
                "slot": ref.consult.slot,
                "first": ref.consult.first,
                "wait_a": ca.wait if ca and ca.served else None,
                "wait_b": cb.wait if cb and cb.served else None,
                "served_a": bool(ca and ca.served),
                "served_b": bool(cb and cb.served),
                "floor_a": ca.floor if ca else None,
                "floor_b": cb.floor if cb else None,
            }
        )
    paired_rows = [r for r in rows if r["wait_a"] is not None and r["wait_b"] is not None]
    diffs = [r["wait_b"] - r["wait_a"] for r in paired_rows]

    def classes(key):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        out = {}
        for name, group in sorted(groups.items()):
            both = [r for r in group if r["wait_a"] is not None and r["wait_b"] is not None]
            out[name] = {
                "consults": len(group),
                "served_a": sum(r["served_a"] for r in group),
                "served_b": sum(r["served_b"] for r in group),
                "mean_a": mean([r["wait_a"] for r in both]),
                "mean_b": mean([r["wait_b"] for r in both]),
                "mean_diff": mean([r["wait_b"] - r["wait_a"] for r in both]),
                "b_longer": sum(1 for r in both if r["wait_b"] > r["wait_a"]),
                "b_shorter": sum(1 for r in both if r["wait_b"] < r["wait_a"]),
                "b_equal": sum(1 for r in both if r["wait_b"] == r["wait_a"]),
            }
        return out

    return {
        "consults": len(rows),
        "paired": len(paired_rows),
        "a_only": sum(1 for r in rows if r["served_a"] and not r["served_b"]),
        "b_only": sum(1 for r in rows if r["served_b"] and not r["served_a"]),
        "neither": sum(1 for r in rows if not r["served_a"] and not r["served_b"]),
        "mean_diff_b_minus_a": mean(diffs),
        "median_diff_b_minus_a": quantile(diffs, 0.5),
        "p90_diff_b_minus_a": quantile(diffs, 0.9),
        "b_longer": sum(1 for d in diffs if d > 0),
        "b_shorter": sum(1 for d in diffs if d < 0),
        "by_skill": classes(lambda r: r["skill"]),
        "by_origin": classes(lambda r: r["origin"]),
        "by_slot": classes(lambda r: str(r["slot"])),
        "rows": rows,
    }


def cap_table(arms: dict[str, Arm], caps: Iterable[int] = (18, 22, 25, 27)) -> dict:
    """Cap compliance with censoring kept visible.

    ``definitely_within`` counts served consults at or under cap;
    ``definitely_over`` counts served consults over cap plus unserved consults
    whose provable floor already exceeds it; everything else is undecided.
    """
    out: dict = {}
    for cap in caps:
        per_rule: dict[str, dict] = {}
        for rule, arm in arms.items():
            groups: dict[tuple, list[Case]] = defaultdict(list)
            for c in arm.cases:
                groups[(c.consult.skill, c.consult.origin)].append(c)
            per_rule[rule] = {}
            for key, group in sorted(groups.items()):
                ok = sum(1 for c in group if c.served and c.wait <= cap)
                over = sum(
                    1
                    for c in group
                    if (c.served and c.wait > cap) or (not c.served and c.floor > cap)
                )
                undecided = len(group) - ok - over
                lo, hi = wilson(ok, len(group))
                per_rule[rule][f"{key[0]}|{key[1]}"] = {
                    "n": len(group),
                    "definitely_within": ok,
                    "definitely_over": over,
                    "undecided": undecided,
                    "share_ci": (lo, hi),
                    "could_hold_80": hi >= 0.8,
                    "could_fail_80": lo < 0.8,
                }
        flips = []
        for cls, ra in per_rule["A"].items():
            rb = per_rule["B"].get(cls)
            if rb is None:
                continue
            if ra["could_hold_80"] != rb["could_hold_80"] or ra["could_fail_80"] != rb["could_fail_80"]:
                flips.append(cls)
        out[str(cap)] = {
            "per_rule": per_rule,
            "verdict_flips_at_80pct": flips,
            "comparison_only": bool(flips),
        }
    return out


def build(ds: Dataset | None = None) -> dict:
    ds = ds or Dataset.read()
    arms = replay(ds)
    p = paired(arms)
    summary = {
        rule: {
            "overall": arm.overall,
            "by_skill": arm.by(lambda c: c.consult.skill),
            "by_origin": arm.by(lambda c: c.consult.origin),
            "by_slot": arm.by(lambda c: str(c.consult.slot)),
            "loans": len(arm.loans),
        }
        for rule, arm in arms.items()
    }
    return {
        "days": ds.candidate_days(),
        "summary": summary,
        "paired": {k: v for k, v in p.items() if k != "rows"},
        "cap_table": cap_table(arms),
        "rows": p["rows"],
        "loans": arms["A"].loans,
    }
