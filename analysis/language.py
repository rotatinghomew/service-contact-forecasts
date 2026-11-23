"""Language-group gaps, with intervals a duty manager can trace back.

The recurring trap in this file is comparing *answered* travellers:

* Mandarin has the lowest answer rate (58.4% vs 97-98% for Cantonese and
  English), so the 58% who do get answered are the easiest ones and Mandarin's
  answered wait looks competitive;
* transfers are all Mandarin, so a transfer-cell comparison of "answered wait"
  compares like with unlike.

Every number below is therefore given as a **bounded share of demand**:
answered-inside-cap is the lower bound, answered-inside-cap plus the consults
whose interval still straddles the cap is the upper bound.  A gap between
language groups is only called a gap when the two intervals do not overlap.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .common import Consult, Dataset, SKILLS, SITES, band, mean, quantile


def _bounds(rows: list[Consult], cap: int) -> tuple[float, float]:
    n = len(rows)
    if n == 0:
        return (0.0, 1.0)
    inside = sum(1 for c in rows if c.wait is not None and c.wait <= cap)
    undecided = sum(
        1
        for c in rows
        if c.wait is None and c.censored_lower <= cap < c.censored_upper
    )
    return (inside / n, (inside + undecided) / n)


def skill_gap(consults: Iterable[Consult], caps: Iterable[int] = (18, 22, 25, 27)) -> dict:
    rows = list(consults)
    out: dict = {"overall": {}, "by_site": {}, "gaps": []}
    for skill in SKILLS:
        group = [c for c in rows if c.skill == skill]
        answered = [c for c in group if c.wait is not None]
        out["overall"][skill] = {
            "demand": len(group),
            "answered": len(answered),
            "answer_rate": len(answered) / len(group) if group else None,
            "mean_answered_wait": mean([c.wait for c in answered]),
            "median_answered_wait": quantile([c.wait for c in answered], 0.5),
            "bands": dict(Counter(band(c.wait) for c in answered)),
            "cap_bounds": {str(cap): _bounds(group, cap) for cap in caps},
        }
    for site in SITES:
        out["by_site"][site] = {}
        for skill in SKILLS:
            group = [c for c in rows if c.origin == site and c.skill == skill]
            if not group:
                continue
            answered = [c for c in group if c.wait is not None]
            out["by_site"][site][skill] = {
                "demand": len(group),
                "answered": len(answered),
                "answer_rate": len(answered) / len(group),
                "mean_answered_wait": mean([c.wait for c in answered]),
                "median_answered_wait": quantile([c.wait for c in answered], 0.5),
                "cap_bounds": {str(cap): _bounds(group, cap) for cap in caps},
                "unseen": len(group) - len(answered),
            }
    # language gaps inside each site, at each cap
    for site in SITES:
        skills = sorted(out["by_site"].get(site, {}))
        for i, left in enumerate(skills):
            for right in skills[i + 1 :]:
                a = out["by_site"][site][left]
                b = out["by_site"][site][right]
                for cap in caps:
                    la, ua = a["cap_bounds"][str(cap)]
                    lb, ub = b["cap_bounds"][str(cap)]
                    separated = ua < lb or ub < la
                    out["gaps"].append(
                        {
                            "site": site,
                            "cap": cap,
                            "left": left,
                            "right": right,
                            "left_bounds": (la, ua),
                            "right_bounds": (lb, ub),
                            "separated": separated,
                            "difference_of_means_answered": (
                                (a["mean_answered_wait"] or 0) - (b["mean_answered_wait"] or 0)
                            ),
                        }
                    )
    return out


def answered_survivorship(consults: Iterable[Consult], caps: Iterable[int] = (18, 22, 25, 27)) -> dict:
    """The "answered-only" comparison, shown next to the demand-denominator one."""
    rows = list(consults)
    out = {}
    for skill in SKILLS:
        group = [c for c in rows if c.skill == skill]
        answered = [c for c in group if c.wait is not None]
        out[skill] = {
            "demand": len(group),
            "answered": len(answered),
            "answer_rate": len(answered) / len(group),
            "among_answered": {
                str(cap): (sum(1 for c in answered if c.wait <= cap) / len(answered))
                if answered
                else None
                for cap in caps
            },
            "of_demand": {
                str(cap): _bounds(group, cap) for cap in caps
            },
        }
    return out


def unseen_ranges(consults: Iterable[Consult]) -> dict:
    """Provable range for the wait segments that were never observed."""
    rows = [c for c in consults if c.wait is None]
    groups: dict[tuple[str, int], list[Consult]] = defaultdict(list)
    for c in rows:
        groups[(c.skill, c.slot)].append(c)
    out = {}
    for (skill, slot), group in sorted(groups.items()):
        out[f"{skill}|{slot}"] = {
            "consults": len(group),
            "floor_min": min(c.censored_lower for c in group),
            "floor_max": max(c.censored_lower for c in group),
            "ceiling_min": min(c.censored_upper for c in group),
            "ceiling_max": max(c.censored_upper for c in group),
            "with_mark": sum(1 for c in group if c.departure is not None or c.park is not None),
        }
    return out


def build(ds: Dataset | None = None) -> dict:
    ds = ds or Dataset.read()
    return {
        "skill_gap": skill_gap(ds.consults),
        "survivorship": answered_survivorship(ds.consults),
        "unseen_ranges": unseen_ranges(ds.consults),
    }
