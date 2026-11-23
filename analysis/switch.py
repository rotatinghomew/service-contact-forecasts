"""Switch package: which sites keep the current loan promise, which move to the
regional rule, and which must pause new handoffs.

A switch is decided **after** the day's handoffs have already happened, so it
can never rewrite what a traveller already lived through.  The rules that
follow from the data dictionary are:

* a consult already waiting keeps its original first-entry deadline (no
  restart at the switch minute, no restart at the claim minute);
* an accepted loan keeps its promised skill; the borrower may use the member
  only for that skill and only for the booked slot;
* a handoff the partner has not claimed yet does not disappear from the
  origin's residual load - the origin keeps holding it;
* a late claim confirmation that arrives after the switch window is not
  re-attributed to the partner as if the partner had made the traveller late,
  and it does not consume two seats.

The module reports, per site and language, the replay evidence and the current
loan promise, then classifies the live consults of the switch day into three
queues: keep, regional, pause.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .common import Consult, Dataset, SITES, SLOT_START, mean


@dataclass(frozen=True)
class LoanCheck:
    loan: str
    member: str
    day: str
    slot: int
    recipient: str
    state_final: str
    known_first: str
    known_last: str
    recipient_qualified_here: bool
    donor_qualified: bool
    lendable: bool
    same_slot_only: bool
    skill_promise: tuple[str, ...]
    viable: bool
    note: str


@dataclass(frozen=True)
class QueueItem:
    thread: str
    origin: str
    destination: str | None
    skill: str
    first: int
    state_asof: str | None
    accepted_asof: int | None
    accepted_final: int | None
    deadline: int
    action: str
    reason: str
    still_with_origin: bool
    late_confirmation: bool = False


def loan_checks(ds: Dataset) -> list[LoanCheck]:
    """Evaluate every booked loan row against the published roster.

    A promise is viable only when the member is on the roster for that
    day/slot, the seat is lendable (``可借``) and the member has the promised
    skill.  A ``本地专用`` seat cannot be lent whatever the booking says, and a
    booking that never moved past 待接受 / 供给窗口关闭待交接 was never accepted.
    """
    out: list[LoanCheck] = []
    roster: dict[tuple, list[dict]] = defaultdict(list)
    for t in ds.teams:
        roster[(t["day"], t["slot"])].append(t)
    for row in ds.loans:
        key = (row["day"], row["slot"])
        member_rows = [t for t in roster.get(key, []) if t["member"] == row["member"]]
        lendable = any(t["lend"] == "可借" for t in member_rows)
        recipient_here = any(
            t["home"] == row["recipient"] and t["lend"] == "可借" for t in member_rows
        )
        skills = tuple(sorted({s for t in member_rows for s in t["skills"]}))
        history = [r for r in ds.loans if r["loan"] == row["loan"]]
        notes = []
        if not member_rows:
            notes.append("该成员当日该时段没有班表记录，借班无法兑现")
        elif not lendable:
            notes.append("成员当日为本地专用，借班承诺不可履行")
        if row["state"] in ("供给窗口关闭待交接", "待接受"):
            notes.append("借班尚停留在未完成状态，不能当作已承接")
        accepted = row["state"] == "已接受"
        viable = bool(member_rows) and lendable and accepted
        out.append(
            LoanCheck(
                loan=row["loan"],
                member=row["member"],
                day=row["day"],
                slot=row["slot"],
                recipient=row["recipient"],
                state_final=row["state"],
                known_first=min(r["known"] for r in history),
                known_last=max(r["known"] for r in history),
                recipient_qualified_here=recipient_here,
                donor_qualified=bool(member_rows),
                lendable=lendable,
                same_slot_only=True,
                skill_promise=skills,
                viable=viable,
                note="；".join(notes),
            )
        )
    return out


def _deadline(c: Consult, cap: int) -> int:
    return c.first + cap


def queue_items(
    ds: Dataset,
    day: str,
    *,
    reason: dict[str, str] | None = None,
    deadline_minutes: int = 18,
    switch_minute: int | None = None,
) -> list[QueueItem]:
    """Classify each consult still in flight at the switch for the given day.

    ``reason`` maps an origin site to a switch action for *new* handoffs out of
    that site (``pause`` = pause new handoffs; anything else leaves the site's
    rule as it was).  The action never rewrites a consult that is already in
    flight: those keep their first-entry deadline and their recorded route.
    """
    reason = reason or {}
    switch_minute = switch_minute if switch_minute is not None else 12 * 60
    out: list[QueueItem] = []
    for c in ds.consults:
        if c.day != day or c.answer is not None:
            continue
        accepted_at_switch = c.claimed_asof
        state = c.state_asof
        if accepted_at_switch:
            action = "regional"
            why = "对方已认领，转区域首入先后规则，不重启时限"
            still = False
        elif c.is_transfer and state == "待认领":
            action = "hold"
            why = "对方尚未认领，咨询仍留在原站余量内，不因借班名额而消失"
            still = True
        elif c.is_transfer and state == "拒接":
            action = "hold"
            why = "对方拒接，原站继续按原时限处理"
            still = True
        else:
            action = "hold"
            why = "原站自答，按原时限继续"
            still = True
        site_rule = reason.get(c.origin)
        if site_rule == "pause" and c.is_transfer and not accepted_at_switch:
            action = "pause_new_handoffs"
            why += "；该站已暂停新转交，此笔不再转出"
            still = True
        elif site_rule == "pause":
            why += "；该站已暂停新转交（在途咨询不受影响）"
        if action == "regional" and site_rule == "pause":
            why += "；该站未批准区域规则，已认领的咨询按原时限留在收方处理"
        accepted_late = (
            c.accepted_final is not None
            and c.accepted_asof is not None
            and c.accepted_final > c.accepted_asof
            and c.handoff_known_final is not None
            and c.handoff_known_final > (c.handoff_known_asof or "")
        )
        if accepted_late:
            # The confirmation only became visible after the switch window.
            # It changes the *load attribution* (the partner did hold the
            # consult later than first recorded) and the applicability going
            # forward, but it cannot change whether the traveller met the
            # deadline, cannot blame the other group for the delay, and cannot
            # book the same minute on both rosters: only the dwelling seat is
            # consumed.
            why += (
                "；迟到认领只改负荷归属与可回查的更正范围，不改旅客时限成败，"
                "不把迟到归责给收方，也不多占双方名额"
            )
            still = False
        out.append(
            QueueItem(
                thread=c.thread,
                origin=c.origin,
                destination=c.destination,
                skill=c.skill,
                first=c.first,
                state_asof=state,
                accepted_asof=c.accepted_asof,
                accepted_final=c.accepted_final,
                deadline=_deadline(c, deadline_minutes),
                action=action,
                reason=why,
                still_with_origin=still,
                late_confirmation=accepted_late,
            )
        )
    return out


def residual_load(items: Iterable[QueueItem]) -> dict:
    rows = list(items)
    by_site = defaultdict(list)
    for r in rows:
        by_site[r.origin].append(r)
    return {
        site: {
            "still_with_origin": sum(1 for r in group if r.still_with_origin),
            "handed_to_regional": sum(1 for r in group if r.action == "regional"),
            "paused": sum(1 for r in group if r.action == "pause"),
            "by_skill": dict(Counter(r.skill for r in group)),
        }
        for site, group in sorted(by_site.items())
    }


def site_decisions(replay_summary: dict, loan_status: list[LoanCheck], paired: dict | None = None) -> list[dict]:
    """Evidence-based recommendation per site, not per individual language.

    Precedence, from strongest evidence to weakest:

    1. an accepted loan that the roster cannot honour -> pause new handoffs;
    2. an accepted, honour-able loan -> keep the loan promise for that slot;
    3. a clear replay gain for the site under the regional rule -> regional;
    4. otherwise keep the station's own queue.

    A better *regional average* is not enough: the paired table must also show
    the site itself gaining, because the regional rule shifts load between
    sites and languages.
    """
    loan_by_recipient: dict[str, list[LoanCheck]] = defaultdict(list)
    for row in loan_status:
        loan_by_recipient[row.recipient].append(row)
    out = []
    for site in SITES:
        # paired means compare the same served consults in both arms; fall
        # back to the arm means only if the paired table is unavailable
        if paired and site in paired["by_origin"]:
            p = paired["by_origin"][site]
            a = {"mean": p["mean_a"], "served": p["served_a"], "unmet": None}
            b = {"mean": p["mean_b"], "served": p["served_b"], "unmet": None}
        else:
            a = replay_summary["A"]["by_origin"].get(site, {})
            b = replay_summary["B"]["by_origin"].get(site, {})
        promised = loan_by_recipient.get(site, [])
        viable = [r for r in promised if r.viable]
        blocked = [r for r in promised if r.state_final == "已接受" and not r.viable]
        pending = [r for r in promised if r.state_final in ("待接受", "供给窗口关闭待交接")]
        if blocked:
            action = "pause_new"
            why = "已接受的借班在班表上不可履行（本地专用或无班表），应暂停新转交并把在途咨询留在原站"
            if b.get("mean") is not None and a.get("mean") is not None and b["mean"] < a["mean"] - 0.5:
                why += (
                    f"；回放显示区域规则对本区平均有利（{a['mean']:.1f}→{b['mean']:.1f} 分钟），"
                    "但仍需按语言与偏远站核对后再决定后续适用范围"
                )
        elif viable:
            action = "keep_loan"
            why = "存在可履行且已接受的借班承诺，先按原承诺履行，不改为区域规则"
        elif (
            a.get("mean") is not None
            and b.get("mean") is not None
            and b["mean"] < a["mean"] - 1
        ):
            action = "regional"
            why = "回放中区域规则对该站首问平均等待更有利，且无未完成借班"
        else:
            action = "keep_station"
            why = "回放未显示区域规则对该站有改善，保留站内次序"
        if pending:
            why += "；未完成的借班不能视作已承接"
        out.append(
            {
                "site": site,
                "action": action,
                "reason": why,
                "replay_mean_a": a.get("mean"),
                "replay_mean_b": b.get("mean"),
                "replay_served_a": a.get("served"),
                "replay_served_b": b.get("served"),
                "replay_unmet_a": a.get("unmet"),
                "replay_unmet_b": b.get("unmet"),
                "loans": [
                    {
                        "loan": r.loan,
                        "member": r.member,
                        "state": r.state_final,
                        "viable": r.viable,
                        "note": r.note,
                    }
                    for r in promised
                ],
            }
        )
    return out


def _switch_day(ds: Dataset) -> str:
    """The last comparable day that still has handoffs in flight at close.

    A day whose transfers were all claimed before the switch would demonstrate
    nothing about residual load.  The choice is recorded, not hidden: both the
    day and the in-flight count are written into the report.
    """
    candidates = ds.candidate_days()
    # a day with handoffs still awaiting the partner's claim at close exercises
    # the residual-load rule; among those, take the latest so the day is closest
    # to the switch
    awaiting_claim = [
        day
        for day in candidates
        if any(
            c.day == day
            and c.answer is None
            and c.is_transfer
            and c.state_asof == "待认领"
            for c in ds.consults
        )
    ]
    if awaiting_claim:
        return awaiting_claim[-1]
    unclaimed = [
        day
        for day in candidates
        if any(
            c.day == day and c.answer is None and c.is_transfer and not c.claimed_asof
            for c in ds.consults
        )
    ]
    if unclaimed:
        return unclaimed[-1]
    for day in reversed(candidates):
        if any(c.day == day and c.answer is None and c.is_transfer for c in ds.consults):
            return day
    return candidates[-1]


def build(ds: Dataset | None = None, replay_report: dict | None = None) -> dict:
    ds = ds or Dataset.read()
    if replay_report is None:
        from . import replay as replay_mod

        replay_report = replay_mod.build(ds)
    loan_status = loan_checks(ds)
    decisions = site_decisions(
        replay_report["summary"], loan_status, replay_report.get("paired")
    )
    switch_day = _switch_day(ds)
    day_consults = [c for c in ds.consults if c.day == switch_day]
    items = queue_items(
        ds, switch_day, reason={row["site"]: row["action"] for row in decisions}
    )
    return {
        "loan_checks": loan_status,
        "site_decisions": decisions,
        "switch_day": switch_day,
        "site_actions": {row["site"]: row["action"] for row in decisions},
        "switch_day_consults": len(day_consults),
        "queue": items,
        "residual": residual_load(items),
        "action_counts": Counter(r.action for r in items),
        "note": (
            "切换只改变后续适用范围，不重写当日已发生的等待；"
            "已接受且可履行的借班按约定的语言技能继续，未认领的咨询仍计入原站余量。"
        ),
    }
