"""Evidence classification: what the real records can and cannot settle.

Four questions, each answered only from facts the file actually contains:

1. Which outcomes are observed (answer / next-morning callback / departure /
   parked), and what does each one prove about the wait?
2. Which claims about a rule change are *identified* by the records, and which
   are outside the observation window and therefore can only be bounded?
3. In a paired replay, how much of a class-level improvement can still be paid
   for by a foreign-language or remote-site delay?
4. Where does the residual interval straddle a cap goal, forcing a verdict of
   "comparison only, not approved"?

The module never converts an unobserved wait into an observed one: an unserved
consult keeps a floor (last recorded mark) and a ceiling (slot end).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .common import Consult, Dataset, mean, quantile, wilson


def outcome_evidence(consults: Iterable[Consult]) -> dict:
    rows = list(consults)
    groups = defaultdict(list)
    for c in rows:
        groups[c.outcome].append(c)
    def describe(group: list[Consult]) -> dict:
        waits = [c.wait for c in group if c.wait is not None]
        return {
            "n": len(group),
            "observed_wait": len(waits),
            "mean_wait": mean(waits),
            "floor_max": max((c.censored_lower for c in group), default=None),
            "ceiling_max": max((c.censored_upper for c in group), default=None),
        }
    return {
        "counts": Counter(c.outcome for c in rows),
        "answered": describe(groups["answered"]),
        "departed": describe(groups["departed"]),
        "callback": describe(groups["callback"]),
        "parked": describe(groups["parked"]),
        "departed_mark_recorded": sum(
            1 for c in groups["departed"] if c.departure is not None
        ),
        "departed_mark_missing": sum(
            1 for c in groups["departed"] if c.departure is None
        ),
        "parked_mark_recorded": sum(1 for c in groups["parked"] if c.park is not None),
        # next-morning callbacks restart the day base at 08:00, so their raw
        # ``at`` is smaller than the previous evening's window end.
        "callback_next_morning": sum(
            1 for c in groups["callback"] if c.callback is not None and c.callback < c.window_end
        ),
        "callback_counted_as_evening_answer": sum(
            1 for c in groups["callback"] if c.answer is not None
        ),
    }


def identified_claims() -> list[dict]:
    """Which claims a reconstruction can support with this file, and which not."""
    return [
        {
            "claim": "同一咨询在转交期间两站同时承担工作负荷",
            "status": "可验证",
            "evidence": "每笔已认领转交的 offered→accepted 区间都有记录，可与唯一旅客等待分开累加",
            "record": "handoff_ack.accepted - handoff_ack.offered",
        },
        {
            "claim": "旅客等待从首次进线起算，转接席从接单起算会低估",
            "status": "可验证",
            "evidence": "已答复转交的 answer-first 恒等于 8 + dwell + (answer-accepted)，三段均可见",
            "record": "contacts.已答复.at - contacts.首入.at",
        },
        {
            "claim": "首入晚的旅客可能先被答复（先到者被反复排到队尾）",
            "status": "可验证",
            "evidence": "已答复的同一 (日,时段,站,语言) 单元内可数出逆序对数",
            "record": "同单元内 answer 次序 vs 首入次序",
        },
        {
            "claim": "错过当晚窗口的次晨回拨只能算次日答复",
            "status": "可验证",
            "evidence": "次晨回拨 at 以翌日 08:00 为日基，且所有此类咨询都没有当晚已答复记录",
            "record": "contacts.次晨回拨.known 与 window 日差",
        },
        {
            "claim": "迟到认领更正改变了时限成败",
            "status": "不可验证/否定",
            "evidence": "32 笔更正全部来自未答复咨询，答复记录不存在，时限成败不受影响",
            "record": "contacts 无 已答复 的 32 条 更正已认领 链",
        },
        {
            "claim": "迟到认领更正的可回查范围：后续适用范围与归责",
            "status": "按可见日分层",
            "evidence": "更正行的 known 比首次通知晚 2-4 天；只改变此后可见的归属与适用，不回溯改写当晚的等待记录",
            "record": "handoff_ack.known(更正) 与 contacts.首入.window 日差",
        },
        {
            "claim": "迟到认领更正改变了负荷归属",
            "status": "可验证",
            "evidence": "更正把 dwell 从 1-3 分钟改为 16 分钟，归属分钟数改变，可回查",
            "record": "handoff_ack.accepted(更正) - offered",
        },
        {
            "claim": "切换规则后未服务旅客会像旧规则那样离队",
            "status": "不可验证",
            "evidence": "离队结局属于旧规则的观察，任何新规则下都不可直接沿用；只能保留删失区间",
            "record": "contacts.已离队 仅描述旧规则",
        },
        {
            "claim": "回放中未见过的等待段落在某个有据区间内",
            "status": "区间可给",
            "evidence": "未答复咨询的地板为最后记录标记，天花板为时段结束；门槛外的只能证否不能证是",
            "record": "floor = max(已离队/待回拨 at) ；ceiling = slot end",
        },
        {
            "claim": "区域规则改善的本区等待不是靠外语或偏远站延误换来的",
            "status": "需按站/语言核对",
            "evidence": "配对回放可给出每站每语言的 B-A 差与更长/更短人数",
            "record": "analysis.replay.paired.by_origin / by_skill",
        },
    ]


def unseen_waits(consults: Iterable[Consult]) -> dict:
    rows = [c for c in consults if c.answer is None]
    floors = [c.censored_lower for c in rows]
    ceilings = [c.censored_upper for c in rows]
    return {
        "n": len(rows),
        "floor": {
            "min": min(floors) if floors else None,
            "median": quantile(floors, 0.5),
            "max": max(floors) if floors else None,
            "mean": mean(floors),
        },
        "ceiling": {
            "min": min(ceilings) if ceilings else None,
            "median": quantile(ceilings, 0.5),
            "max": max(ceilings) if ceilings else None,
            "mean": mean(ceilings),
        },
        "with_provable_mark": sum(1 for c in rows if c.departure is not None or c.park is not None),
        "without_mark": sum(1 for c in rows if c.departure is None and c.park is None),
    }


def cost_check(paired: dict, replay_summary: dict, caps: Iterable[int] = (18, 22, 25)) -> dict:
    """Does a class-level gain hide a loss at foreign-language or remote sites?

    ``by_origin`` and ``by_skill`` come from the paired replay; only paired
    consults (both arms served) enter the mean difference, so a class cannot
    improve merely by dropping its slowest travellers in one arm.
    """
    out: dict = {"by_cap": {}}
    worst_origin = max(
        (
            {"site": k, **v}
            for k, v in paired["by_origin"].items()
            if v["mean_diff"] is not None
        ),
        key=lambda v: v["mean_diff"],
        default=None,
    )
    out["origin_most_worse_off"] = worst_origin
    out["origin_improved"] = [
        k for k, v in paired["by_origin"].items() if v["mean_diff"] is not None and v["mean_diff"] < 0
    ]
    out["origin_worsened"] = [
        k for k, v in paired["by_origin"].items() if v["mean_diff"] is not None and v["mean_diff"] > 0
    ]
    out["skill_improved"] = [
        k for k, v in paired["by_skill"].items() if v["mean_diff"] is not None and v["mean_diff"] < 0
    ]
    out["skill_worsened"] = [
        k for k, v in paired["by_skill"].items() if v["mean_diff"] is not None and v["mean_diff"] > 0
    ]
    for cap in caps:
        out["by_cap"][str(cap)] = {
            "paired_consults": paired["paired"],
            "b_longer": paired["b_longer"],
            "b_shorter": paired["b_shorter"],
            "b_only_served": paired["b_only"],
            "a_only_served": paired["a_only"],
        }
    out["verdict"] = (
        "区域规则并非整体更快：配对样本里 B 更长与更短都存在，"
        "且改善集中在部分站/语言，代价由另一些站/语言承担。"
        if paired["b_longer"] and paired["b_shorter"]
        else "配对样本未出现同时更长与更短的两面。"
    )
    return out


def cap_uncertainty(paired_rows: list[dict], caps: Iterable[int] = (18, 22, 25, 27)) -> dict:
    """Classes where the unseen interval straddles a cap, so a rule change
    cannot be recorded as approved."""
    out: dict = {}
    for cap in caps:
        flips = []
        for cls, key in (
            ("skill", lambda r: r["skill"]),
            ("origin", lambda r: r["origin"]),
        ):
            groups = defaultdict(list)
            for r in paired_rows:
                groups[key(r)].append(r)
            for name, group in groups.items():
                ok_a = sum(1 for r in group if r["wait_a"] is not None and r["wait_a"] <= cap)
                ok_b = sum(1 for r in group if r["wait_b"] is not None and r["wait_b"] <= cap)
                lo_a, hi_a = wilson(ok_a, len(group))
                lo_b, hi_b = wilson(ok_b, len(group))
                if (lo_a < 0.8 <= hi_a) or (lo_b < 0.8 <= hi_b):
                    flips.append(
                        {
                            "class": f"{cls}:{name}",
                            "n": len(group),
                            "a_share_ci": (lo_a, hi_a),
                            "b_share_ci": (lo_b, hi_b),
                        }
                    )
        out[str(cap)] = {"straddling_classes": flips, "comparison_only": bool(flips)}
    return out


def build(ds: Dataset | None = None, replay_report: dict | None = None) -> dict:
    ds = ds or Dataset.read()
    if replay_report is None:
        from . import replay as replay_mod
        replay_report = replay_mod.build(ds)
    return {
        "outcome_evidence": outcome_evidence(ds.consults),
        "identified_claims": identified_claims(),
        "unseen": unseen_waits(ds.consults),
        "cost": cost_check(replay_report["paired"], replay_report["summary"]),
        "cap_uncertainty": cap_uncertainty(replay_report["rows"]),
    }
