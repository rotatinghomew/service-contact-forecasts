#!/usr/bin/env python3
"""Build every report from the local data files.

Usage::

    python3 scripts/run_analysis.py            # writes outputs/
    python3 scripts/run_analysis.py --print    # writes outputs/ and prints a summary

All numbers are computed from ``data/``; nothing is hand-copied.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import evidence, guarantees, language, replay, switch, waits  # noqa: E402
from analysis.common import Dataset  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs"


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in obj.__dict__.items()}
    return obj


def build() -> dict:
    ds = Dataset.read()
    waits_report = waits.build(ds)
    guarantee_report = guarantees.build(ds)
    replay_report = replay.build(ds)
    switch_report = switch.build(ds, replay_report)
    evidence_report = evidence.build(ds, replay_report)
    language_report = language.build(ds)
    return {
        "waits": waits_report,
        "guarantees": guarantee_report,
        "replay": replay_report,
        "switch": switch_report,
        "evidence": evidence_report,
        "language": language_report,
    }


def write(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = {
        "waits": "waits.json",
        "guarantees": "guarantees.json",
        "replay": "replay.json",
        "switch": "switch.json",
        "evidence": "evidence.json",
        "language": "language.json",
    }
    for key, name in names.items():
        (OUT / name).write_text(
            json.dumps(to_jsonable(report[key]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "chain": report["waits"]["chain_summary"].__dict__,
                "slices": report["waits"]["chain_slices"],
                "dual": report["waits"]["dual_occupancy"],
                "queue": report["waits"]["queue"],
                "censoring": report["waits"]["censoring"],
                "guarantees": [a.__dict__ for a in report["guarantees"]["assessments"]],
                "replay": {
                    rule: report["replay"]["summary"][rule]["overall"]
                    for rule in ("A", "B")
                },
                "replay_paired": {
                    k: v for k, v in report["replay"]["paired"].items() if k != "rows"
                },
                "site_decisions": [
                    {k: v for k, v in row.items() if k != "loans"}
                    for row in report["switch"]["site_decisions"]
                ],
                "queue_actions": report["switch"]["action_counts"],
                "residual": report["switch"]["residual"],
                "evidence": report["evidence"]["identified_claims"],
                "unseen": report["evidence"]["unseen"],
                "language": {
                    "answer_rate": {
                        k: v["answer_rate"] for k, v in report["language"]["survivorship"].items()
                    },
                    "gaps_separated": [
                        g for g in report["language"]["skill_gap"]["gaps"] if g["separated"]
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _fmt(value, digits=2):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def summarise(report: dict) -> str:
    w = report["waits"]
    lines = [
        "== 咨询链（一个旅客、一次从首入起算） ==",
        f"咨询 {w['chain_summary'].n} 笔，有答复 {w['chain_summary'].n - w['chain_summary'].n_unseen} 笔，"
        f"平均等待 {_fmt(w['chain_summary'].mean)} 分钟，中位 {w['chain_summary'].median:g}，"
        f"P90 {w['chain_summary'].p90:g}，范围 {w['chain_summary'].minimum}-{w['chain_summary'].maximum}。",
        f"等待分解（分钟）：首段站内 {w['chain_slices']['origin_before_offer']:.0f}，"
        f"双方同席 {w['chain_slices']['dwell_dual']:.0f}，认领后收方 {w['chain_slices']['destination_after_claim']:.0f}，"
        f"原站续办 {w['chain_slices']['origin_resume']:.0f}，从未转出 {w['chain_slices']['direct_at_origin']:.0f}；"
        f"合计 {w['chain_slices']['components_sum']:.0f} = 链上等待 {w['chain_slices']['chain']:.0f}。",
        f"同时占用两席：{w['dual_occupancy']['consults']} 笔已认领转交，每侧 "
        f"{w['dual_occupancy']['seat_minutes_per_side']} 席位分钟（不进入旅客等待）。",
        f"先到后被越过：{w['queue']['inversions']}/{w['queue']['answered_pairs']} 对（{w['queue']['inversion_rate']:.1%}），"
        f"单元内首位旅客平均等 {_fmt(w['queue']['first_arrival_wait_mean'])} 分钟。",
        f"未答复 {w['censoring']['unresolved']} 笔："
        + "；".join(
            f"上限{a}: 可证超限 {v['definitely_over']}，未定 {v['undecided']}"
            for a, v in w["censoring"]["per_cap"].items()
        ),
        "",
        "== 服务水平（同一组需求与任务） ==",
    ]
    for a in report["guarantees"]["assessments"]:
        lines.append(
            f"{a.scope}/{a.skill}/时段{a.slot} 上限{a.cap}: 需求 {a.demand}，已答复 {a.answered}，"
            f"按需求占比 {a.share_of_demand:.1%}（CI {a.share_of_demand_ci[0]:.1%}-{a.share_of_demand_ci[1]:.1%}），"
            f"仅按已答复 {a.share_among_answered:.1%}，目标 {a.goal:.0%} → {a.verdict}"
        )
    dec = report["guarantees"]["cell_wait_decomposition"]
    lines.append(
        f"单元内直接到答：短等(3-5分) {dec['short_wait']['direct']} 笔，长等(14-29分) {dec['long_wait']['direct']} 笔；"
        f"转接到答短等 {dec['short_wait']['transfer']} 笔、长等 {dec['long_wait']['transfer']} 笔。"
    )
    lines += ["", "== 回放（A=站内次序+临时借班；B=区域首入先后） =="]
    for rule in ("A", "B"):
        o = report["replay"]["summary"][rule]["overall"]
        lines.append(
            f"{rule}: 服务 {o['served']}/{o['consults']}，均值 {_fmt(o['mean'])}，"
            f"中位 {o['median']:g}，P90 {o['p90']:g}，未服务 {o['unmet']}"
        )
    p = report["replay"]["paired"]
    lines.append(
        f"配对 {p['paired']} 笔：B 更长 {p['b_longer']}，更短 {p['b_shorter']}，"
        f"仅A服务 {p['a_only']}，仅B服务 {p['b_only']}；平均差 B-A {p['mean_diff_b_minus_a']:+.2f} 分钟"
    )
    for cls in ("by_skill", "by_origin", "by_slot"):
        lines.append(
            f"  {cls}: "
            + "；".join(
                f"{k} A={_fmt(v['mean_a'],1)} B={_fmt(v['mean_b'],1)} 差{v['mean_diff']:+.1f} 长{v['b_longer']}/短{v['b_shorter']}"
                for k, v in p[cls].items()
                if v["mean_a"] is not None
            )
        )
    for cap, v in report["replay"]["cap_table"].items():
        flips = v["verdict_flips_at_80pct"]
        lines.append(
            f"  上限{cap}: 判定翻转类别 {len(flips)} 个（{', '.join(flips) if flips else '无'}）"
            f"{'，只可比较、不可记为已批准' if v['comparison_only'] else ''}"
        )
    lines += ["", "== 切换建议 =="]
    for row in report["switch"]["site_decisions"]:
        lines.append(f"{row['site']}: {row['action']} — {row['reason']}")
    lines.append(
        f"切换日 {report['switch']['switch_day']} 在途 {len(report['switch']['queue'])} 笔："
        + "，".join(f"{k} {v}" for k, v in report["switch"]["action_counts"].items())
    )
    lines.append("原站余量：" + "；".join(
        f"{site} 留原站 {v['still_with_origin']}、转区域 {v['handed_to_regional']}"
        for site, v in report["switch"]["residual"].items()
    ))
    lines += ["", "== 语言组差距（按需求口径给区间） =="]
    for k, v in report["language"]["survivorship"].items():
        lo, hi = v["of_demand"]["22"]
        lines.append(
            f"{k}: 需求 {v['demand']}，答复率 {v['answer_rate']:.1%}，"
            f"22 分钟内按需求 {lo:.1%}-{hi:.1%}（仅按已答复 {v['among_answered']['22']:.1%}）"
        )
    sep = [g for g in report["language"]["skill_gap"]["gaps"] if g["separated"]]
    lines.append(f"区间互不重叠的语言差距 {len(sep)} 条；其余差距被未答复区间覆盖，只能作观察。")
    lines += ["", "== 证据等级 =="]
    for row in report["evidence"]["identified_claims"]:
        lines.append(f"[{row['status']}] {row['claim']} — {row['evidence']}")
    un = report["evidence"]["unseen"]
    lines.append(
        f"未见等待段 {un['n']} 笔：地板 中位 {un['floor']['median']:g}、最大 {un['floor']['max']}；"
        f"天花板 中位 {un['ceiling']['median']:g}；有可证标记 {un['with_provable_mark']}，无可证标记 {un['without_mark']}。"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", dest="do_print")
    args = parser.parse_args()
    report = build()
    write(report)
    if args.do_print:
        print(summarise(report))


if __name__ == "__main__":
    main()
