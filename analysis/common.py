"""Shared loaders and consult-chain reconstruction.

The unit of analysis is one *consult* (one ``thread``).  A consult enters once
(``首入``) and may be handed off at most once.  Later records (answer, callback,
departure, parked-for-callback) belong to the same consult and never create a
second traveller or a second clock.

All times are local minutes after midnight.  The four service slots start at
08:00, 11:00, 14:00 and 17:00 and are 180 minutes long.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SITES = ("港口", "滨海馆", "芦湾酒店", "老城馆")
SKILLS = ("普通话", "英语", "粤语")
SLOT_START = (480, 660, 840, 1020)
SLOT_MINUTES = 180
SLOT_END = tuple(s + SLOT_MINUTES for s in SLOT_START)

#: Answers are recorded up to a quarter of an hour after the nominal slot end
#: (the latest recorded answer is 12:07 into the next nominal slot).  This is
#: the only grace used when bounding an unseen wait; it is not a promise.
ANSWER_GRACE = 15

#: Service minutes used for a consult whose own duration was never recorded.
#: The medians are taken from the recorded answers (Mandarin is longer because
#: it includes the interpreter/coordination overhead in this data set).
DURATION_FALLBACK = {"普通话": 8.0, "英语": 6.0, "粤语": 6.0}

OUTCOME_LABELS = {
    "answered": "已答复",
    "callback": "次晨回拨",
    "departed": "已离队",
    "parked": "待回拨",
}


def load(name: str) -> list[dict[str, Any]]:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def slot_of(window: str) -> int:
    return int(window.rsplit("-", 1)[1])


def day_of(window: str) -> str:
    return window[:10]


def slot_end(slot: int) -> int:
    return SLOT_END[slot]


@dataclass(frozen=True)
class Consult:
    thread: str
    window: str
    day: str
    slot: int
    origin: str
    skill: str
    first: int
    transfer_offered: int | None = None
    destination: str | None = None
    answer: int | None = None
    answer_site: str | None = None
    duration: float | None = None
    departure: int | None = None
    departure_recorded: bool = False
    park: int | None = None
    park_recorded: bool = False
    callback: int | None = None
    callback_duration: float | None = None
    state_asof: str | None = None
    state_final: str | None = None
    accepted_asof: int | None = None
    accepted_final: int | None = None
    handoff_known_asof: str | None = None
    handoff_known_final: str | None = None
    covered: bool = True
    coverage_rate: float | None = None
    outcome: str = "pending"

    # -- derived convenience -------------------------------------------------
    @property
    def claimed_asof(self) -> bool:
        return (
            self.destination is not None
            and self.accepted_asof is not None
            and self.state_asof in ("已认领", "更正已认领")
        )

    @property
    def claimed_final(self) -> bool:
        return (
            self.destination is not None
            and self.accepted_final is not None
            and self.state_final in ("已认领", "更正已认领")
        )

    @property
    def wait(self) -> int | None:
        """Consult-level wait: answer time minus first entry (one clock)."""
        if self.answer is None:
            return None
        return self.answer - self.first

    @property
    def window_end(self) -> int:
        return slot_end(self.slot)

    @property
    def recorded_window_end(self) -> int:
        """Latest minute an answer was ever recorded in this slot."""
        return slot_end(self.slot) + ANSWER_GRACE

    @property
    def pre_offer(self) -> int | None:
        if self.transfer_offered is None:
            return None
        return self.transfer_offered - self.first

    @property
    def dwell_asof(self) -> int | None:
        if self.transfer_offered is None or self.accepted_asof is None:
            return None
        return self.accepted_asof - self.transfer_offered

    @property
    def dwell_final(self) -> int | None:
        if self.transfer_offered is None or self.accepted_final is None:
            return None
        return self.accepted_final - self.transfer_offered

    @property
    def dest_service_asof(self) -> int | None:
        if self.answer is None or self.accepted_asof is None:
            return None
        return self.answer - self.accepted_asof

    @property
    def origin_resume(self) -> int | None:
        """For an unclaimed handoff later answered by the origin."""
        if self.answer is None or self.transfer_offered is None:
            return None
        if self.accepted_asof is not None:
            return None
        return self.answer - self.transfer_offered

    @property
    def is_transfer(self) -> bool:
        return self.transfer_offered is not None

    @property
    def censored_lower(self) -> int:
        """Provable minimum wait for a consult without an in-window answer."""
        if self.answer is not None:
            return self.answer - self.first
        if self.departure is not None:
            return self.departure - self.first
        if self.park is not None:
            return self.park - self.first
        return 0

    @property
    def censored_upper(self) -> int:
        if self.answer is not None:
            return self.answer - self.first
        return max(0, self.window_end + ANSWER_GRACE - self.first)


@dataclass
class Dataset:
    contacts: list[dict[str, Any]]
    handoff: list[dict[str, Any]]
    coverage: list[dict[str, Any]]
    dispatch: list[dict[str, Any]]
    teams: list[dict[str, Any]]
    guarantees: list[dict[str, Any]]
    loans: list[dict[str, Any]]
    replay: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self.coverage_by_window = {r["window"]: r for r in self.coverage}
        self.replay_by_day = {r["day"]: r for r in self.replay}
        self.contacts_by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in self.contacts:
            self.contacts_by_thread[r["thread"]].append(r)
        self.handoff_by_thread: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in self.handoff:
            self.handoff_by_thread[r["thread"]].append(r)
        self._consults: list[Consult] | None = None
        self._by_thread: dict[str, Consult] | None = None

    # -- constructors --------------------------------------------------------
    @classmethod
    def read(cls) -> "Dataset":
        return cls(
            contacts=load("contacts"),
            handoff=load("handoff_ack"),
            coverage=load("coverage"),
            dispatch=load("dispatch_calendar"),
            teams=load("teams"),
            guarantees=load("guarantees"),
            loans=load("booked_loans"),
            replay=load("replay_windows"),
        )

    # -- chain reconstruction ------------------------------------------------
    def build_consults(self) -> list[Consult]:
        if self._consults is not None:
            return self._consults
        out: list[Consult] = []
        for thread, events in self.contacts_by_thread.items():
            firsts = [e for e in events if e["kind"] == "首入"]
            if len(firsts) != 1:
                raise ValueError(f"{thread}: expected exactly one 首入, got {len(firsts)}")
            first = firsts[0]
            if len({e["window"] for e in events}) != 1:
                raise ValueError(f"{thread}: events span more than one window")
            if len({e["skill"] for e in events}) != 1:
                raise ValueError(f"{thread}: events span more than one skill")
            transfers = [e for e in events if e["kind"] == "转接"]
            if len(transfers) > 1:
                raise ValueError(f"{thread}: more than one transfer")
            answers = [e for e in events if e["kind"] == "已答复"]
            callbacks = [e for e in events if e["kind"] == "次晨回拨"]
            departures = [e for e in events if e["kind"] == "已离队"]
            parks = [e for e in events if e["kind"] == "待回拨"]
            for name, rows in (
                ("已答复", answers),
                ("次晨回拨", callbacks),
                ("已离队", departures),
                ("待回拨", parks),
            ):
                if len(rows) > 1:
                    raise ValueError(f"{thread}: more than one {name}")

            transfer = transfers[0] if transfers else None
            answer = answers[0] if answers else None
            callback = callbacks[0] if callbacks else None
            departure = departures[0] if departures else None
            park = parks[0] if parks else None

            if answer is not None:
                outcome = "answered"
            elif callback is not None:
                outcome = "callback"
            elif departure is not None:
                outcome = "departed"
            elif park is not None:
                outcome = "parked"
            else:
                outcome = "pending"

            handoff_rows = self.handoff_by_thread.get(thread, [])
            originals = [r for r in handoff_rows if r["state"] != "更正已认领"]
            corrections = [r for r in handoff_rows if r["state"] == "更正已认领"]
            if len(originals) > 1:
                raise ValueError(f"{thread}: more than one original handoff row")
            original = originals[0] if originals else None
            correction = corrections[0] if corrections else None
            handoff = correction or original
            if handoff is not None:
                if transfer is None:
                    raise ValueError(f"{thread}: handoff without a transfer event")
                if handoff["origin"] != first["site"]:
                    raise ValueError(f"{thread}: handoff origin differs from first-entry site")
                if handoff["destination"] != transfer["site"]:
                    raise ValueError(f"{thread}: handoff destination differs from transfer event")
                if handoff["offered"] != transfer["at"]:
                    raise ValueError(f"{thread}: handoff offered differs from transfer event")

            cov = self.coverage_by_window[first["window"]]
            out.append(
                Consult(
                    thread=thread,
                    window=first["window"],
                    day=day_of(first["window"]),
                    slot=slot_of(first["window"]),
                    origin=first["site"],
                    skill=first["skill"],
                    first=first["at"],
                    transfer_offered=transfer["at"] if transfer else None,
                    destination=handoff["destination"] if handoff else None,
                    answer=answer["at"] if answer else None,
                    answer_site=answer["site"] if answer else None,
                    duration=float(answer["duration"]) if answer and answer["duration"] is not None else (
                        float(callback["duration"]) if callback and callback["duration"] is not None else None
                    ),
                    departure=departure["at"] if departure and departure["at"] is not None else None,
                    departure_recorded=departure is not None,
                    park=park["at"] if park and park["at"] is not None else None,
                    park_recorded=park is not None,
                    callback=callback["at"] if callback else None,
                    callback_duration=float(callback["duration"]) if callback and callback["duration"] is not None else None,
                    state_asof=original["state"] if original else None,
                    state_final=handoff["state"] if handoff else None,
                    accepted_asof=original["accepted"] if original else None,
                    accepted_final=handoff["accepted"] if handoff else None,
                    handoff_known_asof=original["known"] if original else None,
                    handoff_known_final=handoff["known"] if handoff else None,
                    covered=bool(cov["covered"]),
                    coverage_rate=cov["estimated_rate"],
                    outcome=outcome,
                )
            )
        out.sort(key=lambda c: (c.day, c.slot, c.first, c.thread))
        self._consults = out
        self._by_thread = {c.thread: c for c in out}
        return out

    @property
    def consults(self) -> list[Consult]:
        return self.build_consults()

    def consult(self, thread: str) -> Consult:
        if self._by_thread is None:
            self.build_consults()
        assert self._by_thread is not None
        return self._by_thread[thread]

    # -- roster helpers ------------------------------------------------------
    def seats(self, day: str, slot: int) -> list[dict[str, Any]]:
        return [t for t in self.teams if t["day"] == day and t["slot"] == slot]

    def qualified_seats(self, day: str, slot: int, site: str, skill: str) -> list[dict[str, Any]]:
        return [
            t
            for t in self.teams
            if t["day"] == day
            and t["slot"] == slot
            and t["home"] == site
            and skill in t["skills"]
        ]

    def lendable_seats(self, day: str, slot: int, site: str, skill: str) -> list[dict[str, Any]]:
        return [t for t in self.qualified_seats(day, slot, site, skill) if t["lend"] == "可借"]

    def reply_cutoff(self, day: str) -> int:
        row = self.replay_by_day.get(day)
        if row is not None:
            return int(row["last_reply"])
        return 1200

    def accept_minutes(self, day: str) -> int | None:
        row = self.replay_by_day.get(day)
        return int(row["accept_minutes"]) if row is not None else None

    def candidate_days(self) -> list[str]:
        return sorted(r["day"] for r in self.replay if r["candidate"])


# -- small statistics helpers ---------------------------------------------


def quantile(values: Iterable[float], p: float) -> float | None:
    data = sorted(float(v) for v in values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def mean(values: Iterable[float]) -> float | None:
    data = [float(v) for v in values]
    return sum(data) / len(data) if data else None


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial share."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def band(wait: int | None, *, unseen: bool = False) -> str:
    if unseen or wait is None:
        return "未见答复"
    if wait <= 5:
        return "0-5"
    if wait <= 10:
        return "6-10"
    if wait <= 15:
        return "11-15"
    if wait <= 20:
        return "16-20"
    if wait <= 25:
        return "21-25"
    if wait <= 29:
        return "26-29"
    return "30+"


def outcome_counter(consults: Iterable[Consult]) -> Counter[str]:
    return Counter(c.outcome for c in consults)
