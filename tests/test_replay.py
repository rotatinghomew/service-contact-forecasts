"""Promise and switch invariants: caps, replays, and the live switch queue."""
import unittest
from collections import Counter

from analysis import guarantees, replay, switch, waits
from analysis.common import Dataset


class GuaranteeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.read()
        cls.report = guarantees.build(cls.ds)

    def test_every_scope_uses_demand_and_answered_side_by_side(self):
        for a in self.report["assessments"]:
            self.assertEqual(a.demand, a.answered + sum(
                1 for c in guarantees.scope_consults(self.ds, {
                    "site": a.scope, "skill": a.skill, "slot": a.slot
                }) if c.answer is None
            ))
            self.assertLessEqual(a.share_of_demand, a.share_among_answered)

    def test_unseen_never_counted_as_compliant(self):
        for a in self.report["assessments"]:
            self.assertLessEqual(a.compliant_answered, a.answered)
            self.assertLessEqual(a.compliant_answered, a.demand)

    def test_known_failing_scope(self):
        by_key = {(a.scope, a.skill, a.slot): a for a in self.report["assessments"]}
        port_english = by_key[("港口", "英语", 3)]
        self.assertLess(port_english.share_of_demand_ci[1], port_english.goal)

    def test_answered_share_hides_failures(self):
        by_key = {(a.scope, a.skill, a.slot): a for a in self.report["assessments"]}
        old = by_key[("老城馆", "普通话", 2)]
        self.assertTrue(old.share_among_answered >= old.goal)
        self.assertLess(old.share_of_demand_ci[1], old.goal)


class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.read()
        cls.report = replay.build(cls.ds)

    def test_both_rules_see_the_same_arrivals(self):
        a = self.report["summary"]["A"]["overall"]
        b = self.report["summary"]["B"]["overall"]
        self.assertEqual(a["consults"], b["consults"])

    def test_paired_totals_match(self):
        p = self.report["paired"]
        self.assertEqual(p["paired"] + p["a_only"] + p["b_only"] + p["neither"], p["consults"])

    def test_no_old_departure_outcome_is_reused(self):
        for row in self.report["rows"]:
            self.assertNotIn("departed", row)
            self.assertNotIn("callback", row)

    def test_unserved_keeps_a_bounded_interval(self):
        arm = replay.replay(self.ds)
        unmet = [c for c in arm["B"].cases if not c.served]
        self.assertTrue(unmet)
        for c in unmet:
            self.assertGreaterEqual(c.ceiling, c.floor)

    def test_local_only_seats_do_not_shorten_foreign_queues(self):
        """Regional rule must not use a 本地专用 seat for a consult first
        entered elsewhere unless that site claimed it."""
        arm = replay.replay(self.ds)["B"]
        for c in arm.cases:
            if not c.served:
                continue
            ep = c.episode
            assert ep is not None
            seats = self.ds.qualified_seats(ep.day, ep.slot, ep.site, ep.skill)
            if not seats:
                continue
            lendable = self.ds.lendable_seats(ep.day, ep.slot, ep.site, ep.skill)
            if lendable:
                continue
            self.assertTrue(
                c.consult.origin == ep.site
                or (c.consult.claimed_asof and c.consult.destination == ep.site),
                c.consult.thread,
            )

    def test_cap_table_reports_undecided(self):
        for cap, row in self.report["cap_table"].items():
            for rule, classes in row["per_rule"].items():
                for cls, v in classes.items():
                    self.assertEqual(
                        v["definitely_within"] + v["definitely_over"] + v["undecided"], v["n"]
                    )

    def test_uncertainty_can_flip_a_verdict_so_rule_change_is_not_approved(self):
        flips = [row["verdict_flips_at_80pct"] for row in self.report["cap_table"].values()]
        self.assertTrue(any(flips))
        self.assertTrue(any(row["comparison_only"] for row in self.report["cap_table"].values()))


class SwitchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.read()
        cls.report = switch.build(cls.ds)

    def test_loan_promises_checked_against_roster(self):
        for row in self.report["loan_checks"]:
            if row.lendable and row.state_final == "已接受":
                self.assertTrue(row.viable)
            if not row.lendable:
                self.assertFalse(row.viable)

    def test_local_only_member_cannot_be_lent(self):
        l1 = next(r for r in self.report["loan_checks"] if r.loan == "L1")
        self.assertFalse(l1.lendable)
        self.assertFalse(l1.viable)

    def test_unclaimed_handoffs_stay_with_origin(self):
        for item in self.report["queue"]:
            if item.state_asof == "待认领":
                self.assertTrue(item.still_with_origin)
                self.assertNotEqual(item.action, "regional")

    def test_every_inflight_keeps_first_entry_deadline(self):
        for item in self.report["queue"]:
            self.assertGreater(item.deadline, item.first)

    def test_corrections_do_not_relabel_deadline_success(self):
        corrected = [
            item for item in self.report["queue"]
            if item.accepted_final is not None
            and item.accepted_asof is not None
            and item.accepted_final > item.accepted_asof
        ]
        for item in corrected:
            self.assertNotEqual(item.action, "regional")
            self.assertTrue(item.still_with_origin)

    def test_residual_load_explains_whole_region(self):
        residual = self.report["residual"]
        total = sum(v["still_with_origin"] + v["handed_to_regional"] for v in residual.values())
        self.assertEqual(total, len(self.report["queue"]))


if __name__ == "__main__":
    unittest.main()
