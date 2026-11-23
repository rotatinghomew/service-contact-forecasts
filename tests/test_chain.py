"""Business-chain invariants: one traveller, one clock, no silent censoring."""
import unittest

from analysis import waits
from analysis.common import Dataset, SLOT_END


class ChainReconstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.read()
        cls.consults = cls.ds.consults

    def test_one_first_entry_one_transfer_one_answer(self):
        threads = [c.thread for c in self.consults]
        self.assertEqual(len(threads), len(set(threads)))
        for c in self.consults:
            self.assertIsNotNone(c.first)

    def test_answer_never_before_claim(self):
        for c in self.consults:
            if c.answer is not None and c.accepted_asof is not None:
                self.assertGreaterEqual(c.answer, c.accepted_asof, c.thread)

    def test_offer_always_after_eight_minutes(self):
        for c in self.consults:
            if c.transfer_offered is not None:
                self.assertEqual(c.transfer_offered - c.first, 8, c.thread)

    def test_answer_inside_service_slot_with_recorded_grace(self):
        for c in self.consults:
            if c.answer is not None:
                self.assertLessEqual(c.answer, c.recorded_window_end, c.thread)
                self.assertGreaterEqual(c.answer, c.first, c.thread)

    def test_waits_are_chain_waits_not_station_waits(self):
        summary = waits.summarise(self.consults)
        self.assertEqual(summary.n, len(self.consults))
        answered = [c for c in self.consults if c.answer is not None]
        self.assertEqual(summary.n_unseen, len(self.consults) - len(answered))


class DecompositionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.read()
        cls.slices = waits.chain_slices(cls.ds.consults)

    def test_components_add_up_to_chain_wait(self):
        self.assertAlmostEqual(
            self.slices["components_sum"], self.slices["chain"], places=6
        )

    def test_dwell_is_dual_occupancy_and_not_double_counted(self):
        dual = waits.dual_occupancy(self.ds.consults)
        answered_claimed = [
            c
            for c in self.ds.consults
            if c.claimed_asof and c.answer is not None
        ]
        self.assertEqual(
            sum(c.dwell_asof for c in answered_claimed), self.slices["dwell_dual"]
        )
        self.assertEqual(dual["dual_seat_minutes_all_sites"], 2 * dual["seat_minutes_per_side"])
        self.assertLessEqual(self.slices["dwell_dual"], dual["seat_minutes_per_side"])
        # the traveller's clock counts the dwell once, the two sites hold the
        # same minutes concurrently
        self.assertLess(self.slices["dwell_dual"], self.slices["chain"])

    def test_unclaimed_transfer_is_credited_to_origin_resume(self):
        unclaimed = [
            c
            for c in self.ds.consults
            if c.transfer_offered is not None and c.accepted_asof is None and c.answer is not None
        ]
        self.assertTrue(unclaimed)
        for c in unclaimed:
            self.assertIsNone(c.dest_service_asof)
            self.assertEqual(
                c.origin_resume, c.answer - c.transfer_offered
            )


class CensoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.read()
        cls.report = waits.censoring(cls.ds.consults)

    def test_unresolved_stays_demand(self):
        total = self.report["total"]
        answered = self.report["by_outcome"]["answered"]
        self.assertEqual(self.report["unresolved"], total - answered)
        self.assertGreater(self.report["unresolved"], 0)

    def test_departure_never_removes_demand(self):
        self.assertGreater(self.report["departed_still_demand"], 0)
        self.assertGreater(self.report["parked_still_demand"], 0)

    def test_next_morning_callback_not_booked_back(self):
        self.assertEqual(self.report["callback_with_evening_answer"], 0)
        self.assertEqual(
            self.report["callback_consults"],
            self.report["callback_next_morning_base"],
        )

    def test_cap_verdicts_are_only_proved_from_floors(self):
        for cap, row in self.report["per_cap"].items():
            self.assertGreaterEqual(row["definitely_over"], 0)
            self.assertLessEqual(
                row["definitely_over"] + row["undecided"] + row["provably_within"],
                self.report["unresolved"],
            )

    def test_ceiling_is_slot_end_plus_recorded_grace(self):
        for c in self.ds.consults:
            if c.answer is None:
                self.assertLessEqual(c.censored_upper, c.recorded_window_end - c.first)


class OrderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = waits.queue_placement(Dataset.read().consults)

    def test_inversions_are_measured_not_assumed(self):
        self.assertGreater(self.report["answered_pairs"], 0)
        self.assertGreaterEqual(self.report["inversions"], 0)
        self.assertLessEqual(self.report["inversions"], self.report["answered_pairs"])

    def test_first_arrival_is_slowest(self):
        first = self.report["first_arrival_wait_mean"]
        overall = waits.summarise(Dataset.read().consults).mean
        self.assertGreater(first, overall)


class SiteSkillTableTest(unittest.TestCase):
    def test_rows_use_demand_and_answered_denominators(self):
        table = waits.build()["by_site_skill"]
        self.assertTrue(table)
        for row in table:
            self.assertEqual(row["demand"], row["answered"] + row["unseen"])
            for cap, share in row["share_of_demand"].items():
                if share is not None:
                    self.assertLessEqual(share, row["share_of_answered"][cap])


class RegionReconciliationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = Dataset.read()
        cls.recon = waits.region_reconciliation(cls.ds.consults)

    def test_site_layers_explain_the_whole_region(self):
        self.assertTrue(self.recon["balances"])
        self.assertAlmostEqual(
            self.recon["sum_of_layers"], self.recon["observed_chain_wait"], places=6
        )

    def test_unattributed_is_a_bounded_interval_and_not_added_to_observed(self):
        un = self.recon["unattributed"]
        self.assertGreater(un["consults"], 0)
        self.assertLess(un["floor_minutes"], un["ceiling_minutes"])

    def test_site_rows_sum_to_region(self):
        per_site = self.recon["per_site"]
        self.assertAlmostEqual(
            sum(v["held_at_own_site"] for v in per_site.values()),
            self.recon["site_held_total"],
            places=6,
        )
        self.assertAlmostEqual(
            sum(v["held_after_claim"] for v in per_site.values()),
            self.recon["destination_after_claim_total"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
