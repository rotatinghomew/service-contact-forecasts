"""核对交接资料的文件结构和明确引用。"""
import json
from pathlib import Path
import unittest


DATA = Path(__file__).resolve().parents[1] / "data"


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


class InputContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = load("format")
        self.records = {name: load(name) for name in self.contract["fields"]}

    def test_table_columns(self):
        for name, columns in self.contract["fields"].items():
            with self.subTest(table=name):
                self.assertIsInstance(self.records[name], list)
                self.assertTrue(self.records[name])
                for row in self.records[name]:
                    self.assertCountEqual(row, columns)

    def test_source_references(self):
        for table, field, target, key in self.contract["references"]:
            with self.subTest(table=table, field=field):
                keys = {row[key] for row in self.records[target]}
                values = {row[field] for row in self.records[table] if row[field] is not None}
                self.assertLessEqual(values, keys)


if __name__ == "__main__":
    unittest.main()
