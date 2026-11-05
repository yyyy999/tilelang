"""Local stdlib-only preflight regressions; do not import TileLang or torch."""

import ast
import unittest

from examples.ascend.hy4_mla.check_resources import HERE, evaluate, inspect


class ResourcePreflight(unittest.TestCase):
    def test_specializations_fit_without_alias_reuse(self):
        for topk in (64, 128, 192, 256, 320, 512, 640, 704, 960, 1024, 2048):
            for pipeline in (False, True):
                with self.subTest(topk=topk, pipeline=pipeline):
                    result = inspect(topk, pipeline)
                    self.assertTrue(result["fits"], result["pools"])

    def inspect_modified(self, before, after):
        source = (HERE / "kernel.py").read_text(encoding="utf-8")
        self.assertIn(before, source)
        return inspect(source_text=source.replace(before, after, 1))

    def test_three_gather_versions_exceed_budget(self):
        result = self.inspect_modified("kv: 2,", "kv: 3,")
        self.assertFalse(result["fits"])
        self.assertGreater(result["pools"]["UB"]["no_reuse_bytes"], result["pools"]["UB"]["capacity"])

    def test_shrinking_l0_view_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preserve bit count"):
            self.inspect_modified("bv = T.view(bq, shape=(64, 128))", "bv = T.view(bq, shape=(64, 64))")

    def test_overlapping_l1_rings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.inspect_modified("q_l1: 147456", "q_l1: 131072")

    def test_qk_tiles_cover_576_once_with_aligned_tail(self):
        tree = ast.parse((HERE / "kernel.py").read_text(encoding="utf-8"))
        loop = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For) and isinstance(n.body[0], ast.Assign) and ast.unparse(n.body[0].targets[0]) == "qk_width"
        )
        count = evaluate(loop.iter.args[0], {})
        expression = loop.body[0].value
        covered = []
        widths = []
        for d in range(count):
            width = min(evaluate(arg, {"d": d}) for arg in expression.args)
            self.assertGreater(width, 0)
            self.assertEqual(width % 16, 0)
            self.assertLessEqual(width, 128)
            covered.extend(range(d * 128, d * 128 + width))
            widths.append(width)
        self.assertEqual(covered, list(range(576)))
        self.assertEqual(widths[-1], 64)


if __name__ == "__main__":
    unittest.main()
