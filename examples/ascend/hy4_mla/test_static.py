"""Local stdlib-only preflight regressions; do not import TileLang or torch."""

import ast
import unittest

from examples.ascend.hy4_mla.check_resources import HERE, evaluate, inspect
from examples.ascend.hy4_mla.mfu import count_valid_slots, cube_bf16_tflops, gemm_flops, mfu


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
            self.inspect_modified("bv = T.view(bq, shape=(64, qk_tile))", "bv = T.view(bq, shape=(64, 64))")

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
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hy4_sparse_attention")
        for packed, topk, expected_widths in (
            (False, 2048, [192, 192, 192]),
            (False, 64, [128, 128, 128, 128, 64]),
            (True, 2048, [128, 128, 128, 128, 64]),
        ):
            with self.subTest(packed=packed, topk=topk):
                env = {"packed": packed, "topk": topk, "pipeline_stages": True}
                for name in ("wide_qk", "qk_tile", "qk_parts"):
                    assignment = next(
                        n for n in builder.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name
                    )
                    env[name] = evaluate(assignment.value, env)
                count = evaluate(loop.iter.args[0], env)
                expression = loop.body[0].value
                covered = []
                widths = []
                for d in range(count):
                    width = min(evaluate(arg, {**env, "d": d}) for arg in expression.args)
                    self.assertGreater(width, 0)
                    self.assertEqual(width % 16, 0)
                    covered.extend(range(d * env["qk_tile"], d * env["qk_tile"] + width))
                    widths.append(width)
                self.assertEqual(covered, list(range(576)))
                self.assertEqual(widths, expected_widths)


class MfuAccounting(unittest.TestCase):
    def test_dense_gemm_matches_two_mnk(self):
        flops = 2 * 4096 * 4096 * 4096
        self.assertAlmostEqual(mfu(flops, 1.0e-3, 432.0), flops / (1.0e-3 * 432.0e12))
        self.assertAlmostEqual(mfu(flops, 1.0e-3, 432.0) * 100, 31.81457256, places=5)

    def test_sparse_numerator_counts_retained_slots_only(self):
        indices = [[0, -1, 3, 9], [4, 4, 1, 2]]
        self.assertEqual(count_valid_slots(indices, nk=5, lengths=[4, 2]), 4)
        self.assertEqual(gemm_flops(1), 64 * (2 * 576 + 2 * 512))
        self.assertEqual(cube_bf16_tflops(28), 378.0)
        self.assertAlmostEqual(mfu(gemm_flops(4), 2.0e-3, 432.0) * 2, mfu(gemm_flops(8), 2.0e-3, 432.0))

    def test_unknown_aic_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "36 AIC"):
            cube_bf16_tflops(36)


if __name__ == "__main__":
    unittest.main()
