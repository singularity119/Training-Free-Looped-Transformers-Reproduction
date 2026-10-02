"""Pure Python fake tensor checks; no real tensor/model equivalence claims."""
import copy
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tflt.loopscope.phase10_runtime import direction_schedule, make_runtime, validate_calls
from tflt.loopscope.phase9_runtime import Phase9Runtime


class Tensor:
    device = "fake"
    dtype = "fake32"
    def __init__(self, data):
        self.data = copy.deepcopy(data)
    @property
    def shape(self):
        value, shape = self.data, []
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0]
        return tuple(shape)
    @property
    def ndim(self):
        return len(self.shape)
    def __getitem__(self, index):
        value = self.data
        for item in index if isinstance(index, tuple) else (index,):
            value = value[item]
        return Tensor(value)
    def __setitem__(self, index, value):
        indices = index if isinstance(index, tuple) else (index,)
        target = self.data
        for item in indices[:-1]:
            target = target[item]
        target[indices[-1]] = copy.deepcopy(value.data if isinstance(value, Tensor) else value)
    def index_select(self, dim, indices):
        return Tensor([self.data[index] for index in indices])
    def float(self):
        return self
    def to(self, **kwargs):
        return self
    def numel(self):
        return math.prod(self.shape)
    def clone(self):
        return Tensor(self.data)
    def __mul__(self, value):
        return Tensor([item * value for item in self.data])
    __rmul__ = __mul__
    def __sub__(self, other):
        return Tensor([left - right for left, right in zip(self.data, other.data)])


def mutation(torch, before, after, position):
    differences = [[abs(x-y) for x, y in zip(left, right)] for left, right in zip(before.data[0], after.data[0])]
    return {"same_object": before is after, "answer_max_abs_change": max(differences[position]),
        "nonanswer_max_abs_change": max([max(row) for index, row in enumerate(differences) if index != position] or [0]),
        "changed_element_count": sum(value != 0 for row in differences for value in row)}


FAKE_TORCH = SimpleNamespace(
    tensor=lambda value, **kwargs: value, long="long", isfinite=lambda value: math.isfinite(value),
    dot=lambda a, b: sum(left*right for left, right in zip(a.data, b.data)),
    zeros_like=lambda value: Tensor([0]*len(value.data)),
    linalg=SimpleNamespace(vector_norm=lambda value: math.sqrt(sum(item*item for item in value.data))))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cell = {"arm": "Online", "intervention_mode": "spectral", "direction_policy": "fixed_t0", "k": 2, "strength": 0.5}
    def execute(self, policy, strength=0.5, legacy=False):
        runtime = Phase9Runtime("Online-t0", strength) if legacy else make_runtime({**self.cell, "direction_policy": policy, "strength": strength})
        delta = Tensor([[[4., 0.], [3., 4.], [8., 9.]]])
        modules = ("tflt.loopscope.phase9_runtime", "tflt.loopscope.phase9_gate_e_runtime")
        fit = lambda rows, torch: (Tensor([1., 0.]), {"algorithm": "fake_not_svd"})
        with patch.dict(sys.modules, torch=FAKE_TORCH):
            with patch(modules[0]+"._fit_direction", fit), patch(modules[1]+"._fit_direction", fit), \
                 patch(modules[0]+"._finite_tensor"), patch(modules[1]+"._finite_tensor"), \
                 patch(modules[0]+"._tensor_change", mutation), patch(modules[1]+"._tensor_change", mutation):
                with runtime.context("sample", 1, (0, 1), token_ids=(10,11,12)):
                    self.assertIs(runtime(delta, 0), delta)
                    result = runtime(delta, 1)
        self.assertEqual(result.data[0][0], delta.data[0][0])
        self.assertEqual(result.data[0][2], delta.data[0][2])
        self.assertEqual(validate_calls(runtime, 2), 1)
        return runtime, result, delta
    def test_k2_fake_callbacks_equivalent_and_lambda_wired(self):
        self.assertEqual(direction_schedule("fixed_t0", 2), direction_schedule("lag1", 2))
        for strength in (0.1, 0.5, 0.9):
            fixed, fixed_out, _ = self.execute("fixed_t0", strength)
            lag, lag_out, _ = self.execute("lag1", strength)
            self.assertEqual(fixed_out.data, lag_out.data)
            self.assertAlmostEqual(fixed_out.data[0][1][0], 3*(1-strength))
            self.assertEqual(fixed.strength, lag.strength)
    def test_lambda_zero_same_object_and_half_legacy_compatibility(self):
        _, result, delta = self.execute("fixed_t0", 0)
        self.assertIs(result, delta)
        _, new, _ = self.execute("fixed_t0", .5)
        _, historical, _ = self.execute("fixed_t0", .5, legacy=True)
        self.assertEqual(new.data, historical.data)
    def test_lag_trajectory_restarts_for_repeated_candidate_context(self):
        runtime = make_runtime({**self.cell, "direction_policy": "lag1", "k":3})
        with runtime.context("same", 1, (0,1), token_ids=(10,11,12)):
            runtime._direction = "last_v1"
            runtime._fit_metadata = {0:{}, 1:{}}
        with runtime.context("same", 1, (0,1), token_ids=(10,11,13)):
            self.assertIsNone(runtime.direction)
            self.assertEqual(runtime.fit_metadata, {})
        self.assertEqual(direction_schedule("lag1",3), [(0,None,0),(1,0,1),(2,1,None)])
    def test_no_matched_or_extra_k(self):
        for field, value in (("arm", "Matched-norm"), ("intervention_mode", "matched_norm"), ("k",4)):
            with self.assertRaises(ValueError):
                make_runtime({**self.cell, field:value})
        self.assertIsNone(make_runtime({"arm":"Native"}))
        self.assertIsNone(make_runtime({"arm":"Loop"}))

    def test_current_fits_raw_prompt_before_mutation_every_round_including_t0(self):
        runtime = make_runtime({**self.cell,"direction_policy":"current_t","k":3})
        fits=[]
        def fit(rows,torch):
            fits.append(copy.deepcopy(rows.data))
            axis = [0.,1.] if rows.data[0][1] else [1.,0.]
            return Tensor(axis), {"algorithm":"fake_not_svd"}
        current_module="tflt.loopscope.phase10_runtime"
        with patch.dict(sys.modules,torch=FAKE_TORCH), \
             patch("tflt.loopscope.phase9_gate_e_runtime._fit_direction",fit), \
             patch(current_module+"._finite_tensor"), patch(current_module+"._tensor_change",mutation):
            with runtime.context("own-trajectory",1,(0,1),token_ids=(10,11,99)):
                raw0=Tensor([[[4.,0.],[3.,4.],[90.,100.]]])
                out0=runtime(raw0,0)
                self.assertEqual(out0.data[0][1],[1.5,4.])
                raw1=Tensor([[[0.,6.],[out0.data[0][1][0],8.],[90.,100.]]])
                out1=runtime(raw1,1)
                self.assertEqual(out1.data[0][1],[1.5,4.])
                raw2=Tensor([[[9.,0.],[out1.data[0][1][0]+1,out1.data[0][1][1]+1],[90.,100.]]])
                out2=runtime(raw2,2)
                self.assertEqual(out2.data[0][1],[1.25,5.])
        self.assertEqual(fits,[[[4.,0.],[3.,4.]],[[0.,6.],[1.5,8.]],[[9.,0.],[2.5,5.]]])
        for before,after in ((raw0,out0),(raw1,out1),(raw2,out2)):
            self.assertEqual(before.data[0][0],after.data[0][0])
            self.assertEqual(before.data[0][2],after.data[0][2])
        self.assertEqual(validate_calls(runtime,3),1)
        self.assertEqual([(call["direction_fit_t"],call["direction_used_fit_t"],call["applied_t"])
            for call in runtime.calls],[(0,0,0),(1,1,1),(2,2,2)])
        self.assertEqual(set(runtime.fit_metadata),{0,1,2})
        self.assertEqual(direction_schedule("current_t",2),[(0,0,0),(1,1,1)])
        self.assertNotEqual(direction_schedule("current_t",2),direction_schedule("fixed_t0",2))
        self.assertEqual(runtime.summary()["schedule"][0],{"t":0,"direction_used_fit_t":0,"direction_fit_t":0})

    def test_current_zero_strength_preserves_raw_object_and_restarts_candidate(self):
        runtime=make_runtime({**self.cell,"direction_policy":"current_t","strength":0})
        module="tflt.loopscope.phase10_runtime"
        fits=[]
        def fit(rows,torch):
            fits.append(copy.deepcopy(rows.data))
            return Tensor([1.,0.]),{"algorithm":"fake_not_svd"}
        raw=Tensor([[[4.,0.],[3.,4.],[100.,100.]]])
        with patch.dict(sys.modules,torch=FAKE_TORCH), \
             patch("tflt.loopscope.phase9_gate_e_runtime._fit_direction",fit), \
             patch(module+"._finite_tensor"), patch(module+"._tensor_change",mutation):
            for candidate in range(2):
                with runtime.context("same-prompt",1,(0,1),token_ids=(10,11,20+candidate)):
                    self.assertIsNone(runtime.direction)
                    self.assertIs(runtime(raw,0),raw)
                    self.assertIs(runtime(raw,1),raw)
        self.assertEqual(len(fits),4)
        self.assertEqual(validate_calls(runtime,2),2)
        self.assertFalse(any(call["applied"] for call in runtime.calls))


if __name__ == "__main__":
    unittest.main()
