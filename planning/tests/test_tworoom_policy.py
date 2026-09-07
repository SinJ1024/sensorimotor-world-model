"""CPU regression tests: python -m unittest discover -s planning/tests -v."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_tworoom_policy import ActionNormalizer, DirectPolicy, sample_tasks, load_evaluation_state
from module import PolicyModel


class FakeModel(torch.nn.Module):
    def __init__(self, context=3, frameskip=2, use_action=True):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.policy_model = SimpleNamespace(context=context, use_action=use_action)
        self.frameskip = frameskip
        self.calls = []

    def encode(self, info):
        return {"emb": info["pixels"][:, :1, 0, 0].float().unsqueeze(1)}

    def predict_next_action(self, z, a):
        self.calls.append((z.clone(), None if a is None else a.clone()))
        # Two future blocks: only first should ever execute.
        prediction = torch.zeros((len(z), 2, self.frameskip * 2))
        prediction[:, 0] = torch.arange(self.frameskip * 2).float() * 10
        prediction[:, 1] = -999
        return prediction


class EvaluationTests(unittest.TestCase):
    def test_inverse_checkpoint_with_and_without_unused_head(self):
        for saved_head in [False, True]:
            model = torch.nn.Module()
            model.encoder = torch.nn.Linear(2, 2)
            model.policy_model = torch.nn.Linear(2, 2)
            state = {k: v.clone() for k, v in model.state_dict().items()
                     if saved_head or not k.startswith("policy_model.")}
            load_evaluation_state(model, state, 0, "cem")
            self.assertIsNone(model.policy_model)
            broken = dict(state)
            broken.pop("encoder.weight")
            with self.assertRaises(RuntimeError):
                load_evaluation_state(model, broken, 0, "cem")

    def test_legacy_unused_policy_head_and_strict_active_weights(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(2, 2)
        model.policy_model = torch.nn.Linear(2, 2)
        state = {"encoder.weight": torch.full((2, 2), 3.),
                 "encoder.bias": torch.full((2,), 4.)}
        for layer in [0, 2, 4]:
            state[f"policy_model.net.{layer}.weight"] = torch.zeros(7, 7)
            state[f"policy_model.net.{layer}.bias"] = torch.zeros(7)
        original_keys = set(state)
        load_evaluation_state(model, state, 0, "cem")
        torch.testing.assert_close(model.encoder.weight, state["encoder.weight"])
        torch.testing.assert_close(model.encoder.bias, state["encoder.bias"])
        self.assertEqual(set(state), original_keys)
        with self.assertRaises(RuntimeError):
            load_evaluation_state(model, {**state, "unknown.weight": torch.zeros(1)}, 0, "cem")
        with self.assertRaises(RuntimeError):
            load_evaluation_state(model, {**state, "encoder.weight": torch.zeros(3, 3)}, 0, "cem")
        model.policy_model = torch.nn.Linear(2, 2)
        with self.assertRaises(RuntimeError):
            load_evaluation_state(model, state, 0.1, "cem")

    def test_untrained_or_missing_policy_is_not_evaluated_directly(self):
        model = torch.nn.Module()
        model.policy_model = torch.nn.Linear(2, 2)
        for mode in ["direct", "both"]:
            with self.assertRaises(ValueError):
                load_evaluation_state(model, model.state_dict(), 0, mode)
        with self.assertRaises(RuntimeError):
            load_evaluation_state(model, {}, 0.1, "cem")

    def setUp(self):
        self.norm = ActionNormalizer(np.array([[1., -1.], [3., 1.]], np.float32))
        self.env = SimpleNamespace(action_space=SimpleNamespace(shape=(2, 2),
            low=np.full((2, 2), -2, np.float32), high=np.full((2, 2), 2, np.float32)))

    def info(self, value):
        return {"pixels": np.full((2, 1, 4, 4, 3), value, np.uint8)}

    def test_sampling_last_start_and_short_episodes(self):
        episodes, starts = sample_tasks([2, 4, 1], 3, 1, 12)
        self.assertEqual(len(set(zip(episodes, starts))), 3)
        episodes, starts = sample_tasks([2, 4, 1], 4, 1, 12)
        self.assertEqual(set(zip(episodes, starts)), {(0, 0), (1, 0), (1, 1), (1, 2)})
        self.assertEqual(sample_tasks([26], 1, 25, 0), ([0], [0]))
        with self.assertRaises(ValueError):
            sample_tasks([25], 1, 25, 0)

    def test_training_std_and_inverse(self):
        np.testing.assert_allclose(self.norm.std, [np.sqrt(2), np.sqrt(2)])
        x = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        np.testing.assert_allclose(self.norm.inverse_transform(self.norm.transform(x)), x, atol=1e-6)
        with self.assertRaises(ValueError):
            ActionNormalizer(np.ones((2, 2), np.float32))

    def test_blocks_history_clipping_and_reset(self):
        model = FakeModel()
        policy = DirectPolicy(model, self.norm, 2)
        policy.set_env(self.env)
        first = policy.get_action(self.info(10))
        second = policy.get_action(self.info(11))
        self.assertEqual(len(model.calls), 1)
        policy.get_action(self.info(12))
        z, a = model.calls[1]
        np.testing.assert_array_equal(z[0, :, 0], [10, 10, 12])
        actual = np.stack([first, second], axis=1)
        np.testing.assert_allclose(a[:, -1].numpy(), self.norm.transform(actual).reshape(2, -1))
        np.testing.assert_allclose(model.calls[0][1][:, 0].numpy(),
            self.norm.transform(np.zeros((2, 2, 2), np.float32)).reshape(2, -1))
        self.assertTrue((actual <= 2).all() and (actual >= -2).all())
        policy.set_env(self.env)
        policy.get_action(self.info(99))
        np.testing.assert_array_equal(model.calls[-1][0][0, :, 0], [99, 99, 99])

    def test_single_context_and_action_free(self):
        for use_action in [False, True]:
            model = FakeModel(context=1, use_action=use_action)
            policy = DirectPolicy(model, self.norm, 2)
            policy.set_env(self.env)
            policy.get_action(self.info(0))
            a = model.calls[0][1]
            if use_action:
                self.assertEqual(tuple(a.shape), (2, 0, 4))
            else:
                self.assertIsNone(a)

    def test_real_heads_c16_frameskip5(self):
        for arch in ["mlp", "resmlp", "gru", "transformer", "mamba"]:
            for use_action in [True, False]:
                head = PolicyModel(embed_dim=192, action_dim=10, hidden_dim=32,
                    context=16, num_future=3, arch=arch, depth=2, use_action=use_action).eval()
                with torch.inference_mode():
                    y = head(torch.randn(2, 16, 192), torch.randn(2, 15, 10) if use_action else None)
                self.assertEqual(tuple(y.shape), (2, 3, 10))
                self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
