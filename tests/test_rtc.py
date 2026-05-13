import unittest
import importlib.util

import numpy as np
import torch

from starVLA.model.modules.action_model.rtc import (
    RTCActionQueue,
    RTCConfig,
    RTCAttentionSchedule,
    apply_rtc_guidance,
    make_prefix_attention_weights,
    prepare_prev_chunk_left_over,
    resolve_prefix_attention_horizon,
)


class RTCConfigTest(unittest.TestCase):
    def test_defaults_and_default_prefix_horizon(self):
        cfg = RTCConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.execution_horizon, 8)
        self.assertEqual(resolve_prefix_attention_horizon(cfg, 16), 8)

    def test_asymmetric_default_prefix_horizon(self):
        cfg = RTCConfig(enabled=True, execution_horizon=4)
        self.assertEqual(resolve_prefix_attention_horizon(cfg, 16), 12)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            RTCConfig(execution_horizon=0)
        with self.assertRaises(ValueError):
            RTCConfig(prefix_attention_horizon=-1)
        with self.assertRaises(ValueError):
            RTCConfig(max_guidance_weight=-1)
        with self.assertRaises(ValueError):
            RTCConfig(prefix_attention_schedule="BAD")


class PrefixWeightsTest(unittest.TestCase):
    def test_exp_schedule(self):
        cfg = RTCConfig(enabled=True, execution_horizon=8, prefix_attention_schedule="EXP")
        weights = make_prefix_attention_weights(16, 4, cfg)
        self.assertTrue(torch.allclose(weights[:4], torch.ones(4)))
        self.assertGreater(float(weights[4]), float(weights[5]))
        self.assertGreater(float(weights[5]), float(weights[6]))
        self.assertGreater(float(weights[7]), 0.0)
        self.assertTrue(torch.allclose(weights[8:], torch.zeros(8)))

    def test_linear_schedule(self):
        cfg = RTCConfig(enabled=True, execution_horizon=8, prefix_attention_schedule="LINEAR")
        weights = make_prefix_attention_weights(16, 4, cfg)
        self.assertTrue(torch.allclose(weights[:4], torch.ones(4)))
        self.assertAlmostEqual(float(weights[4]), 1.0)
        self.assertAlmostEqual(float(weights[7]), 0.0)
        self.assertTrue(torch.allclose(weights[8:], torch.zeros(8)))

    def test_ones_and_zeros(self):
        ones = make_prefix_attention_weights(
            16,
            4,
            RTCConfig(enabled=True, execution_horizon=8, prefix_attention_schedule=RTCAttentionSchedule.ONES),
        )
        zeros = make_prefix_attention_weights(
            16,
            4,
            RTCConfig(enabled=True, execution_horizon=8, prefix_attention_schedule=RTCAttentionSchedule.ZEROS),
        )
        self.assertTrue(torch.allclose(ones[:8], torch.ones(8)))
        self.assertTrue(torch.allclose(ones[8:], torch.zeros(8)))
        self.assertTrue(torch.allclose(zeros, torch.zeros(16)))


class LeftoverShapeTest(unittest.TestCase):
    def test_leftover_2d_and_3d(self):
        leftover_2d = np.zeros((8, 7), dtype=np.float32)
        out_2d = prepare_prev_chunk_left_over(
            leftover_2d,
            batch_size=2,
            action_dim=7,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(tuple(out_2d.shape), (2, 8, 7))

        leftover_3d = np.zeros((2, 8, 7), dtype=np.float32)
        out_3d = prepare_prev_chunk_left_over(
            leftover_3d,
            batch_size=2,
            action_dim=7,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(tuple(out_3d.shape), (2, 8, 7))

    def test_leftover_dim_mismatch(self):
        with self.assertRaises(ValueError):
            prepare_prev_chunk_left_over(
                np.zeros((8, 6), dtype=np.float32),
                batch_size=1,
                action_dim=7,
                device="cpu",
                dtype=torch.float32,
            )


class RTCGuidanceSmokeTest(unittest.TestCase):
    def test_apply_guidance_shape_and_finite(self):
        actions = torch.randn(2, 16, 7, requires_grad=True)
        pred_velocity = torch.tanh(actions)
        leftover = torch.zeros(2, 8, 7)
        cfg = RTCConfig(enabled=True, execution_horizon=8, max_guidance_weight=5.0)
        guided, metadata = apply_rtc_guidance(
            actions,
            pred_velocity,
            t_cont=0.25,
            prev_chunk_left_over=leftover,
            inference_delay=4,
            rtc_config=cfg,
        )
        self.assertEqual(tuple(guided.shape), (2, 16, 7))
        self.assertFalse(torch.isnan(guided).any().item())
        self.assertTrue(metadata["applied"])

    def test_fake_flow_head_7d_and_14d(self):
        for action_dim in (7, 14):
            head = _FakeRTCFlowHead(action_horizon=16, action_dim=action_dim)
            leftover = torch.zeros(1, 8, action_dim)
            out = head.predict_action_rtc(
                batch_size=1,
                prev_chunk_left_over=leftover,
                inference_delay=4,
                rtc_config=RTCConfig(enabled=True, execution_horizon=8),
            )
            self.assertEqual(tuple(out.shape), (1, 16, action_dim))
            self.assertFalse(torch.isnan(out).any().item())


class _FakeRTCFlowHead:
    def __init__(self, action_horizon: int, action_dim: int):
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.num_inference_timesteps = 4

    def predict_action_rtc(self, *, batch_size, prev_chunk_left_over, inference_delay, rtc_config):
        actions = torch.randn(batch_size, self.action_horizon, self.action_dim)
        prev_chunk_left_over = prepare_prev_chunk_left_over(
            prev_chunk_left_over,
            batch_size=batch_size,
            action_dim=self.action_dim,
            device="cpu",
            dtype=torch.float32,
        )
        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            actions = actions.detach().requires_grad_(True)
            pred_velocity = torch.tanh(actions)
            pred_velocity, _ = apply_rtc_guidance(
                actions,
                pred_velocity,
                t_cont=t_cont,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                rtc_config=rtc_config,
            )
            actions = (actions + dt * pred_velocity).detach()
        return actions


class RTCActionQueueTest(unittest.TestCase):
    def test_merge_skips_stale_prefix_7d_and_14d(self):
        for action_dim in (7, 14):
            queue = RTCActionQueue(execution_horizon=8)
            queue.reset(np.zeros((16, action_dim), dtype=np.float32))
            new_chunk = np.arange(16 * action_dim, dtype=np.float32).reshape(16, action_dim)
            queue.merge(new_chunk, skip_steps=5)
            leftover = queue.get_left_over()
            np.testing.assert_array_equal(leftover, new_chunk[5:])

    def test_underflow_raises(self):
        queue = RTCActionQueue()
        with self.assertRaises(RuntimeError):
            queue.pop()


class WebsocketRTCRouteTest(unittest.TestCase):
    def test_infer_rtc_route_calls_policy(self):
        if importlib.util.find_spec("msgpack") is None:
            self.skipTest("msgpack is not installed in this environment")

        from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

        class FakePolicy:
            def __init__(self):
                self.called = False

            def predict_action_rtc(self, **payload):
                self.called = True
                return {
                    "normalized_actions": np.zeros((1, 16, 7), dtype=np.float32),
                    "rtc": {"applied": False},
                }

        policy = FakePolicy()
        server = WebsocketPolicyServer(policy)
        response = server._route_message(
            {
                "type": "infer_rtc",
                "request_id": "test-step",
                "payload": {
                    "examples": [{"image": [], "lang": "test"}],
                    "rtc": {"enabled": True, "prev_chunk_left_over": None},
                },
            }
        )
        self.assertTrue(policy.called)
        self.assertTrue(response["ok"])
        self.assertEqual(response["type"], "rtc_inference_result")
        self.assertIn("normalized_actions", response["data"])


if __name__ == "__main__":
    unittest.main()
