#!/usr/bin/env python3
"""Unit tests for container-aware telemetry and bottleneck diagnosis."""

import os
import unittest
from unittest.mock import patch

import telemetry


class TestTelemetry(unittest.TestCase):
    def test_get_allocated_cpus_env_override(self):
        with patch.dict(os.environ, {"RUNPOD_CPU_COUNT": "8"}):
            self.assertEqual(telemetry.get_allocated_cpus(), 8.0)

        with patch.dict(os.environ, {"POD_CPU_COUNT": "16"}):
            self.assertEqual(telemetry.get_allocated_cpus(), 16.0)

    def test_diagnose_bottleneck_cpu_throttled(self):
        # High CPU, low/moderate GPU -> CPU_BOTTLENECK
        code, msg = telemetry.diagnose_bottleneck(34.6, 98.0, 595.0, 24564.0)
        self.assertEqual(code, "CPU_BOTTLENECK")
        self.assertIn("Do NOT increase actors", msg)

        code, msg = telemetry.diagnose_bottleneck(50.0, 85.0, 595.0, 24564.0)
        self.assertEqual(code, "CPU_BOTTLENECK")

    def test_diagnose_bottleneck_under_subscribed(self):
        # Both CPU and GPU low -> UNDER_SUBSCRIBED
        code, msg = telemetry.diagnose_bottleneck(25.0, 30.0, 595.0, 24564.0)
        self.assertEqual(code, "UNDER_SUBSCRIBED")

    def test_diagnose_bottleneck_gpu_saturated(self):
        code, msg = telemetry.diagnose_bottleneck(85.0, 50.0, 595.0, 24564.0)
        self.assertEqual(code, "GPU_SATURATED")

    def test_diagnose_bottleneck_balanced(self):
        code, msg = telemetry.diagnose_bottleneck(60.0, 65.0, 595.0, 24564.0)
        self.assertEqual(code, "BALANCED")

    def test_read_container_cpu_usage_sec_runs_cleanly(self):
        # Verify function executes without crashing
        res = telemetry.read_container_cpu_usage_sec()
        self.assertTrue(res is None or isinstance(res, float))


if __name__ == "__main__":
    unittest.main()
