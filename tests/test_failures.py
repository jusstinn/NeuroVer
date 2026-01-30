"""
Protocol-Verify: Failure Scenario Tests

These tests verify that the system CORRECTLY REJECTS non-compliant training.
All tests should PASS because the system correctly catches violations.
"""

import sys
import json
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.monitor import WeightMonitor, SafetyInvariants, compute_frobenius_norm
from core.proof_gen import MockProofGenerator


class TestWeightNormViolations:
    """Tests that verify the system REJECTS training exceeding weight norm limits."""
    
    @pytest.fixture
    def policy_threshold(self) -> float:
        config_path = Path(__file__).parent.parent / "policy" / "safety_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        return config["weight_constraints"]["max_weight_norm"]
    
    def test_FAIL_when_norm_exceeds_threshold_by_10_percent(self, policy_threshold):
        """Training 10% over limit should be REJECTED."""
        np.random.seed(100)
        r, d = 8, 768
        
        base_A = np.random.randn(r, d)
        base_B = np.random.randn(d, r)
        base_norm = compute_frobenius_norm(base_A, base_B)
        
        target_norm = policy_threshold * 1.1
        scale = target_norm / base_norm
        A, B = base_A * scale, base_B * scale
        
        actual_norm = compute_frobenius_norm(A, B)
        assert actual_norm > policy_threshold, "Test setup: norm should exceed threshold"
        
        safety = SafetyInvariants(max_weight_norm=policy_threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "test_hash")
        
        assert not result.passed, f"SECURITY BUG: Verification should FAIL when norm exceeds threshold!"
    
    def test_FAIL_when_norm_exceeds_threshold_by_50_percent(self, policy_threshold):
        """Training 50% over limit should be REJECTED."""
        np.random.seed(200)
        r, d = 8, 768
        
        base_A = np.random.randn(r, d)
        base_B = np.random.randn(d, r)
        base_norm = compute_frobenius_norm(base_A, base_B)
        
        scale = (policy_threshold * 1.5) / base_norm
        A, B = base_A * scale, base_B * scale
        
        safety = SafetyInvariants(max_weight_norm=policy_threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
        
        assert not result.passed, "SECURITY BUG: 50% over threshold should FAIL!"
    
    def test_FAIL_catastrophic_weight_explosion(self, policy_threshold):
        """Catastrophic weight explosion should be REJECTED."""
        np.random.seed(666)
        A = np.random.randn(8, 768) * 0.5
        B = np.random.randn(768, 8) * 0.5
        
        safety = SafetyInvariants(max_weight_norm=policy_threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
        
        assert not result.passed, "SECURITY BUG: Catastrophic explosion should FAIL!"
    
    def test_FAIL_proof_rejected_for_excessive_weights(self, policy_threshold):
        """Proof for excessive weights should indicate non-compliance."""
        np.random.seed(400)
        r, d = 8, 768
        
        base_A = np.random.randn(r, d)
        base_B = np.random.randn(d, r)
        base_norm = compute_frobenius_norm(base_A, base_B)
        
        scale = (policy_threshold * 2) / base_norm
        A, B = base_A * scale, base_B * scale
        
        proof_gen = MockProofGenerator("./test_failure_proofs")
        proof_gen.setup(A.shape, B.shape, threshold=policy_threshold)
        proof = proof_gen.generate_proof(A, B, "test_hash")
        
        assert proof["public_inputs"]["result"] == 0, "SECURITY BUG: Proof should indicate non-compliance!"
    
    def test_FAIL_verification_returns_false(self, policy_threshold):
        """Verification should return FALSE for non-compliant proof."""
        np.random.seed(500)
        A = np.random.randn(8, 768) * 0.15
        B = np.random.randn(768, 8) * 0.15
        
        proof_gen = MockProofGenerator("./test_failure_verify")
        proof_gen.setup(A.shape, B.shape, threshold=policy_threshold)
        proof_gen.generate_proof(A, B, "test_hash")
        
        is_valid = proof_gen.verify_proof("./test_failure_verify/mock_proof.json")
        
        assert not is_valid, "SECURITY BUG: Verification should return FALSE!"


class TestBasemodelTamperingViolations:
    """Tests that verify the system REJECTS tampered base models."""
    
    @pytest.fixture
    def policy_threshold(self) -> float:
        config_path = Path(__file__).parent.parent / "policy" / "safety_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        return config["weight_constraints"]["max_weight_norm"]
    
    @pytest.fixture
    def compliant_weights(self):
        np.random.seed(42)
        r, d = 8, 768
        return {"layer_0": (np.random.randn(r, d) * 0.001, np.random.randn(d, r) * 0.001)}
    
    def test_FAIL_when_base_model_hash_doesnt_match(self, policy_threshold, compliant_weights):
        """Tampered base model hash should be REJECTED."""
        expected_hash = "expected_clean_" + "a" * 50
        provided_hash = "tampered_base_" + "b" * 50
        
        safety = SafetyInvariants(max_weight_norm=policy_threshold, expected_model_hash=expected_hash)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants(compliant_weights, provided_hash)
        
        assert not result.passed, "SECURITY BUG: Tampered base model should FAIL!"
        assert not result.base_hash_valid, "Hash should be marked invalid"
    
    def test_FAIL_single_character_difference(self, policy_threshold, compliant_weights):
        """Single-char hash difference should be REJECTED."""
        expected_hash = "a" * 64
        tampered_hash = "a" * 63 + "b"
        
        safety = SafetyInvariants(max_weight_norm=policy_threshold, expected_model_hash=expected_hash)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants(compliant_weights, tampered_hash)
        
        assert not result.passed, "SECURITY BUG: Single-char difference should FAIL!"


class TestBoundaryConditions:
    """Tests at the exact boundary of acceptable limits."""
    
    @pytest.fixture
    def policy_threshold(self) -> float:
        config_path = Path(__file__).parent.parent / "policy" / "safety_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        return config["weight_constraints"]["max_weight_norm"]
    
    def test_PASS_when_exactly_at_threshold(self, policy_threshold):
        """Exactly at threshold should PASS (boundary is inclusive)."""
        np.random.seed(600)
        r, d = 8, 768
        
        base_A = np.random.randn(r, d)
        base_B = np.random.randn(d, r)
        base_norm = compute_frobenius_norm(base_A, base_B)
        
        scale = policy_threshold / base_norm
        A, B = base_A * scale, base_B * scale
        
        safety = SafetyInvariants(max_weight_norm=policy_threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "test_hash")
        
        assert result.passed, f"At exactly threshold should PASS"
    
    def test_FAIL_when_barely_over_threshold(self, policy_threshold):
        """0.1% over threshold should FAIL."""
        np.random.seed(700)
        r, d = 8, 768
        
        base_A = np.random.randn(r, d)
        base_B = np.random.randn(d, r)
        base_norm = compute_frobenius_norm(base_A, base_B)
        
        target_norm = policy_threshold * 1.001
        scale = target_norm / base_norm
        A, B = base_A * scale, base_B * scale
        
        safety = SafetyInvariants(max_weight_norm=policy_threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "test_hash")
        
        assert not result.passed, "Even 0.1% over threshold should FAIL!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
