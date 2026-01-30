#!/usr/bin/env python3
"""
Protocol-Verify: MASSIVE Comprehensive Test Suite

Testing across:
- 20+ Learning rate configurations
- 8 LoRA ranks (2, 4, 8, 16, 32, 64, 128, 256)
- 10 weight scales
- 25 boundary conditions
- 15 layers
- 10 different "model" configurations (simulated architectures)
- Multiple attack scenarios
"""

import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from core.monitor import WeightMonitor, SafetyInvariants, compute_frobenius_norm
from core.proof_gen import MockProofGenerator


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    details: str = ""
    category: str = "general"
    norm: float = 0.0
    threshold: float = 10.0
    detected: bool = False  # For before/after comparison


@dataclass
class TestSuiteResults:
    tests: List[TestResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def add(self, result: TestResult):
        self.tests.append(result)
    
    @property
    def total_tests(self) -> int:
        return len(self.tests)
    
    @property
    def passed_tests(self) -> int:
        return sum(1 for t in self.tests if t.passed)
    
    @property
    def failed_tests(self) -> int:
        return self.total_tests - self.passed_tests
    
    @property
    def pass_rate(self) -> float:
        return self.passed_tests / self.total_tests * 100 if self.total_tests > 0 else 0.0
    
    @property
    def total_duration_ms(self) -> float:
        return sum(t.duration_ms for t in self.tests)
    
    def by_category(self, category: str) -> List[TestResult]:
        return [t for t in self.tests if t.category == category]


def load_safety_config() -> dict:
    config_path = Path(__file__).parent / "policy" / "safety_config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def run_test(name: str, category: str, test_func) -> TestResult:
    start = time.perf_counter()
    try:
        passed, details, norm, threshold, detected = test_func()
    except Exception as e:
        passed, details, norm, threshold, detected = False, f"Exception: {e}", 0.0, 10.0, False
    duration = (time.perf_counter() - start) * 1000
    
    return TestResult(
        name=name, passed=passed, duration_ms=duration,
        details=details, category=category, norm=norm, threshold=threshold, detected=detected
    )


def create_weights_with_target_norm(target_norm: float, r: int = 8, d: int = 768, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create weight matrices A, B such that ||BA||_F ≈ target_norm."""
    np.random.seed(seed)
    
    base_A = np.random.randn(r, d)
    base_B = np.random.randn(d, r)
    
    P = base_B @ base_A
    current_norm = np.sqrt(np.sum(P ** 2))
    
    if current_norm < 1e-10:
        current_norm = 1e-10
    
    scale = np.sqrt(target_norm / current_norm)
    
    return base_A * scale, base_B * scale


# =============================================================================
# MODEL ARCHITECTURE CONFIGURATIONS (simulated different model sizes)
# =============================================================================

MODEL_CONFIGS = {
    "tiny": {"hidden": 256, "layers": 4},
    "small": {"hidden": 512, "layers": 6},
    "base": {"hidden": 768, "layers": 12},
    "medium": {"hidden": 1024, "layers": 12},
    "large": {"hidden": 1280, "layers": 24},
    "xl": {"hidden": 1600, "layers": 24},
    "xxl": {"hidden": 2048, "layers": 32},
    "gpt2": {"hidden": 768, "layers": 12},
    "gpt2-medium": {"hidden": 1024, "layers": 24},
    "gpt2-large": {"hidden": 1280, "layers": 36},
}


# =============================================================================
# LEARNING RATE TESTS
# =============================================================================

def create_lr_test(lr_value: float, expected_pass: bool, seed: int):
    """Factory to create learning rate tests."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        np.random.seed(seed)
        scale = lr_value * 10
        
        A = np.random.randn(8, 768) * scale
        B = np.random.randn(768, 8) * scale
        
        norm = compute_frobenius_norm(A, B)
        
        safety = SafetyInvariants(max_weight_norm=threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
        
        # Detection: did our system catch a violation?
        is_violation = norm > threshold
        detected = (is_violation and not result.passed) or (not is_violation and result.passed)
        
        if expected_pass:
            passed = result.passed
            status = "ACCEPTED" if passed else "WRONGLY REJECTED"
        else:
            passed = not result.passed
            status = "REJECTED" if passed else "WRONGLY ACCEPTED"
        
        return passed, f"LR={lr_value}, norm={norm:.4f}, {status}", norm, threshold, detected
    
    return test_func


# =============================================================================
# LORA RANK TESTS
# =============================================================================

def create_rank_test(rank: int, hidden: int, scale: float, expected_pass: bool, seed: int, model_name: str = "base"):
    """Factory to create LoRA rank tests for different model sizes."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        np.random.seed(seed)
        
        A = np.random.randn(rank, hidden) * scale
        B = np.random.randn(hidden, rank) * scale
        
        norm = compute_frobenius_norm(A, B)
        
        safety = SafetyInvariants(max_weight_norm=threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
        
        is_violation = norm > threshold
        detected = (is_violation and not result.passed) or (not is_violation and result.passed)
        
        if expected_pass:
            passed = result.passed
        else:
            passed = not result.passed
        
        return passed, f"{model_name}: rank={rank}, h={hidden}, norm={norm:.4f}", norm, threshold, detected
    
    return test_func


# =============================================================================
# BOUNDARY CONDITION TESTS
# =============================================================================

def create_boundary_test(percent_of_threshold: float, seed: int):
    """Factory to create boundary condition tests."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        target_norm = threshold * (percent_of_threshold / 100.0)
        A, B = create_weights_with_target_norm(target_norm, seed=seed)
        
        actual_norm = compute_frobenius_norm(A, B)
        
        safety = SafetyInvariants(max_weight_norm=threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
        
        expected_pass = percent_of_threshold <= 100.0
        is_violation = actual_norm > threshold
        detected = (is_violation and not result.passed) or (not is_violation and result.passed)
        
        if expected_pass:
            passed = result.passed
            status = "ACCEPTED" if passed else "WRONGLY REJECTED"
        else:
            passed = not result.passed
            status = "REJECTED" if passed else "WRONGLY ACCEPTED"
        
        return passed, f"{percent_of_threshold}% of threshold, norm={actual_norm:.4f}, {status}", actual_norm, threshold, detected
    
    return test_func


# =============================================================================
# MULTI-LAYER TESTS
# =============================================================================

def create_multi_layer_test(num_layers: int, scale: float, expected_pass: bool, seed: int):
    """Factory to create multi-layer tests."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        np.random.seed(seed)
        
        weights = {}
        for i in range(num_layers):
            A = np.random.randn(8, 768) * scale
            B = np.random.randn(768, 8) * scale
            weights[f"layer_{i}"] = (A, B)
        
        safety = SafetyInvariants(max_weight_norm=threshold)
        monitor = WeightMonitor(safety)
        total_norm = monitor.get_total_norm(weights)
        result = monitor.verify_invariants(weights, "hash")
        
        is_violation = total_norm > threshold
        detected = (is_violation and not result.passed) or (not is_violation and result.passed)
        
        if expected_pass:
            passed = result.passed
        else:
            passed = not result.passed
        
        return passed, f"{num_layers} layers, total_norm={total_norm:.4f}", total_norm, threshold, detected
    
    return test_func


# =============================================================================
# TAMPERING TESTS
# =============================================================================

def create_tamper_test(hash_diff_type: str, seed: int):
    """Factory to create tampering tests."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        np.random.seed(seed)
        A = np.random.randn(8, 768) * 0.001
        B = np.random.randn(768, 8) * 0.001
        
        norm = compute_frobenius_norm(A, B)
        
        expected_hash = "a" * 64
        
        if hash_diff_type == "clean":
            provided_hash = expected_hash
            should_pass = True
        elif hash_diff_type == "single_bit":
            provided_hash = "a" * 63 + "b"
            should_pass = False
        elif hash_diff_type == "first_char":
            provided_hash = "b" + "a" * 63
            should_pass = False
        elif hash_diff_type == "middle_char":
            provided_hash = "a" * 32 + "b" + "a" * 31
            should_pass = False
        elif hash_diff_type == "completely_different":
            provided_hash = "b" * 64
            should_pass = False
        elif hash_diff_type == "truncated":
            provided_hash = "a" * 32
            should_pass = False
        else:
            provided_hash = "x" * 64
            should_pass = False
        
        safety = SafetyInvariants(max_weight_norm=threshold, expected_model_hash=expected_hash)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, provided_hash)
        
        # Tampering is always a violation if hash doesn't match
        is_tampering = provided_hash != expected_hash
        detected = (is_tampering and not result.passed) or (not is_tampering and result.passed)
        
        if should_pass:
            passed = result.passed
            status = "ACCEPTED" if passed else "WRONGLY REJECTED"
        else:
            passed = not result.passed
            status = "REJECTED" if passed else "WRONGLY ACCEPTED"
        
        return passed, f"hash_diff={hash_diff_type}, {status}", norm, threshold, detected
    
    return test_func


# =============================================================================
# ATTACK SCENARIO TESTS
# =============================================================================

def create_attack_test(attack_name: str, norm_multiplier: float, seed: int):
    """Factory to create attack scenario tests."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        target_norm = threshold * norm_multiplier
        A, B = create_weights_with_target_norm(target_norm, seed=seed)
        
        actual_norm = compute_frobenius_norm(A, B)
        
        safety = SafetyInvariants(max_weight_norm=threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
        
        # All attacks should exceed threshold (multiplier > 1.0)
        should_be_rejected = norm_multiplier > 1.0
        is_violation = actual_norm > threshold
        detected = is_violation and not result.passed
        
        if should_be_rejected:
            passed = not result.passed
            status = "BLOCKED" if passed else "MISSED!"
        else:
            passed = result.passed
            status = "ALLOWED" if passed else "FALSE POSITIVE"
        
        return passed, f"{attack_name}: norm={actual_norm:.2f} ({norm_multiplier*100:.0f}%), {status}", actual_norm, threshold, detected
    
    return test_func


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def create_perf_test(operation: str, target_ms: float, iterations: int = 20):
    """Factory to create performance tests."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        np.random.seed(42)
        A = np.random.randn(8, 768) * 0.001
        B = np.random.randn(768, 8) * 0.001
        
        safety = SafetyInvariants(max_weight_norm=threshold)
        monitor = WeightMonitor(safety)
        weights = {"layer_0": (A, B)}
        
        if operation == "norm":
            compute_frobenius_norm(A, B)
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                compute_frobenius_norm(A, B)
                times.append((time.perf_counter() - start) * 1000)
        
        elif operation == "verify":
            monitor.verify_invariants(weights, "hash")
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                monitor.verify_invariants(weights, "hash")
                times.append((time.perf_counter() - start) * 1000)
        
        elif operation == "proof_gen":
            proof_gen = MockProofGenerator("./test_perf_proof")
            proof_gen.setup(A.shape, B.shape, threshold)
            proof_gen.generate_proof(A, B, "hash")
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                proof_gen.generate_proof(A, B, "hash")
                times.append((time.perf_counter() - start) * 1000)
        
        elif operation == "proof_verify":
            proof_gen = MockProofGenerator("./test_perf_verify")
            proof_gen.setup(A.shape, B.shape, threshold)
            proof_gen.generate_proof(A, B, "hash")
            proof_gen.verify_proof("./test_perf_verify/mock_proof.json")
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                proof_gen.verify_proof("./test_perf_verify/mock_proof.json")
                times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        passed = avg_time < target_ms
        
        return passed, f"avg={avg_time:.2f}ms (±{std_time:.2f}), target<{target_ms}ms", 0.0, threshold, True
    
    return test_func


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def print_scorecard(results: TestSuiteResults, verbose: bool = False):
    """Print the comprehensive test scorecard."""
    
    print("\n")
    print("+" + "=" * 78 + "+")
    print("|" + " " * 23 + "PROTOCOL-VERIFY TEST SCORECARD" + " " * 24 + "|")
    print("|" + " " * 20 + "AI Governance ZK-ML Verification" + " " * 23 + "|")
    print("+" + "=" * 78 + "+")
    
    print(f"|  Total Tests:      {results.total_tests:>4}                                                       |")
    print(f"|  Passed:           {results.passed_tests:>4}                                                       |")
    print(f"|  Failed:           {results.failed_tests:>4}                                                       |")
    print(f"|  Pass Rate:       {results.pass_rate:>5.1f}%                                                      |")
    print(f"|  Total Time:      {results.total_duration_ms:>7.0f}ms                                                   |")
    print("+" + "=" * 78 + "+")
    
    categories = [
        ("lr_sweep", "LEARNING RATE SWEEP", "Tests across LR 1e-6 to 1.0"),
        ("rank", "LORA RANK TESTS", "Tests across ranks 2 to 256"),
        ("model", "MODEL ARCHITECTURE", "Tests across 10 model configs"),
        ("boundary", "BOUNDARY CONDITIONS", "Tests at threshold boundaries"),
        ("multi_layer", "MULTI-LAYER TESTS", "Tests with 1-15 LoRA layers"),
        ("tamper", "TAMPERING DETECTION", "Base model integrity tests"),
        ("attack", "ATTACK SCENARIOS", "Real-world attack simulations"),
        ("perf", "PERFORMANCE", "Speed benchmarks"),
    ]
    
    for cat_key, cat_name, cat_desc in categories:
        cat_tests = results.by_category(cat_key)
        if not cat_tests:
            continue
            
        cat_passed = sum(1 for t in cat_tests if t.passed)
        cat_total = len(cat_tests)
        
        status = "[PASS]" if cat_passed == cat_total else "[FAIL]"
        
        print(f"|                                                                              |")
        print(f"|  {status} {cat_name:<35} [{cat_passed:>2}/{cat_total:<2}]                     |")
    
    print("+" + "=" * 78 + "+")
    
    # Detection stats
    violations = [t for t in results.tests if t.norm > t.threshold]
    detected = sum(1 for t in violations if t.detected)
    
    print(f"|                                                                              |")
    print(f"|  SECURITY METRICS:                                                           |")
    print(f"|    Violations in test set: {len(violations):>3}                                               |")
    print(f"|    Violations detected:    {detected:>3} ({detected/len(violations)*100 if violations else 100:.0f}%)                                         |")
    print(f"|    False negatives:        {len(violations) - detected:>3}                                               |")
    print(f"|                                                                              |")
    
    if results.pass_rate == 100:
        print("|" + " " * 78 + "|")
        print("|" + " " * 12 + "*** ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT ***" + " " * 11 + "|")
        print("|" + " " * 78 + "|")
    elif results.pass_rate >= 90:
        print("|" + " " * 78 + "|")
        print("|" + " " * 17 + "WARNING: MOSTLY PASSING - REVIEW FAILED TESTS" + " " * 16 + "|")
        print("|" + " " * 78 + "|")
    else:
        print("|" + " " * 78 + "|")
        print("|" + " " * 12 + "!!! CRITICAL FAILURES - SECURITY ISSUES DETECTED !!!" + " " * 13 + "|")
        print("|" + " " * 78 + "|")
    
    print("+" + "=" * 78 + "+")
    
    if results.failed_tests > 0 and verbose:
        print("\nFAILED TEST DETAILS:")
        print("-" * 70)
        for test in results.tests:
            if not test.passed:
                print(f"  [X] {test.name}")
                print(f"      Details: {test.details}")
                print()


def run_test_suite(skip_perf: bool = False, verbose: bool = False) -> TestSuiteResults:
    """Run the comprehensive test suite."""
    
    results = TestSuiteResults()
    
    print("\n" + "=" * 80)
    print("PROTOCOL-VERIFY: MASSIVE COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    tests = []
    
    # ==========================================================================
    # LEARNING RATE SWEEP (20 tests)
    # ==========================================================================
    lr_tests = [
        # Should PASS (compliant)
        (1e-6, True, 101),
        (1e-5, True, 102),
        (5e-5, True, 103),
        (1e-4, True, 104),
        (2e-4, True, 105),
        (5e-4, True, 106),
        (1e-3, True, 107),
        (2e-3, True, 108),
        (3e-3, True, 109),
        (5e-3, True, 110),
        # Should FAIL (violations)
        (1e-2, False, 111),
        (2e-2, False, 112),
        (3e-2, False, 113),
        (5e-2, False, 114),
        (0.1, False, 115),
        (0.15, False, 116),
        (0.2, False, 117),
        (0.5, False, 118),
        (0.8, False, 119),
        (1.0, False, 120),
    ]
    
    for lr, should_pass, seed in lr_tests:
        status = "compliant" if should_pass else "violation"
        tests.append((f"LR={lr} ({status})", "lr_sweep", create_lr_test(lr, should_pass, seed)))
    
    # ==========================================================================
    # LORA RANK TESTS ACROSS MODEL ARCHITECTURES (40 tests)
    # ==========================================================================
    lora_ranks = [2, 4, 8, 16, 32, 64, 128, 256]
    
    for model_name, config in MODEL_CONFIGS.items():
        hidden = config["hidden"]
        # Test each rank at safe scale
        for rank in [4, 8, 16, 32]:
            seed = 200 + hash(f"{model_name}_{rank}") % 1000
            tests.append(
                (f"{model_name}: r={rank} (safe)", "model", 
                 create_rank_test(rank, hidden, 0.001, True, seed, model_name))
            )
    
    # Test violation scenarios with different ranks
    for rank in lora_ranks:
        seed = 300 + rank
        scale = 0.001 if rank <= 8 else (0.0008 if rank <= 32 else 0.0005)
        tests.append((f"Rank={rank}, safe scale", "rank", create_rank_test(rank, 768, scale, True, seed)))
    
    # Violation tests
    for rank in [8, 16, 32, 64]:
        seed = 350 + rank
        tests.append((f"Rank={rank}, excessive scale", "rank", create_rank_test(rank, 768, 0.1, False, seed)))
    
    # ==========================================================================
    # BOUNDARY CONDITION TESTS (25 tests)
    # ==========================================================================
    boundary_percentages = [
        10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 
        92, 94, 96, 98, 99, 99.5, 99.9, 100,
        100.01, 100.1, 100.5, 101, 102, 105, 110
    ]
    
    for i, pct in enumerate(boundary_percentages):
        status = "pass" if pct <= 100 else "fail"
        tests.append((f"Threshold {pct}% ({status})", "boundary", create_boundary_test(pct, 400 + i)))
    
    # ==========================================================================
    # MULTI-LAYER TESTS (15 tests)
    # ==========================================================================
    # Compliant multi-layer
    for num_layers in [1, 2, 3, 4, 6, 8, 10, 12, 15]:
        seed = 500 + num_layers
        tests.append((f"{num_layers} layers, safe", "multi_layer", create_multi_layer_test(num_layers, 0.001, True, seed)))
    
    # Violation multi-layer
    for num_layers in [4, 6, 8, 10, 12, 15]:
        seed = 550 + num_layers
        tests.append((f"{num_layers} layers, excessive", "multi_layer", create_multi_layer_test(num_layers, 0.05, False, seed)))
    
    # ==========================================================================
    # TAMPERING DETECTION TESTS (10 tests)
    # ==========================================================================
    tamper_types = [
        ("clean", 601),
        ("single_bit", 602),
        ("first_char", 603),
        ("middle_char", 604),
        ("completely_different", 605),
        ("truncated", 606),
        ("random_1", 607),
        ("random_2", 608),
        ("random_3", 609),
        ("random_4", 610),
    ]
    
    for tamper_type, seed in tamper_types:
        tests.append((f"Hash tampering: {tamper_type}", "tamper", create_tamper_test(tamper_type, seed)))
    
    # ==========================================================================
    # ATTACK SCENARIO TESTS (15 tests)
    # ==========================================================================
    attack_scenarios = [
        # Legitimate usage (should pass)
        ("Minimal finetune", 0.1, 701),
        ("Light finetune", 0.3, 702),
        ("Standard finetune", 0.5, 703),
        ("Heavy finetune", 0.8, 704),
        ("Maximum safe", 0.99, 705),  # Just under threshold to avoid float precision issues
        
        # Attack scenarios (should be blocked)
        ("Slight overstep", 1.01, 706),
        ("5% over limit", 1.05, 707),
        ("10% over limit", 1.1, 708),
        ("20% over limit", 1.2, 709),
        ("50% over limit", 1.5, 710),
        ("2x limit (evasion attempt)", 2.0, 711),
        ("5x limit (major violation)", 5.0, 712),
        ("10x limit (catastrophic)", 10.0, 713),
        ("50x limit (adversarial)", 50.0, 714),
        ("100x limit (malicious)", 100.0, 715),
    ]
    
    for attack_name, multiplier, seed in attack_scenarios:
        tests.append((attack_name, "attack", create_attack_test(attack_name, multiplier, seed)))
    
    # ==========================================================================
    # PERFORMANCE TESTS (8 tests)
    # ==========================================================================
    if not skip_perf:
        perf_tests = [
            ("norm", 10),
            ("norm", 50),
            ("verify", 50),
            ("verify", 100),
            ("proof_gen", 100),
            ("proof_gen", 500),
            ("proof_verify", 50),
            ("proof_verify", 100),
        ]
        
        for op, target in perf_tests:
            tests.append((f"Perf: {op} <{target}ms", "perf", create_perf_test(op, target)))
    
    # Run all tests
    total = len(tests)
    print(f"\nRunning {total} tests...\n")
    
    for i, (name, category, test_func) in enumerate(tests):
        print(f"  [{i+1:>3}/{total}] {name[:50]:<50}", end=" ", flush=True)
        result = run_test(name, category, test_func)
        results.add(result)
        
        status = "[OK]" if result.passed else "[FAIL]"
        print(f"{status} ({result.duration_ms:.1f}ms)")
        
        if verbose and result.details:
            print(f"         -> {result.details}")
    
    results.end_time = datetime.now()
    
    return results


def export_results_for_visualization(results: TestSuiteResults, output_path: str = "test_results.json"):
    """Export results for visualization."""
    
    data = {
        "timestamp": results.start_time.isoformat(),
        "total_tests": results.total_tests,
        "passed": results.passed_tests,
        "failed": results.failed_tests,
        "pass_rate": results.pass_rate,
        "total_duration_ms": results.total_duration_ms,
        "categories": {},
        "tests": []
    }
    
    # Group by category
    for test in results.tests:
        if test.category not in data["categories"]:
            data["categories"][test.category] = {"passed": 0, "total": 0, "tests": []}
        
        data["categories"][test.category]["total"] += 1
        if test.passed:
            data["categories"][test.category]["passed"] += 1
        
        data["categories"][test.category]["tests"].append({
            "name": test.name,
            "passed": test.passed,
            "norm": test.norm,
            "threshold": test.threshold,
            "detected": test.detected,
            "duration_ms": test.duration_ms,
        })
        
        data["tests"].append({
            "name": test.name,
            "category": test.category,
            "passed": test.passed,
            "duration_ms": test.duration_ms,
            "details": test.details,
            "norm": test.norm,
            "threshold": test.threshold,
            "detected": test.detected,
        })
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults exported to: {output_path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Protocol-Verify Massive Test Suite")
    parser.add_argument("-q", "--quick", action="store_true", help="Skip performance tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--json", type=str, default="test_results.json", help="Export results to JSON file")
    
    args = parser.parse_args()
    
    results = run_test_suite(skip_perf=args.quick, verbose=args.verbose)
    print_scorecard(results, verbose=args.verbose)
    
    export_results_for_visualization(results, args.json)
    
    sys.exit(0 if results.pass_rate == 100 else 1)


if __name__ == "__main__":
    main()
