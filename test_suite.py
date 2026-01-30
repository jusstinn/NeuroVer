#!/usr/bin/env python3
"""
Protocol-Verify: Comprehensive Test Suite

Extensive testing covering:
- Multiple learning rates (1e-5 to 1.0)
- Multiple LoRA ranks (4, 8, 16, 32, 64)
- Boundary conditions (99%, 100%, 101% of threshold)
- Various attack vectors
- Performance benchmarks
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
        passed, details, norm, threshold = test_func()
    except Exception as e:
        passed, details, norm, threshold = False, f"Exception: {e}", 0.0, 10.0
    duration = (time.perf_counter() - start) * 1000
    
    return TestResult(
        name=name, passed=passed, duration_ms=duration,
        details=details, category=category, norm=norm, threshold=threshold
    )


def create_weights_with_target_norm(target_norm: float, r: int = 8, d: int = 768, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create weight matrices A, B such that ||BA||_F ≈ target_norm."""
    np.random.seed(seed)
    
    base_A = np.random.randn(r, d)
    base_B = np.random.randn(d, r)
    
    P = base_B @ base_A
    current_norm = np.sqrt(np.sum(P ** 2))
    
    # Scale to hit target: ||sB @ sA|| = s^2 * ||BA||
    scale = np.sqrt(target_norm / current_norm)
    
    return base_A * scale, base_B * scale


# =============================================================================
# LEARNING RATE SWEEP TESTS
# =============================================================================

def create_lr_test(lr_value: float, expected_pass: bool, seed: int):
    """Factory to create learning rate tests."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        np.random.seed(seed)
        # Simulate weight magnitude proportional to learning rate
        # LR=1e-4 -> scale=0.001, LR=1e-3 -> scale=0.003, etc.
        scale = lr_value * 10  # Maps LR to reasonable weight scale
        
        A = np.random.randn(8, 768) * scale
        B = np.random.randn(768, 8) * scale
        
        norm = compute_frobenius_norm(A, B)
        
        safety = SafetyInvariants(max_weight_norm=threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
        
        if expected_pass:
            passed = result.passed
            status = "ACCEPTED" if passed else "WRONGLY REJECTED"
        else:
            passed = not result.passed
            status = "REJECTED" if passed else "WRONGLY ACCEPTED"
        
        return passed, f"LR={lr_value}, norm={norm:.4f}, {status}", norm, threshold
    
    return test_func


# =============================================================================
# LORA RANK TESTS
# =============================================================================

def create_rank_test(rank: int, scale: float, expected_pass: bool, seed: int):
    """Factory to create LoRA rank tests."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        np.random.seed(seed)
        d = 768
        
        A = np.random.randn(rank, d) * scale
        B = np.random.randn(d, rank) * scale
        
        norm = compute_frobenius_norm(A, B)
        
        safety = SafetyInvariants(max_weight_norm=threshold)
        monitor = WeightMonitor(safety)
        result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
        
        if expected_pass:
            passed = result.passed
        else:
            passed = not result.passed
        
        return passed, f"rank={rank}, norm={norm:.4f}", norm, threshold
    
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
        if expected_pass:
            passed = result.passed
            status = "ACCEPTED" if passed else "WRONGLY REJECTED"
        else:
            passed = not result.passed
            status = "REJECTED" if passed else "WRONGLY ACCEPTED"
        
        return passed, f"{percent_of_threshold}% of threshold, norm={actual_norm:.4f}, {status}", actual_norm, threshold
    
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
        
        if expected_pass:
            passed = result.passed
        else:
            passed = not result.passed
        
        return passed, f"{num_layers} layers, total_norm={total_norm:.4f}", total_norm, threshold
    
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
        
        if should_pass:
            passed = result.passed
            status = "ACCEPTED" if passed else "WRONGLY REJECTED"
        else:
            passed = not result.passed
            status = "REJECTED" if passed else "WRONGLY ACCEPTED"
        
        return passed, f"hash_diff={hash_diff_type}, {status}", norm, threshold
    
    return test_func


# =============================================================================
# PROOF GENERATION TESTS
# =============================================================================

def create_proof_test(norm_multiplier: float, should_be_compliant: bool, seed: int):
    """Factory to create proof generation tests."""
    def test_func():
        config = load_safety_config()
        threshold = config["weight_constraints"]["max_weight_norm"]
        
        target_norm = threshold * norm_multiplier
        A, B = create_weights_with_target_norm(target_norm, seed=seed)
        
        actual_norm = compute_frobenius_norm(A, B)
        
        proof_gen = MockProofGenerator(f"./test_proof_{seed}")
        proof_gen.setup(A.shape, B.shape, threshold)
        proof = proof_gen.generate_proof(A, B, "hash")
        
        is_compliant = proof["public_inputs"]["result"] == 1
        
        if should_be_compliant:
            passed = is_compliant
            status = "COMPLIANT" if passed else "WRONGLY NON-COMPLIANT"
        else:
            passed = not is_compliant
            status = "NON-COMPLIANT" if passed else "WRONGLY COMPLIANT"
        
        return passed, f"norm={actual_norm:.4f}, proof={status}", actual_norm, threshold
    
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
            # Warmup
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
        
        return passed, f"avg={avg_time:.2f}ms (±{std_time:.2f}), target<{target_ms}ms", 0.0, threshold
    
    return test_func


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def print_scorecard(results: TestSuiteResults, verbose: bool = False):
    """Print the comprehensive test scorecard."""
    
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 23 + "PROTOCOL-VERIFY TEST SCORECARD" + " " * 24 + "║")
    print("║" + " " * 20 + "AI Governance ZK-ML Verification" + " " * 23 + "║")
    print("╠" + "═" * 78 + "╣")
    
    print(f"║  📊 Total Tests:      {results.total_tests:>4}                                                     ║")
    print(f"║  ✅ Passed:           {results.passed_tests:>4}                                                     ║")
    print(f"║  ❌ Failed:           {results.failed_tests:>4}                                                     ║")
    print(f"║  📈 Pass Rate:       {results.pass_rate:>5.1f}%                                                    ║")
    print(f"║  ⏱️  Total Time:      {results.total_duration_ms:>7.0f}ms                                                 ║")
    print("╠" + "═" * 78 + "╣")
    
    categories = [
        ("lr_sweep", "🎚️  LEARNING RATE SWEEP", "Tests across LR 1e-5 to 1.0"),
        ("rank", "📐 LORA RANK TESTS", "Tests across ranks 4 to 64"),
        ("boundary", "📍 BOUNDARY CONDITIONS", "Tests at threshold boundaries"),
        ("multi_layer", "🗂️  MULTI-LAYER TESTS", "Tests with 1-12 LoRA layers"),
        ("tamper", "🔓 TAMPERING DETECTION", "Base model integrity tests"),
        ("proof", "🔐 PROOF GENERATION", "ZK proof tests"),
        ("perf", "⚡ PERFORMANCE", "Speed benchmarks"),
    ]
    
    for cat_key, cat_name, cat_desc in categories:
        cat_tests = results.by_category(cat_key)
        if not cat_tests:
            continue
            
        cat_passed = sum(1 for t in cat_tests if t.passed)
        cat_total = len(cat_tests)
        
        status = "✅" if cat_passed == cat_total else "❌"
        
        print(f"║                                                                              ║")
        print(f"║  {status} {cat_name:<35} [{cat_passed:>2}/{cat_total:<2}]                   ║")
        
        if verbose:
            for test in cat_tests:
                t_status = "✓" if test.passed else "✗"
                name_short = test.name[:42]
                print(f"║     {t_status} {name_short:<42} {test.duration_ms:>6.1f}ms    ║")
    
    print("╠" + "═" * 78 + "╣")
    
    if results.pass_rate == 100:
        print("║" + " " * 78 + "║")
        print("║" + " " * 15 + "🏆 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT 🏆" + " " * 12 + "║")
        print("║" + " " * 78 + "║")
    elif results.pass_rate >= 90:
        print("║" + " " * 78 + "║")
        print("║" + " " * 20 + "⚠️  MOSTLY PASSING - REVIEW FAILED TESTS" + " " * 17 + "║")
        print("║" + " " * 78 + "║")
    else:
        print("║" + " " * 78 + "║")
        print("║" + " " * 15 + "🚨 CRITICAL FAILURES - SECURITY ISSUES DETECTED 🚨" + " " * 13 + "║")
        print("║" + " " * 78 + "║")
    
    print("╚" + "═" * 78 + "╝")
    
    if results.failed_tests > 0:
        print("\n📋 FAILED TEST DETAILS:")
        print("-" * 70)
        for test in results.tests:
            if not test.passed:
                print(f"  ❌ {test.name}")
                print(f"     Details: {test.details}")
                print()


def run_test_suite(skip_perf: bool = False, verbose: bool = False) -> TestSuiteResults:
    """Run the comprehensive test suite."""
    
    results = TestSuiteResults()
    
    print("\n" + "=" * 80)
    print("🧪 PROTOCOL-VERIFY: COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    tests = []
    
    # ==========================================================================
    # LEARNING RATE SWEEP (15 tests)
    # ==========================================================================
    lr_tests = [
        # Should PASS (compliant) - LR * 10 * randn gives small norms
        (1e-5, True, 101),
        (5e-5, True, 102),
        (1e-4, True, 103),
        (5e-4, True, 104),
        (1e-3, True, 105),
        (2e-3, True, 106),
        (5e-3, True, 107),
        # Should FAIL (violations) - higher LRs exceed threshold
        (1e-2, False, 108),
        (2e-2, False, 109),
        (5e-2, False, 110),
        (0.1, False, 111),
        (0.2, False, 112),
        (0.5, False, 113),
        (0.8, False, 114),
        (1.0, False, 115),
    ]
    
    for lr, should_pass, seed in lr_tests:
        status = "compliant" if should_pass else "violation"
        tests.append((f"LR={lr} ({status})", "lr_sweep", create_lr_test(lr, should_pass, seed)))
    
    # ==========================================================================
    # LORA RANK TESTS (10 tests)
    # ==========================================================================
    rank_tests = [
        # Small scale = compliant
        (4, 0.001, True, 201),
        (8, 0.001, True, 202),
        (16, 0.001, True, 203),
        (32, 0.001, True, 204),
        (64, 0.001, True, 205),
        # Medium scale = still compliant (norms ~4-6)
        (4, 0.05, True, 206),
        (8, 0.05, True, 207),
        (16, 0.04, True, 208),
        (32, 0.03, True, 209),
        (64, 0.02, True, 210),
        # Large scale = violations (norms > 10)
        (8, 0.1, False, 211),
        (16, 0.08, False, 212),
        (32, 0.06, False, 213),
        (64, 0.05, False, 214),
    ]
    
    for rank, scale, should_pass, seed in rank_tests:
        status = "compliant" if should_pass else "violation"
        tests.append((f"Rank={rank}, scale={scale} ({status})", "rank", create_rank_test(rank, scale, should_pass, seed)))
    
    # ==========================================================================
    # BOUNDARY CONDITION TESTS (15 tests)
    # ==========================================================================
    boundary_percentages = [
        50, 60, 70, 80, 90, 95, 98, 99, 99.5, 100,  # Should pass
        100.1, 100.5, 101, 105, 110, 150, 200, 500   # Should fail
    ]
    
    for i, pct in enumerate(boundary_percentages):
        status = "pass" if pct <= 100 else "fail"
        tests.append((f"Threshold {pct}% ({status})", "boundary", create_boundary_test(pct, 300 + i)))
    
    # ==========================================================================
    # MULTI-LAYER TESTS (10 tests)
    # ==========================================================================
    multi_layer_tests = [
        # Compliant multi-layer (small scales)
        (1, 0.001, True, 401),
        (2, 0.001, True, 402),
        (4, 0.001, True, 403),
        (6, 0.001, True, 404),
        (12, 0.001, True, 405),
        # Medium scale - still compliant
        (2, 0.01, True, 406),
        (4, 0.01, True, 407),
        # Large scale - violations (must exceed threshold with accumulated norm)
        (6, 0.05, False, 408),
        (8, 0.05, False, 409),
        (12, 0.05, False, 410),
    ]
    
    for layers, scale, should_pass, seed in multi_layer_tests:
        status = "compliant" if should_pass else "violation"
        tests.append((f"{layers} layers, scale={scale} ({status})", "multi_layer", create_multi_layer_test(layers, scale, should_pass, seed)))
    
    # ==========================================================================
    # TAMPERING DETECTION TESTS (8 tests)
    # ==========================================================================
    tamper_types = [
        ("clean", 501),
        ("single_bit", 502),
        ("first_char", 503),
        ("middle_char", 504),
        ("completely_different", 505),
        ("truncated", 506),
        ("random_1", 507),
        ("random_2", 508),
    ]
    
    for tamper_type, seed in tamper_types:
        tests.append((f"Hash tampering: {tamper_type}", "tamper", create_tamper_test(tamper_type, seed)))
    
    # ==========================================================================
    # PROOF GENERATION TESTS (10 tests)
    # ==========================================================================
    proof_tests = [
        # Compliant proofs
        (0.1, True, 601),
        (0.3, True, 602),
        (0.5, True, 603),
        (0.7, True, 604),
        (0.9, True, 605),
        (1.0, True, 606),
        # Non-compliant proofs
        (1.01, False, 607),
        (1.1, False, 608),
        (1.5, False, 609),
        (2.0, False, 610),
    ]
    
    for multiplier, should_comply, seed in proof_tests:
        status = "compliant" if should_comply else "violation"
        tests.append((f"Proof at {multiplier*100:.0f}% threshold ({status})", "proof", create_proof_test(multiplier, should_comply, seed)))
    
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
    for i, (name, category, test_func) in enumerate(tests):
        print(f"  [{i+1:>3}/{total}] {name[:50]:<50}", end=" ", flush=True)
        result = run_test(name, category, test_func)
        results.add(result)
        
        status = "✅" if result.passed else "❌"
        print(f"{status} ({result.duration_ms:.1f}ms)")
        
        if verbose and result.details:
            print(f"         → {result.details}")
    
    results.end_time = datetime.now()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Protocol-Verify Comprehensive Test Suite")
    parser.add_argument("-q", "--quick", action="store_true", help="Skip performance tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--json", type=str, help="Export results to JSON file")
    
    args = parser.parse_args()
    
    results = run_test_suite(skip_perf=args.quick, verbose=args.verbose)
    print_scorecard(results, verbose=args.verbose)
    
    if args.json:
        export_data = {
            "timestamp": results.start_time.isoformat(),
            "total_tests": results.total_tests,
            "passed": results.passed_tests,
            "failed": results.failed_tests,
            "pass_rate": results.pass_rate,
            "total_duration_ms": results.total_duration_ms,
            "tests": [
                {
                    "name": t.name,
                    "category": t.category,
                    "passed": t.passed,
                    "duration_ms": t.duration_ms,
                    "details": t.details,
                    "norm": t.norm,
                    "threshold": t.threshold,
                }
                for t in results.tests
            ]
        }
        with open(args.json, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n📁 Results exported to: {args.json}")
    
    sys.exit(0 if results.pass_rate == 100 else 1)


if __name__ == "__main__":
    main()
