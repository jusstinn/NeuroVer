#!/usr/bin/env python3
"""
Protocol-Verify: Master Test Suite

This script runs all verification tests and produces a scorecard
for hackathon judges demonstrating that the ZK-ML system correctly:

1. ✅ ACCEPTS compliant (honest) training runs
2. ❌ REJECTS training that exceeds weight norm limits
3. ❌ REJECTS base model tampering attempts
4. ⚡ Verifies proofs in <100ms

Usage:
    python test_suite.py              # Run all tests
    python test_suite.py --quick      # Skip performance tests
    python test_suite.py --verbose    # Extra output
"""

import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from core.monitor import WeightMonitor, SafetyInvariants, compute_frobenius_norm
from core.proof_gen import MockProofGenerator


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    details: str = ""
    category: str = "general"


@dataclass
class TestSuiteResults:
    """Results from the complete test suite."""
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
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests * 100
    
    @property
    def total_duration_ms(self) -> float:
        return sum(t.duration_ms for t in self.tests)
    
    def by_category(self, category: str) -> List[TestResult]:
        return [t for t in self.tests if t.category == category]


def load_safety_config() -> dict:
    """Load safety configuration."""
    config_path = Path(__file__).parent / "policy" / "safety_config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def run_test(name: str, category: str, test_func) -> TestResult:
    """Run a single test and capture results."""
    start = time.perf_counter()
    try:
        passed, details = test_func()
    except Exception as e:
        passed = False
        details = f"Exception: {e}"
    duration = (time.perf_counter() - start) * 1000
    
    return TestResult(
        name=name,
        passed=passed,
        duration_ms=duration,
        details=details,
        category=category,
    )


# =============================================================================
# HONEST ACTOR TESTS: System correctly ACCEPTS compliant training
# =============================================================================

def test_honest_weight_norms():
    """Test that honest training produces compliant weight norms."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(42)
    r, d = 8, 768
    A = np.random.randn(r, d) * 0.001
    B = np.random.randn(d, r) * 0.001
    
    norm = compute_frobenius_norm(A, B)
    passed = norm <= threshold
    
    return passed, f"norm={norm:.4f} <= threshold={threshold}"


def test_honest_verification():
    """Test that honest training passes verification."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    safety = SafetyInvariants(max_weight_norm=threshold)
    
    np.random.seed(42)
    weights = {"layer_0": (np.random.randn(8, 768) * 0.001, np.random.randn(768, 8) * 0.001)}
    
    monitor = WeightMonitor(safety)
    result = monitor.verify_invariants(weights, "clean_hash_12345")
    
    return result.passed, f"verification_passed={result.passed}"


def test_honest_proof_generation():
    """Test that honest training generates compliant proof."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(42)
    A = np.random.randn(8, 768) * 0.001
    B = np.random.randn(768, 8) * 0.001
    
    proof_gen = MockProofGenerator("./test_suite_honest")
    proof_gen.setup(A.shape, B.shape, threshold)
    proof = proof_gen.generate_proof(A, B, "clean_hash")
    
    passed = proof["public_inputs"]["result"] == 1
    return passed, f"proof_result={proof['public_inputs']['result']} (1=compliant)"


def test_honest_proof_verification():
    """Test that honest proof is verified as valid."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(42)
    A = np.random.randn(8, 768) * 0.001
    B = np.random.randn(768, 8) * 0.001
    
    proof_gen = MockProofGenerator("./test_suite_honest_verify")
    proof_gen.setup(A.shape, B.shape, threshold)
    proof_gen.generate_proof(A, B, "clean_hash")
    
    is_valid = proof_gen.verify_proof("./test_suite_honest_verify/mock_proof.json")
    return is_valid, f"verification_valid={is_valid}"


# =============================================================================
# FAILURE TESTS: System correctly REJECTS non-compliant training
# =============================================================================

def test_REJECT_10_percent_over_threshold():
    """Training 10% over limit should be REJECTED."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(100)
    r, d = 8, 768
    
    # Create base matrices with unit variance
    base_A = np.random.randn(r, d)
    base_B = np.random.randn(d, r)
    
    # Compute the current P = B @ A norm
    P = base_B @ base_A
    current_norm = np.sqrt(np.sum(P ** 2))
    
    # Scale both A and B so that ||B @ A|| = 110% of threshold
    # If we scale A by s and B by s, then ||sB @ sA|| = s^2 * ||B @ A||
    target_norm = threshold * 1.1
    scale = np.sqrt(target_norm / current_norm)
    
    A = base_A * scale
    B = base_B * scale
    norm = compute_frobenius_norm(A, B)
    
    safety = SafetyInvariants(max_weight_norm=threshold)
    monitor = WeightMonitor(safety)
    result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
    
    rejected = not result.passed
    return rejected, f"norm={norm:.2f} (target 110% of {threshold}={threshold*1.1:.1f}) → REJECTED={rejected}"


def test_REJECT_50_percent_over_threshold():
    """Training 50% over limit should be REJECTED."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(200)
    r, d = 8, 768
    
    base_A = np.random.randn(r, d)
    base_B = np.random.randn(d, r)
    
    P = base_B @ base_A
    current_norm = np.sqrt(np.sum(P ** 2))
    
    # Scale to 150% of threshold
    target_norm = threshold * 1.5
    scale = np.sqrt(target_norm / current_norm)
    
    A = base_A * scale
    B = base_B * scale
    norm = compute_frobenius_norm(A, B)
    
    safety = SafetyInvariants(max_weight_norm=threshold)
    monitor = WeightMonitor(safety)
    result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
    
    rejected = not result.passed
    return rejected, f"norm={norm:.2f} (target 150% of {threshold}={threshold*1.5:.1f}) → REJECTED={rejected}"


def test_REJECT_explosive_weights():
    """Catastrophic weight explosion (LR=1.0) should be REJECTED."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(666)
    A = np.random.randn(8, 768) * 0.1  # ~100x larger than normal
    B = np.random.randn(768, 8) * 0.1
    
    norm = compute_frobenius_norm(A, B)
    
    safety = SafetyInvariants(max_weight_norm=threshold)
    monitor = WeightMonitor(safety)
    result = monitor.verify_invariants({"layer_0": (A, B)}, "hash")
    
    rejected = not result.passed
    return rejected, f"norm={norm:.2f} ({norm/threshold:.1f}x threshold) → REJECTED={rejected}"


def test_REJECT_proof_non_compliant():
    """Proof for excessive weights should show result=0."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(666)
    A = np.random.randn(8, 768) * 0.1
    B = np.random.randn(768, 8) * 0.1
    
    proof_gen = MockProofGenerator("./test_suite_reject_proof")
    proof_gen.setup(A.shape, B.shape, threshold)
    proof = proof_gen.generate_proof(A, B, "hash")
    
    non_compliant = proof["public_inputs"]["result"] == 0
    return non_compliant, f"proof_result={proof['public_inputs']['result']} (0=non-compliant)"


def test_REJECT_verification_false():
    """Verification should return FALSE for non-compliant training."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(666)
    A = np.random.randn(8, 768) * 0.1
    B = np.random.randn(768, 8) * 0.1
    
    proof_gen = MockProofGenerator("./test_suite_reject_verify")
    proof_gen.setup(A.shape, B.shape, threshold)
    proof_gen.generate_proof(A, B, "hash")
    
    is_valid = proof_gen.verify_proof("./test_suite_reject_verify/mock_proof.json")
    rejected = not is_valid
    return rejected, f"verify()={is_valid} → REJECTED={rejected}"


# =============================================================================
# BASE MODEL TAMPERING TESTS
# =============================================================================

def test_tamper_clean_accepted():
    """Test that clean model is accepted."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    clean_hash = "clean_model_hash_" + "0" * 48
    
    safety = SafetyInvariants(max_weight_norm=threshold, expected_model_hash=clean_hash)
    
    np.random.seed(42)
    weights = {"layer_0": (np.random.randn(8, 768) * 0.001, np.random.randn(768, 8) * 0.001)}
    
    monitor = WeightMonitor(safety)
    result = monitor.verify_invariants(weights, clean_hash)
    
    return result.passed, f"clean_accepted={result.passed}"


def test_REJECT_tampered_model():
    """Tampered base model should be REJECTED."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    clean_hash = "clean_model_hash_" + "0" * 48
    tampered_hash = "tampered_model_hash_" + "1" * 45
    
    safety = SafetyInvariants(max_weight_norm=threshold, expected_model_hash=clean_hash)
    
    np.random.seed(42)
    weights = {"layer_0": (np.random.randn(8, 768) * 0.001, np.random.randn(768, 8) * 0.001)}
    
    monitor = WeightMonitor(safety)
    result = monitor.verify_invariants(weights, tampered_hash)
    
    rejected = not result.passed
    return rejected, f"tampered_model → REJECTED={rejected}"


def test_REJECT_single_bit_tamper():
    """Single-bit hash difference should be REJECTED."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    clean_hash = "a" * 64
    single_bit_tampered = "a" * 63 + "b"
    
    safety = SafetyInvariants(max_weight_norm=threshold, expected_model_hash=clean_hash)
    
    np.random.seed(42)
    weights = {"layer_0": (np.random.randn(8, 768) * 0.001, np.random.randn(768, 8) * 0.001)}
    
    monitor = WeightMonitor(safety)
    result = monitor.verify_invariants(weights, single_bit_tampered)
    
    rejected = not result.passed
    return rejected, f"single_bit_diff → REJECTED={rejected}"


def test_REJECT_hidden_attack():
    """Compliant weights + tampered base should be REJECTED."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    clean_hash = "clean_base_" + "0" * 53
    tampered_hash = "tampered_base_" + "1" * 50
    
    safety = SafetyInvariants(max_weight_norm=threshold, expected_model_hash=clean_hash)
    
    np.random.seed(42)
    A = np.random.randn(8, 768) * 0.001
    B = np.random.randn(768, 8) * 0.001
    weights = {"layer_0": (A, B)}
    
    # Verify weights ARE compliant
    norm = compute_frobenius_norm(A, B)
    weights_compliant = norm < threshold
    
    # But base is tampered
    monitor = WeightMonitor(safety)
    result = monitor.verify_invariants(weights, tampered_hash)
    
    detected = weights_compliant and not result.passed
    return detected, f"weights_ok={weights_compliant}, base_tampered → REJECTED={not result.passed}"


# =============================================================================
# PERFORMANCE / STRESS TESTS
# =============================================================================

def test_verification_speed():
    """Test that verification completes in <100ms."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    safety = SafetyInvariants(max_weight_norm=threshold)
    
    np.random.seed(42)
    weights = {"layer_0": (np.random.randn(8, 768) * 0.001, np.random.randn(768, 8) * 0.001)}
    
    monitor = WeightMonitor(safety)
    
    # Warmup
    monitor.verify_invariants(weights, "hash")
    
    # Timed runs
    times = []
    for _ in range(10):
        start = time.perf_counter()
        monitor.verify_invariants(weights, "hash")
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times)
    passed = avg_time < 100
    
    return passed, f"avg_time={avg_time:.2f}ms (target <100ms)"


def test_proof_generation_speed():
    """Test proof generation performance."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(42)
    A = np.random.randn(8, 768) * 0.001
    B = np.random.randn(768, 8) * 0.001
    
    proof_gen = MockProofGenerator("./test_suite_perf")
    proof_gen.setup(A.shape, B.shape, threshold)
    
    # Warmup
    proof_gen.generate_proof(A, B, "hash")
    
    # Timed runs
    times = []
    for _ in range(10):
        start = time.perf_counter()
        proof_gen.generate_proof(A, B, "hash")
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times)
    passed = avg_time < 500  # 500ms for proof gen
    
    return passed, f"avg_time={avg_time:.2f}ms (target <500ms)"


def test_proof_verification_speed():
    """Test that proof verification completes in <100ms."""
    config = load_safety_config()
    threshold = config["weight_constraints"]["max_weight_norm"]
    
    np.random.seed(42)
    A = np.random.randn(8, 768) * 0.001
    B = np.random.randn(768, 8) * 0.001
    
    proof_gen = MockProofGenerator("./test_suite_verify_perf")
    proof_gen.setup(A.shape, B.shape, threshold)
    proof_gen.generate_proof(A, B, "hash")
    
    # Warmup
    proof_gen.verify_proof("./test_suite_verify_perf/mock_proof.json")
    
    # Timed runs
    times = []
    for _ in range(10):
        start = time.perf_counter()
        proof_gen.verify_proof("./test_suite_verify_perf/mock_proof.json")
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times)
    passed = avg_time < 100
    
    return passed, f"avg_time={avg_time:.2f}ms (target <100ms)"


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def print_scorecard(results: TestSuiteResults, verbose: bool = False):
    """Print the test scorecard for judges."""
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "PROTOCOL-VERIFY TEST SCORECARD" + " " * 19 + "║")
    print("║" + " " * 15 + "AI Governance ZK-ML Verification" + " " * 18 + "║")
    print("╠" + "═" * 68 + "╣")
    
    # Overall stats
    print(f"║  📊 Total Tests:    {results.total_tests:>3}                                          ║")
    print(f"║  ✅ Passed:         {results.passed_tests:>3}                                          ║")
    print(f"║  ❌ Failed:         {results.failed_tests:>3}                                          ║")
    print(f"║  📈 Pass Rate:      {results.pass_rate:>5.1f}%                                        ║")
    print(f"║  ⏱️  Total Time:     {results.total_duration_ms:>7.2f}ms                                  ║")
    print("╠" + "═" * 68 + "╣")
    
    # Category breakdown
    categories = [
        ("honest", "🎓 COMPLIANT TRAINING (should PASS)", "Tests that honest training is accepted"),
        ("reject", "🚫 LIMIT EXCEEDED (should REJECT)", "Tests that violations are caught"),
        ("tamper", "🔓 BASE MODEL TAMPER (should REJECT)", "Tests tampering detection"),
        ("perf", "⚡ PERFORMANCE", "Speed benchmarks"),
    ]
    
    for cat_key, cat_name, cat_desc in categories:
        cat_tests = results.by_category(cat_key)
        if not cat_tests:
            continue
            
        cat_passed = sum(1 for t in cat_tests if t.passed)
        cat_total = len(cat_tests)
        
        status = "✅" if cat_passed == cat_total else "❌"
        
        print(f"║                                                                    ║")
        print(f"║  {status} {cat_name:<40} [{cat_passed}/{cat_total}]    ║")
        
        if verbose:
            for test in cat_tests:
                t_status = "✓" if test.passed else "✗"
                name_short = test.name[:38]
                print(f"║     {t_status} {name_short:<38} {test.duration_ms:>6.2f}ms ║")
    
    print("╠" + "═" * 68 + "╣")
    
    # Final verdict
    if results.pass_rate == 100:
        print("║" + " " * 68 + "║")
        print("║        🏆 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT 🏆        ║")
        print("║" + " " * 68 + "║")
    elif results.pass_rate >= 80:
        print("║" + " " * 68 + "║")
        print("║        ⚠️  MOSTLY PASSING - REVIEW FAILED TESTS                    ║")
        print("║" + " " * 68 + "║")
    else:
        print("║" + " " * 68 + "║")
        print("║        🚨 CRITICAL FAILURES - SECURITY ISSUES DETECTED 🚨          ║")
        print("║" + " " * 68 + "║")
    
    print("╚" + "═" * 68 + "╝")
    
    # Detailed failures if any
    if results.failed_tests > 0:
        print("\n📋 FAILED TEST DETAILS:")
        print("-" * 60)
        for test in results.tests:
            if not test.passed:
                print(f"  ❌ {test.name}")
                print(f"     Category: {test.category}")
                print(f"     Details: {test.details}")
                print()


def run_test_suite(skip_perf: bool = False, verbose: bool = False) -> TestSuiteResults:
    """Run the complete test suite."""
    
    results = TestSuiteResults()
    
    print("\n" + "=" * 70)
    print("🧪 PROTOCOL-VERIFY: COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Define all tests
    tests = [
        # ACCEPT tests: Honest/compliant training should pass
        ("Honest: Weight norms compliant", "honest", test_honest_weight_norms),
        ("Honest: Verification passes", "honest", test_honest_verification),
        ("Honest: Proof shows compliant", "honest", test_honest_proof_generation),
        ("Honest: Proof verifies TRUE", "honest", test_honest_proof_verification),
        
        # REJECT tests: Non-compliant training should be rejected
        ("REJECT: 10% over threshold", "reject", test_REJECT_10_percent_over_threshold),
        ("REJECT: 50% over threshold", "reject", test_REJECT_50_percent_over_threshold),
        ("REJECT: Explosive weights", "reject", test_REJECT_explosive_weights),
        ("REJECT: Proof shows non-compliant", "reject", test_REJECT_proof_non_compliant),
        ("REJECT: Verification FALSE", "reject", test_REJECT_verification_false),
        
        # TAMPER tests: Base model tampering should be detected
        ("Tamper: Clean model accepted", "tamper", test_tamper_clean_accepted),
        ("REJECT: Tampered base model", "tamper", test_REJECT_tampered_model),
        ("REJECT: Single-bit hash diff", "tamper", test_REJECT_single_bit_tamper),
        ("REJECT: Hidden attack (compliant weights)", "tamper", test_REJECT_hidden_attack),
    ]
    
    # Performance tests
    if not skip_perf:
        tests.extend([
            ("Perf: Verification <100ms", "perf", test_verification_speed),
            ("Perf: Proof gen <500ms", "perf", test_proof_generation_speed),
            ("Perf: Proof verify <100ms", "perf", test_proof_verification_speed),
        ])
    
    # Run all tests
    for name, category, test_func in tests:
        print(f"  Running: {name}...", end=" ", flush=True)
        result = run_test(name, category, test_func)
        results.add(result)
        
        status = "✅" if result.passed else "❌"
        print(f"{status} ({result.duration_ms:.2f}ms)")
        
        if verbose and result.details:
            print(f"    → {result.details}")
    
    results.end_time = datetime.now()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Protocol-Verify Master Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_suite.py              # Run all tests
  python test_suite.py --quick      # Skip performance tests
  python test_suite.py --verbose    # Show detailed output
  python test_suite.py -v --quick   # Verbose without perf tests
        """
    )
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Skip performance/stress tests"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed test output"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Export results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Run tests
    results = run_test_suite(skip_perf=args.quick, verbose=args.verbose)
    
    # Print scorecard
    print_scorecard(results, verbose=args.verbose)
    
    # Export JSON if requested
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
                }
                for t in results.tests
            ]
        }
        with open(args.json, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n📁 Results exported to: {args.json}")
    
    # Exit code
    sys.exit(0 if results.pass_rate == 100 else 1)


if __name__ == "__main__":
    main()
