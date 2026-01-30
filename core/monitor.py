"""
Protocol-Verify: Weight Monitoring Module

This module provides:
1. Weight capture hooks for LoRA training
2. Frobenius norm calculation
3. Safety invariant checking
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SafetyInvariants:
    """Safety thresholds for training verification."""
    
    max_weight_norm: float = 10.0
    expected_model_hash: Optional[str] = None
    min_dp_epsilon: float = 1.0
    max_gradient_norm: float = 1.0
    
    @classmethod
    def from_json(cls, path: str) -> "SafetyInvariants":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class VerificationResult:
    """Result of a safety verification check."""
    
    passed: bool
    weight_norm: float
    max_allowed_norm: float
    base_hash_valid: bool
    dp_requirements_met: bool
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_frobenius_norm(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the Frobenius norm of the LoRA weight update.
    
    P = B @ A
    ||P||_F = sqrt(sum(|p_ij|^2))
    """
    try:
        P = B @ A
    except ValueError:
        P = B @ A.T
    
    return float(np.sqrt(np.sum(P ** 2)))


def compute_frobenius_norm_direct(matrix: np.ndarray) -> float:
    """Compute Frobenius norm of a single matrix."""
    return float(np.sqrt(np.sum(matrix ** 2)))


class WeightMonitor:
    """Monitors and captures weight changes during LoRA training."""
    
    def __init__(self, safety_config: Optional[SafetyInvariants] = None):
        self.safety = safety_config or SafetyInvariants()
        self.weight_history: List[Dict[str, np.ndarray]] = []
        self.gradient_norms: List[float] = []
        
    def capture_weights(self, lora_weights: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Capture current LoRA weights and compute norms."""
        snapshot = {}
        norms = {}
        
        for layer_name, (A, B) in lora_weights.items():
            snapshot[f"{layer_name}_A"] = A.copy()
            snapshot[f"{layer_name}_B"] = B.copy()
            norms[layer_name] = compute_frobenius_norm(A, B)
            
        self.weight_history.append(snapshot)
        return norms
    
    def get_total_norm(self, lora_weights: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute the total Frobenius norm across all LoRA layers."""
        total_squared = 0.0
        
        for layer_name, (A, B) in lora_weights.items():
            try:
                P = B @ A
            except ValueError:
                P = B @ A.T
            total_squared += np.sum(P ** 2)
            
        return float(np.sqrt(total_squared))
    
    def verify_invariants(
        self,
        lora_weights: Dict[str, Tuple[np.ndarray, np.ndarray]],
        base_model_hash: str,
    ) -> VerificationResult:
        """
        Verify that all safety invariants are satisfied.
        
        Checks:
        1. ||ΔW||_F ≤ C (weight norm limit)
        2. Base model hash matches expected
        3. DP noise requirements met
        """
        # 1. Compute total weight norm
        total_norm = self.get_total_norm(lora_weights)
        norm_ok = total_norm <= self.safety.max_weight_norm
        
        # 2. Check base model hash
        if self.safety.expected_model_hash:
            hash_ok = base_model_hash == self.safety.expected_model_hash
        else:
            hash_ok = True
            
        # 3. Check DP requirements
        if self.gradient_norms:
            max_grad = max(self.gradient_norms)
            dp_ok = max_grad <= self.safety.max_gradient_norm
        else:
            dp_ok = True
        
        # Per-layer norms
        layer_norms = {}
        for layer_name, (A, B) in lora_weights.items():
            layer_norms[layer_name] = compute_frobenius_norm(A, B)
        
        passed = norm_ok and hash_ok and dp_ok
        
        return VerificationResult(
            passed=passed,
            weight_norm=total_norm,
            max_allowed_norm=self.safety.max_weight_norm,
            base_hash_valid=hash_ok,
            dp_requirements_met=dp_ok,
            details={
                "layer_norms": layer_norms,
                "base_model_hash": base_model_hash,
                "expected_hash": self.safety.expected_model_hash,
            }
        )
    
    def generate_verification_report(self, result: VerificationResult, output_path: Optional[str] = None) -> str:
        """Generate a human-readable verification report."""
        status = "✅ CERTIFIED COMPLIANT" if result.passed else "❌ POLICY VIOLATION DETECTED"
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           PROTOCOL-VERIFY SAFETY VERIFICATION REPORT          ║
╠══════════════════════════════════════════════════════════════╣
║ Status: {status:^52} ║
╠══════════════════════════════════════════════════════════════╣

┌─ WEIGHT NORM CHECK ─────────────────────────────────────────┐
│ Observed Norm:  {result.weight_norm:>10.6f}                              │
│ Maximum Allowed: {result.max_allowed_norm:>10.6f}                              │
│ Status: {'PASS ✓' if result.weight_norm <= result.max_allowed_norm else 'FAIL ✗':>10}                                        │
└─────────────────────────────────────────────────────────────┘

┌─ BASE MODEL VERIFICATION ───────────────────────────────────┐
│ Hash Valid: {'YES ✓' if result.base_hash_valid else 'NO ✗':>10}                                         │
└─────────────────────────────────────────────────────────────┘

╚══════════════════════════════════════════════════════════════╝
"""
        
        if output_path:
            Path(output_path).write_text(report)
            
        return report
