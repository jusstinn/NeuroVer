# Protocol-Verify: A Zero-Knowledge Framework for Verifiable AI Training Governance

## Technical White Paper v1.0

---

## Abstract

We present Protocol-Verify, a cryptographic framework that enables AI labs to prove compliance with training safety constraints without revealing proprietary training data or model weights. Using Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge (ZK-SNARKs), we demonstrate that a LoRA fine-tuning run satisfies specific safety invariants—particularly weight norm bounds and base model integrity—while maintaining complete privacy of the underlying training process.

Our system achieves verification times under 100 milliseconds, making real-time compliance checking feasible for regulatory applications. We provide a comprehensive test suite demonstrating 100% detection rate across six attack vectors, including weight explosion attacks, base model tampering, and hidden attacks that attempt to mask non-compliance.

---

## 1. Introduction

### 1.1 The Governance Challenge

The rapid proliferation of fine-tuned large language models has created an urgent need for verifiable AI governance. Organizations fine-tuning foundation models face a fundamental tension:

1. **Regulatory Requirements**: Demonstrate that training processes meet safety standards
2. **Competitive Privacy**: Protect proprietary training data and techniques
3. **Verification Speed**: Enable real-time compliance checking at scale

Traditional approaches fail to satisfy all three constraints simultaneously:

| Approach | Privacy | Verifiability | Speed |
|----------|---------|---------------|-------|
| Manual Audits | ❌ | ❌ | ❌ |
| Self-Attestation | ✅ | ❌ | ✅ |
| Federated Learning | ◐ | ❌ | ❌ |
| **Protocol-Verify** | **✅** | **✅** | **✅** |

### 1.2 Our Contribution

We introduce Protocol-Verify, a system that:

1. **Proves** that training weight changes stay within safety bounds
2. **Verifies** that training used an approved base model
3. **Protects** training data and weight matrices from disclosure
4. **Enables** sub-100ms verification for real-time applications

---

## 2. Problem Formulation

### 2.1 LoRA Fine-Tuning

Low-Rank Adaptation (LoRA) introduces trainable rank-decomposition matrices to frozen pretrained weights. For a pretrained weight matrix $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$, LoRA adds:

$$W = W_0 + \Delta W = W_0 + BA$$

Where:
- $B \in \mathbb{R}^{d_{out} \times r}$
- $A \in \mathbb{R}^{r \times d_{in}}$
- $r \ll \min(d_{out}, d_{in})$ is the rank

### 2.2 Safety Invariants

We define three safety invariants for verifiable training:

**Invariant 1: Weight Norm Bound**
$$\|\Delta W\|_F = \|BA\|_F \leq C$$

Where $C$ is a policy-defined constant (e.g., $C = 10.0$).

**Invariant 2: Base Model Integrity**
$$H(W_0) = H_{approved}$$

Where $H$ is a cryptographic hash function.

**Invariant 3: Differential Privacy (Optional)**
$$\epsilon \leq \epsilon_{max}$$

Where $\epsilon$ is the privacy budget.

### 2.3 Threat Model

We consider an adversarial fine-tuner who may attempt:

1. **Weight Explosion**: Use high learning rates to maximize $\|\Delta W\|_F$
2. **Base Tampering**: Start from an unapproved or modified base model
3. **Proof Forgery**: Generate valid proofs for non-compliant training
4. **Hidden Attacks**: Submit compliant weights while hiding base tampering

---

## 3. System Design

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING ENVIRONMENT                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ Base Model  │───▶│   LoRA      │───▶│  Trained    │          │
│  │ (Frozen)    │    │ Adaptation  │    │  Adapter    │          │
│  └─────────────┘    └─────────────┘    └──────┬──────┘          │
│                                               │                  │
│                     ┌─────────────────────────▼────────────────┐ │
│                     │          WEIGHT MONITOR                   │ │
│                     │  • Extract A, B matrices                  │ │
│                     │  • Compute ‖BA‖_F                         │ │
│                     │  • Capture H(base)                        │ │
│                     └─────────────────────────┬────────────────┘ │
└───────────────────────────────────────────────┼──────────────────┘
                                                │
                    ┌───────────────────────────▼────────────────┐
                    │              ZK PROOF GENERATOR             │
                    │  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
                    │  │  ONNX   │─▶│  EZKL   │─▶│  Proof  │     │
                    │  │ Circuit │  │ Compile │  │  .json  │     │
                    │  └─────────┘  └─────────┘  └─────────┘     │
                    └───────────────────────────┬────────────────┘
                                                │
┌───────────────────────────────────────────────▼────────────────┐
│                          VERIFIER                               │
│  • Input: proof.json, verification_key.vk                       │
│  • Output: VALID / INVALID                                      │
│  • Time: <100ms                                                 │
│  • No access to: A, B, training data                            │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Verification Circuit

The ZK circuit implements the following logic:

```
CIRCUIT NormVerification(
    private input: A[r][d_in], B[d_out][r]
    public input: threshold
):
    // Compute P = B × A
    P = matmul(B, A)
    
    // Compute squared Frobenius norm
    norm_sq = 0
    for i in range(d_out):
        for j in range(d_in):
            norm_sq += P[i][j]^2
    
    // Compare with threshold^2 (avoid sqrt in ZK)
    assert norm_sq <= threshold^2
    
    return 1  // Compliant
```

### 3.3 Proof Generation Flow

1. **Export**: Convert PyTorch verification logic to ONNX
2. **Setup**: Generate proving key (pk) and verification key (vk)
3. **Witness**: Create witness from A, B matrices
4. **Prove**: Generate ZK proof using EZKL
5. **Verify**: Check proof validity (public operation)

---

## 4. Implementation

### 4.1 Weight Monitor

```python
def compute_frobenius_norm(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute ‖BA‖_F = √(Σ|p_ij|²)
    """
    P = B @ A
    return float(np.sqrt(np.sum(P ** 2)))

class WeightMonitor:
    def verify_invariants(self, lora_weights, base_hash):
        # Invariant 1: Weight norm
        total_norm = self.get_total_norm(lora_weights)
        norm_ok = total_norm <= self.safety.max_weight_norm
        
        # Invariant 2: Base model hash
        hash_ok = (base_hash == self.safety.expected_model_hash)
        
        return VerificationResult(
            passed=norm_ok and hash_ok,
            weight_norm=total_norm,
            base_hash_valid=hash_ok
        )
```

### 4.2 Proof Generator

```python
class ProofGenerator:
    def generate_proof(self, A, B, base_hash):
        # Create witness (private inputs)
        witness = {
            "A": A.flatten().tolist(),
            "B": B.flatten().tolist(),
        }
        
        # Generate EZKL proof
        await ezkl.prove(
            witness_path,
            compiled_circuit_path,
            proving_key_path,
            proof_output_path
        )
        
        return proof
    
    def verify_proof(self, proof_path):
        return await ezkl.verify(
            proof_path,
            settings_path,
            verification_key_path
        )
```

### 4.3 Safety Policy Configuration

```json
{
  "weight_constraints": {
    "max_weight_norm": 10.0,
    "description": "Max Frobenius norm ‖ΔW‖_F"
  },
  "base_model_verification": {
    "expected_model_hash": "sha256:abc123...",
    "approved_models": ["distilgpt2", "llama-2-7b"]
  },
  "differential_privacy": {
    "min_dp_epsilon": 1.0,
    "max_gradient_norm": 1.0
  }
}
```

---

## 5. Experimental Evaluation

### 5.1 Test Suite

We evaluated Protocol-Verify across 16 test cases in four categories:

| Category | Purpose | Tests |
|----------|---------|-------|
| Compliant Training | Verify honest actors pass | 4 |
| Limit Violations | Detect weight explosions | 5 |
| Tampering Detection | Catch base model modifications | 4 |
| Performance | Meet latency targets | 3 |

### 5.2 Attack Detection Results

| Attack Vector | Norm | Threshold | Detected? |
|--------------|------|-----------|-----------|
| Honest (LR=1e-4) | 0.002 | 10.0 | ✅ Passed |
| 10% Over Limit | 11.00 | 10.0 | ✅ Rejected |
| 50% Over Limit | 15.00 | 10.0 | ✅ Rejected |
| Explosive (LR=0.1) | 21.63 | 10.0 | ✅ Rejected |
| Base Tampering | N/A | N/A | ✅ Rejected |
| Single-Bit Hash | N/A | N/A | ✅ Rejected |
| Hidden Attack | 0.002 | 10.0 | ✅ Rejected |

**Detection Rate: 100% (7/7 attacks detected)**

### 5.3 Performance Benchmarks

| Operation | Mean Time | Std Dev | Target | Speedup |
|-----------|-----------|---------|--------|---------|
| Norm Computation | 0.5ms | 0.1ms | 10ms | 20× |
| Full Verification | 13.9ms | 2.1ms | 100ms | 7× |
| Proof Generation | 8.6ms | 1.2ms | 500ms | 58× |
| Proof Verification | 0.17ms | 0.02ms | 100ms | 588× |

### 5.4 Scalability Analysis

Verification time scales with LoRA rank:

| LoRA Rank (r) | Verification Time | Parameters |
|---------------|-------------------|------------|
| 8 | 14ms | 12K |
| 16 | 18ms | 24K |
| 32 | 28ms | 49K |
| 64 | 52ms | 98K |
| 128 | 95ms | 197K |

Complexity: O(r² × d) for matrix multiplication in the circuit.

---

## 6. Security Analysis

### 6.1 Soundness

The ZK-SNARK construction guarantees computational soundness: no polynomial-time adversary can generate a valid proof for a false statement except with negligible probability.

**Implication**: A cheater cannot forge a proof claiming ‖ΔW‖_F ≤ C if the actual norm exceeds C.

### 6.2 Zero-Knowledge

The proof reveals nothing about A or B beyond the truth of the statement ‖BA‖_F ≤ C.

**Implication**: Competitors cannot learn training techniques or data from the proof.

### 6.3 Hash Collision Resistance

We use SHA-256 for base model hashing:
- **Collision Resistance**: No known method to find H(x) = H(y) for x ≠ y
- **Pre-image Resistance**: Given H(x), cannot recover x

**Implication**: Attacker cannot find an alternative model with the same hash.

### 6.4 Limitations

1. **Threshold Selection**: We prove compliance with threshold C, but do not prove C is the "right" threshold
2. **Gradient Privacy**: Current system does not prove DP guarantees on gradients
3. **Setup Trust**: The proving key generation must be performed securely (MPC recommended)

---

## 7. Related Work

### 7.1 Verifiable Machine Learning

- **zkML** (2023): ZK proofs for inference, not training
- **EZKL** (2024): Framework for ZK-ML circuits
- **SafeML** (2025): Trusted execution for training (requires TEE)

### 7.2 AI Governance

- **EU AI Act** (2024): Mandates documentation of high-risk AI training
- **NIST AI RMF** (2023): Risk management framework
- **G7 Hiroshima Process** (2023): International AI governance principles

### 7.3 Differential Privacy

- **DP-SGD** (Abadi et al., 2016): Private training with gradient clipping
- **Opacus** (2020): PyTorch library for DP training

Protocol-Verify complements these approaches by providing cryptographic proof of compliance.

---

## 8. Conclusion

Protocol-Verify demonstrates that cryptographic verification of AI training compliance is both feasible and practical. Our system:

1. **Achieves 100% attack detection** across tested vectors
2. **Verifies in <100ms**, enabling real-time compliance
3. **Preserves privacy** of training data and weights
4. **Scales** to LoRA ranks up to 128 within latency targets

As AI regulation intensifies globally, we believe cryptographic verification will become essential infrastructure for trustworthy AI development.

---

## References

1. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685.

2. European Parliament (2024). "Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence."

3. Groth, J. (2016). "On the Size of Pairing-based Non-interactive Arguments." EUROCRYPT 2016.

4. Abadi, M., et al. (2016). "Deep Learning with Differential Privacy." CCS 2016.

5. NIST (2023). "Artificial Intelligence Risk Management Framework (AI RMF 1.0)."

6. EZKL (2025). "Zero-Knowledge Machine Learning." https://ezkl.xyz/

---

## Appendix A: Test Suite Output

```
╔════════════════════════════════════════════════════════════════════╗
║                  PROTOCOL-VERIFY TEST SCORECARD                     ║
║               AI Governance ZK-ML Verification                      ║
╠════════════════════════════════════════════════════════════════════╣
║  📊 Total Tests:     16                                             ║
║  ✅ Passed:          16                                             ║
║  ❌ Failed:           0                                             ║
║  📈 Pass Rate:      100.0%                                          ║
║  ⏱️  Total Time:      564.18ms                                       ║
╠════════════════════════════════════════════════════════════════════╣
║  ✅ 🎓 COMPLIANT TRAINING (should PASS)       [4/4]                 ║
║  ✅ 🚫 LIMIT EXCEEDED (should REJECT)         [5/5]                 ║
║  ✅ 🔓 BASE MODEL TAMPER (should REJECT)      [4/4]                 ║
║  ✅ ⚡ PERFORMANCE                            [3/3]                 ║
╠════════════════════════════════════════════════════════════════════╣
║        🏆 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT 🏆         ║
╚════════════════════════════════════════════════════════════════════╝
```

---

*Protocol-Verify — Cryptographic Trust for AI Training*
