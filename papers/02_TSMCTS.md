# Twice Sequential Monte Carlo for Tree Search (TSMCTS)

**arXiv**: [2511.14220](https://arxiv.org/abs/2511.14220)  
**발표**: 2025년 11월 (v2: 2026년 2월)  
**저자**: Yaniv Oren, Joery A. de Vries, Pascal R. van der Vaart, Matthijs T. J. Spaan, Wendelin Böhmer  
**OpenReview**: [rs06uAQpuH](https://openreview.net/forum?id=rs06uAQpuH)

---

## TL;DR

MCTS보다 GPU 병렬화에 유리한 SMC(Sequential Monte Carlo)를 트리 서치에 적용하되, SMC의 고질적인 **높은 분산(variance)** 과 **path degeneracy** 문제를 MCTS + Sequential Halving 메커니즘으로 해결한다. SMC의 병렬성을 그대로 유지하면서 MCTS 수준의 탐색 품질을 달성.

---

## 배경 및 문제 의식

### SMC vs MCTS

| 특성 | MCTS | SMC |
|---|---|---|
| GPU 병렬화 | 어려움 (순차적 트리 구조) | 자연스러움 (particle 독립 처리) |
| 분산 | 낮음 | 높음 |
| Path degeneracy | 없음 | 심각 (깊이 증가 시) |
| 탐색 효율 | 높음 | 낮음 |

**SMC의 두 가지 핵심 문제:**

1. **높은 추정 분산**: 각 particle이 독립적인 trajectory를 따르므로, 같은 계산 예산 내에서 MCTS보다 가치 추정의 분산이 크다.
2. **Path degeneracy**: 시뮬레이션 깊이가 깊어질수록 모든 particle이 동일한 경로로 수렴하는 현상. 유효 particle 수가 급격히 감소한다.

---

## 방법론

TSMCTS는 두 단계의 "sequential" 구조로 구성된다 (이름의 "Twice"가 여기서 유래).

### 1단계: SMCTS (SMC Tree Search)

SMC의 관점을 "trajectory 추정"에서 **"root에서의 action value 추정"** 으로 전환한다.

- **Backpropagation 도입**: MCTS의 backprop 메커니즘을 SMC에 결합하여 root에서의 가치를 집계
- 이를 통해 깊이가 깊어져도 root의 value estimate가 안정적으로 유지됨
- path degeneracy 완화: backprop이 다양한 경로의 정보를 root로 전달하므로 단일 경로로의 수렴을 방지

```
기존 SMC: particles → 각자 trajectory 탐색 → 독립적 가치 추정
SMCTS:    particles → trajectory 탐색 → backprop → root의 action value 집계
```

### 2단계: Sequential Halving (SH) 통합

root에서의 탐색 자원 배분을 최적화하기 위해 **Sequential Halving**을 적용한다.

- SH는 처음에 모든 액션을 동등하게 탐색하다가, 라운드마다 절반의 액션을 제거
- 각 라운드에서 particle 수를 두 배로 늘려 남은 액션에 더 많은 자원 투입
- 구체적으로: SMCTS를 root에서 halving 수의 액션에 대해 순차적으로 호출, 동시에 particle 수는 두 배씩 증가

```
라운드 1: 모든 A개 액션, N개 particle
라운드 2: A/2개 액션, 2N개 particle  
라운드 3: A/4개 액션, 4N개 particle
...
```

### 전체 알고리즘 흐름

```
TSMCTS(root, budget):
  actions = all_actions
  particles = N
  while |actions| > 1:
    values = SMCTS(root, actions, particles)  ← 병렬 실행
    actions = top_half(actions, values)
    particles = 2 * particles
  return best_action(actions)
```

각 SMCTS 호출은 particle들이 독립적이므로 **완전한 GPU 병렬 실행** 가능.

---

## 핵심 실험 결과

### 정책 개선 연산자로서의 비교

TSMCTS는 다음을 모두 능가:
- SMC 기반 baseline
- 현대적인 MCTS 기반 policy improvement operator (Gumbel MuZero 포함)

### 스케일링 특성

| 계산 예산 증가 | SMC | MCTS | TSMCTS |
|---|---|---|---|
| Sequential compute 확장 | 분산↑, 성능 정체 | 선형적 향상 | **선형 이상의 향상** |
| Parallel compute 확장 | 자연스러움 | 어려움 | 자연스러움 |

### 분산 및 Path Degeneracy

- **분산 감소**: SMCTS의 backpropagation이 SMC 대비 추정 분산을 유의미하게 감소시킴
- **Path degeneracy 완화**: 깊은 탐색에서도 유효 particle 수 유지

---

## 의의 및 한계

**의의**
- SMC의 자연스러운 GPU 병렬성을 유지하면서 탐색 품질을 MCTS 수준으로 끌어올림
- Particle 단위 병렬 처리로 GPU 활용률을 최대화
- 이론적으로 깔끔한 프레임워크 — SMC + MCTS + SH의 통합

**한계**
- TransZero와 달리 MuZero 아키텍처 자체를 개선하지는 않음 (policy improvement operator에 초점)
- Sequential Halving의 라운드 수에 따라 최적 계산 예산 배분이 달라짐
- 실험적 검증 환경의 다양성 추가 필요

---

## 참고 링크

- [arXiv 논문](https://arxiv.org/abs/2511.14220)
- [OpenReview](https://openreview.net/forum?id=rs06uAQpuH)
- [arXiv HTML 버전](https://arxiv.org/html/2511.14220)
