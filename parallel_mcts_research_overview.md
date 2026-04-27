# MuZero MCTS 병렬화 최신 연구 Overview

> 작성일: 2026-04-28

---

## 배경

MuZero의 MCTS는 근본적으로 순차적(sequential) 구조를 가지고 있어 병렬화가 어렵다. 주요 병목은 두 가지다.

1. **Recurrent dynamics model**: 트리를 한 노드씩 순차적으로 확장
2. **UCB 기반 선택**: 방문 카운트(visitation count)에 의존하여 병렬 업데이트가 충돌함

최근 연구들은 아키텍처 교체, 알고리즘 재설계, 텐서 연산화, 프레임워크 최적화 등 다양한 방향으로 이 문제를 해결하고 있다.

---

## 핵심 논문

### 1. TransZero (2025년 9월)

**"TransZero: Parallel Tree Expansion in MuZero using Transformer Networks"**  
arXiv: [2509.11233](https://arxiv.org/abs/2509.11233) | GitHub: [emalmsten/TransZero](https://github.com/emalmsten/TransZero)  
저자: Emil Malmsten, Wendelin Böhmer (TU Delft)

#### 핵심 아이디어

- MuZero의 RNN 기반 dynamics model → **Transformer 기반 네트워크**로 교체
- Transformer는 여러 미래 상태를 **동시에(in parallel)** 생성할 수 있음
- UCB의 순차적 방문 카운트 의존성을 제거하기 위해 **Mean-Variance Constrained (MVC) evaluator** 도입
- 위 두 변경으로 서브트리 전체를 병렬 확장(parallel subtree expansion) 가능

#### 결과

- MiniGrid, LunarLander 실험에서 MuZero 대비 **최대 11배 wall-clock 속도 향상**
- Sample efficiency는 MuZero와 동등하게 유지

→ [📄 상세 정리](papers/01_TransZero.md)

---

### 2. TSMCTS – Twice Sequential Monte Carlo (2025년 11월)

**"Twice Sequential Monte Carlo for Tree Search"**  
arXiv: [2511.14220](https://arxiv.org/abs/2511.14220)  
저자: Yaniv Oren, Joery A. de Vries, Pascal R. van der Vaart, Matthijs T. J. Spaan, Wendelin Böhmer

#### 핵심 아이디어

- SMC(Sequential Monte Carlo)는 MCTS보다 병렬화·GPU 가속에 유리하지만 **높은 분산(variance)** 과 **path degeneracy** 문제가 있음
- TSMCTS는 MCTS + Sequential Halving의 메커니즘을 SMC에 결합하여 이 문제를 해결
- SMC의 자연스러운 병렬성을 살리면서 MCTS 수준의 탐색 품질 달성

#### 결과

- SMC baseline 및 modern MCTS 기반 policy improvement operator를 능가
- Sequential compute 확장에서 유리한 스케일링
- 분산 감소 및 path degeneracy 완화

→ [📄 상세 정리](papers/02_TSMCTS.md)

---

### 3. Tensor-based MCTS (2023)

**"Tensor Implementation of Monte-Carlo Tree Search for Model-Based Reinforcement Learning"**  
[MDPI Applied Sciences](https://www.mdpi.com/2076-3417/13/3/1406)

#### 핵심 아이디어

- MCTS 전체 연산을 **텐서 연산(tensor operations)** 으로 재구현
- 단일 GPU에서 **50~750개 observation을 완전 병렬(fully parallel)** 처리
- MuZero 알고리즘에 적용하여 배치 크기 확장에 따른 효율 검증

→ [📄 상세 정리](papers/03_Tensor_MCTS.md)

---

### 4. LightZero (NeurIPS 2023 Spotlight + 2024 확장)

**"LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios"**  
GitHub: [opendilab/LightZero](https://github.com/opendilab/LightZero)

#### 핵심 특징

- MuZero 계열 9개 알고리즘을 위한 통합 벤치마크 프레임워크
- **혼합 이종 컴퓨팅(mixed heterogeneous computing)** 으로 MCTS 병목 구간 최적화
- 20개 이상의 환경에서 표준화된 평가 제공
- 2024년 확장: **ReZero** ([arXiv 2404.16364](https://arxiv.org/html/2404.16364v1)) — just-in-time reanalyze로 학습 속도 향상

→ [📄 상세 정리](papers/04_LightZero.md)

---

### 5. Google DeepMind mctx (JAX 라이브러리)

GitHub: [google-deepmind/mctx](https://github.com/google-deepmind/mctx)

#### 핵심 특징

- JAX-native MCTS 구현 (AlphaZero, MuZero, Gumbel MuZero 포함)
- **JIT 컴파일 + 벡터화(vmap)** 로 대규모 병렬 배치 추론 지원
- DeepMind 내부에서 사용하는 프로덕션 수준의 구현

→ [📄 상세 정리](papers/05_mctx.md)

---

### 6. Understanding Methods for Scalable MCTS (ICLR 2025 Blog Track)

[ICLR 2025 블로그](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-scalable-mcts-104/blog/scalable-mcts/)

병렬 MCTS 기법 전반을 정리한 survey. 아래 기법들의 원리와 트레이드오프를 체계적으로 다룸.

| 기법 | 설명 |
|---|---|
| Leaf Parallelism | 각 leaf에서의 rollout을 병렬 실행 |
| Root Parallelism | 여러 워커가 독립적으로 트리를 구성 후 결과 병합 |
| Tree Parallelism | 공유 트리에서 여러 스레드가 동시 탐색 (AlphaGo 방식) |
| Virtual Loss | 탐색 중인 노드에 임시 페널티를 줘 워커들이 다른 경로 탐색 |
| Transposition-Driven Scheduling | 트랜스포지션 테이블 기반 분산 스케줄링 |
| Distributed Depth-First Scheduling | 깊이 우선 탐색 기반 분산 처리 |

→ [📄 상세 정리](papers/06_Scalable_MCTS_Survey.md)

---

## 요약 비교표

| 접근 방식 | 대표 연구 | 핵심 아이디어 | 속도 향상 |
|---|---|---|---|
| 아키텍처 교체 | TransZero | RNN → Transformer로 병렬 트리 확장 | 최대 11x |
| 알고리즘 교체 | TSMCTS | MCTS → SMC 기반 GPU 친화적 구조 | 확장성 우수 |
| 텐서 구현 | Tensor MCTS | 모든 연산을 배치 텐서 연산으로 재구현 | 배치 스케일 |
| 프레임워크 | LightZero, mctx | 벡터화·JIT 컴파일로 throughput 향상 | 환경 의존적 |

---

## 추천 시작점

- **이론적 이해**: ICLR 2025 Scalable MCTS 블로그 포스트
- **빠른 구현 참고**: DeepMind `mctx` (JAX, JIT 지원)
- **최신 알고리즘 연구**: TransZero (병렬 트리 확장), TSMCTS (SMC 기반)
- **종합 벤치마크**: LightZero (MuZero 계열 전체 비교)

---

## 참고 링크

- [TransZero arXiv](https://arxiv.org/abs/2509.11233)
- [TransZero GitHub](https://github.com/emalmsten/TransZero)
- [TSMCTS arXiv](https://arxiv.org/abs/2511.14220)
- [Tensor MCTS (MDPI)](https://www.mdpi.com/2076-3417/13/3/1406)
- [LightZero GitHub](https://github.com/opendilab/LightZero)
- [ReZero arXiv](https://arxiv.org/html/2404.16364v1)
- [Google DeepMind mctx](https://github.com/google-deepmind/mctx)
- [Scalable MCTS – ICLR 2025](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-scalable-mcts-104/blog/scalable-mcts/)
