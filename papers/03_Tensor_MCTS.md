# Tensor Implementation of Monte-Carlo Tree Search for Model-Based Reinforcement Learning

**저널**: MDPI Applied Sciences, Vol. 13, No. 3 (2023)  
**논문 링크**: [mdpi.com/2076-3417/13/3/1406](https://www.mdpi.com/2076-3417/13/3/1406)  
**구현**: PyTorch + Ray

---

## TL;DR

MCTS의 모든 연산을 텐서(tensor) 연산으로 재구현하여 단일 GPU에서 수십~수백 개의 observation을 **완전 병렬(fully parallel)**로 처리한다. 기존 CPU/CPU-GPU 혼합 구현 대비 MuZero 환경에서 일관된 성능 향상을 달성.

---

## 배경 및 문제 의식

기존 MCTS 구현의 병렬화 장벽은 자료구조에 있다. 전통적인 MCTS는 트리를 **포인터 기반 노드 객체**로 구성하며, 이는 GPU 텐서 연산과 근본적으로 호환되지 않는다.

- GPU는 행렬/텐서 연산에 최적화되어 있으며, 포인터 추적 및 조건 분기에는 비효율적
- 기존 모델 기반 RL 구현(MuZero 등)은 MCTS를 CPU에서 실행하고, 신경망 추론만 GPU를 활용
- 이로 인해 CPU-GPU 간 데이터 전송 오버헤드가 발생하고, MCTS 자체는 GPU 가속의 혜택을 받지 못함

---

## 방법론

### 텐서 기반 자료구조

트리 전체를 텐서로 표현한다. 영감의 출처: Q-table의 구조.

각 트리의 상태는 다음 텐서들로 구성된다:

| 텐서 | 내용 |
|---|---|
| `values` | 각 노드의 가치 추정값 |
| `visit_counts` | 각 노드의 방문 횟수 N(s,a) |
| `prior_probs` | 정책 네트워크의 사전 확률 P(s,a) |
| `rewards` | 각 전이에서의 보상 |
| `hidden_states` | 각 노드의 latent state |
| `parent_ids` | 부모 노드 인덱스 |

핵심: 이 텐서들의 첫 번째 차원이 **배치(batch) 차원** — 동시에 처리하는 observation 수

### 병렬 처리 흐름

MCTS의 4단계 (Selection → Expansion → Evaluation → Backpropagation)를 모두 배치 텐서 연산으로 처리:

```
배치 크기 B개의 observation에 대해:

Selection:   [B, depth] 텐서 → argmax(UCB) 동시 계산
Expansion:   [B, num_actions] 텐서 → 모든 트리의 리프 동시 확장
Evaluation:  [B, hidden_dim] 텐서 → 신경망 배치 추론 (단 1회 forward pass)
Backprop:    [B, depth] 텐서 → 경로 따라 동시 업데이트
```

신경망 추론이 배치 단위로 단 한 번에 수행되어 **GPU 활용률 최대화**.

### Ray를 통한 분산 처리

- Python의 Ray 라이브러리로 프로세스 관리
- 각 프로세스가 단일 GPU를 사용
- 여러 프로세스 간 결과 집계

### 구현 환경

- **프레임워크**: PyTorch
- **GPU**: NVIDIA GeForce RTX 2080 Ti
- **CPU**: 16-core Intel Xeon E5-2643 @ 3.30GHz
- **평가**: 100번 반복의 평균 및 표준편차

---

## 핵심 실험 결과

### 비교 대상

기존 MuZero 관련 MCTS 구현 3가지와 비교:
- CPU 기반 구현
- CPU-GPU 혼합 구현 (신경망만 GPU)
- 기존 배치 MCTS 구현

### 결과 요약

배치 크기 **50~750개 observation** 범위에서 실험:

| 배치 크기 | 기존 구현 대비 속도 |
|---|---|
| 50 | 유의미한 향상 |
| 200 | 더 큰 향상 |
| 750 | 최대 향상 (배치 클수록 효율적) |

- **Non-DNN 설정**: GPU 텐서 구현이 CPU 구현을 일관되게 능가
- **DNN 설정 (실제 MuZero)**: GPU 원자 연산의 특성에도 불구하고 CPU 및 CPU-GPU 혼합 구현을 능가
- 배치 크기가 커질수록 스케일 효율성이 증가 (GPU 활용률 향상)

---

## 의의 및 한계

**의의**
- MCTS 자료구조 자체를 GPU 친화적으로 재설계한 선구적 접근
- 추가 알고리즘 변경 없이 기존 MuZero에 적용 가능
- 대규모 환경 병렬 학습(vectorized env)과 자연스럽게 결합 가능

**한계**
- 트리 연산의 "원자성(atomicity)" 문제: 동시 업데이트 시 충돌 가능성 (atomic GPU operations로 처리)
- 트리 깊이가 깊어질수록 메모리 사용량이 `O(B × depth × branching_factor)`로 증가
- 단일 observation에 대한 단일 MCTS 실행은 기존 대비 오히려 비효율적

---

## 참고 링크

- [MDPI 논문](https://www.mdpi.com/2076-3417/13/3/1406)
- [ResearchGate PDF](https://www.researchgate.net/publication/367346071_Tensor_Implementation_of_Monte-Carlo_Tree_Search_for_Model-Based_Reinforcement_Learning)
