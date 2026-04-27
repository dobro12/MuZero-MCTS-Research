# TransZero: Parallel Tree Expansion in MuZero using Transformer Networks

**arXiv**: [2509.11233](https://arxiv.org/abs/2509.11233)  
**발표**: 2025년 9월 (TU Delft)  
**저자**: Emil Malmsten, Wendelin Böhmer  
**코드**: [github.com/emalmsten/TransZero](https://github.com/emalmsten/TransZero)

---

## TL;DR

MuZero의 RNN 기반 dynamics model을 Transformer로 교체하고, UCB의 순차적 방문 카운트 의존성을 제거하는 MVC evaluator를 도입해 MCTS 트리 전체를 병렬로 확장한다. MuZero 대비 **최대 11배 wall-clock 속도 향상**, sample efficiency는 동등하게 유지.

---

## 배경 및 문제 의식

MuZero는 두 가지 이유로 병렬화가 근본적으로 어렵다.

1. **RNN dynamics model**: 트리 노드를 하나씩 순차적으로 확장한다. `h_{t+1} = f(h_t, a_t)` 형태로 이전 상태에 의존하기 때문에 여러 노드를 동시에 생성할 수 없다.

2. **UCB 선택 기준의 방문 카운트 의존성**: `UCB(s, a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))` 에서 `N(s,a)` (방문 횟수)는 순차적으로 업데이트되어야 하므로 병렬 worker들 간 충돌이 발생한다.

---

## 방법론

### 1. Transformer Dynamics Network

MuZero의 recurrent dynamics model `g_θ(h_t, a_t) → (r_t, h_{t+1})`을 **Transformer 기반 네트워크**로 교체한다.

- Transformer는 self-attention을 통해 **여러 액션 시퀀스를 동시에 처리** 가능
- 하나의 forward pass에서 서브트리 전체의 미래 상태 `h_{t+1}, h_{t+2}, ..., h_{t+k}`를 병렬 생성
- 입력: 현재 latent state + 액션 시퀀스 배치
- 출력: 해당 시퀀스 각각에 대한 latent state 배치

이를 통해 기존의 순차적 노드 확장을 **서브트리 단위 병렬 확장**으로 전환한다.

### 2. Mean-Variance Constrained (MVC) Evaluator

UCB의 방문 카운트 `N(s,a)` 의존성을 제거하기 위해 **MVC evaluator**를 도입한다.

- 방문 횟수 대신 **추정값의 평균과 분산**을 기반으로 노드를 평가
- 여러 worker가 동시에 노드를 평가하더라도 충돌 없이 독립적으로 계산 가능
- 탐색(exploration)과 활용(exploitation) 균형을 분산 정보로 제어

### 3. Parallel Subtree Expansion

위 두 컴포넌트를 결합하면 다음이 가능해진다.

```
기존 MuZero: root → node1 → node2 → node3 → ... (순차)
TransZero:   root → [node1, node2, node3, ...] (병렬, 서브트리 단위)
```

전체 서브트리를 하나의 Transformer forward pass로 처리하여 MCTS 시뮬레이션 횟수를 대폭 줄인다.

---

## 핵심 실험 결과

### 환경: MiniGrid, LunarLander

| 지표 | MuZero | TransZero | 개선 |
|---|---|---|---|
| Wall-clock 속도 (실험) | 1x | **최대 11x** | ~11배 향상 |
| Wall-clock 속도 (이론) | 1x | **최대 560x** | 1640 sim 기준 |
| Sample efficiency | 기준 | 동등 | 유지 |

### Ablation Study

두 컴포넌트 모두 필수적임을 확인:
- Transformer만 도입 → 일부 속도 향상, 하지만 MVC 없이는 탐색 품질 저하
- MVC만 도입 → RNN 병목으로 인해 병렬화 효과 미미
- 두 가지 모두 → 최대 성능 달성

### 이론적 분석

시뮬레이션 수 `S`에 대해 이론적 속도 향상은 `O(S)`에 근접. 1640번의 MuZero 시뮬레이션이 필요한 시나리오에서 최대 **560x 속도 향상** 가능성 제시.

---

## 의의 및 한계

**의의**
- MCTS 병렬화의 근본 병목(RNN + UCB)을 동시에 해결한 최초 시도
- 아키텍처 교체만으로 sample efficiency 손실 없이 속도 향상 달성
- 코드 공개로 재현 가능

**한계**
- Transformer는 RNN보다 메모리 사용량이 많아 트리가 깊어질수록 메모리 비용 증가
- 실험 환경이 상대적으로 단순함 (MiniGrid, LunarLander) — 복잡한 환경에서의 검증 필요
- Transformer의 긴 시퀀스 처리 시 어텐션 연산 비용 고려 필요

---

## 참고 링크

- [arXiv 논문](https://arxiv.org/abs/2509.11233)
- [GitHub 코드](https://github.com/emalmsten/TransZero)
- [OpenReview](https://openreview.net/forum?id=2GMMMgHRHW)
- [TU Delft Repository](https://repository.tudelft.nl/record/uuid:00d171fe-328e-4c78-a981-050e08c2ba08)
