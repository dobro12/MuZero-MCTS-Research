# Understanding Methods for Scalable MCTS

**발표**: ICLR 2025 Blog Track  
**링크**: [ICLR 2025 블로그](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-scalable-mcts-104/blog/scalable-mcts/)  
**발표자**: Will Knipe  
**형태**: Survey / 튜토리얼 블로그 포스트

---

## TL;DR

병렬 MCTS의 모든 주요 기법(Leaf / Root / Tree 병렬화, Virtual Loss, Transposition-Driven Scheduling, 분산 깊이 우선 탐색)을 체계적으로 정리한 ICLR 2025 survey. 각 기법의 원리, 장단점, 실전 트레이드오프를 비교하며, "단일 최선 방법은 없다"는 결론을 도출한다.

---

## 배경

MCTS는 복잡한 환경에서 강력하지만, 실시간 의사결정을 위해서는 빠른 실행이 필수다. MCTS를 병렬화하는 방법은 크게 세 가지 클래식 방법과 여러 고급 분산 방법으로 나뉜다.

**근본적인 어려움**: MCTS는 Selection → Expansion → Evaluation → Backpropagation의 **순차적 루프**를 반복하는 알고리즘이다. 각 시뮬레이션 결과가 다음 시뮬레이션의 탐색 방향을 결정하기 때문에, 병렬화 시 탐색 품질과 속도 간의 trade-off가 발생한다.

---

## 방법론: 주요 병렬화 기법

### 1. Leaf Parallelism (리프 병렬화)

**개념**: Selection + Expansion은 단일 스레드로 수행하고, 선택된 리프 노드에서의 **simulation(rollout)을 여러 워커가 병렬 실행**.

```
Single thread: Selection → Expansion → [선택된 리프 노드]
                                              ↓
Worker 1: Simulation 1 ──────────────┐
Worker 2: Simulation 2 ──────────────┤ → 결과 집계 → Backpropagation
Worker N: Simulation N ──────────────┘
```

**장점**: 구현 단순, 동기화 오버헤드 최소  
**단점**: Selection/Expansion의 순차 병목 해소 불가, 신경망 기반 MuZero에서는 rollout이 없어 효과 제한적

---

### 2. Root Parallelism (루트 병렬화)

**개념**: 각 워커가 **동일한 루트 상태에서 독립적인 트리를 구축**하고, 최종 결과를 집계.

```
Root state
    ├── Worker 1: 독립 트리 구축 → 통계
    ├── Worker 2: 독립 트리 구축 → 통계
    └── Worker N: 독립 트리 구축 → 통계
                                      ↓
                               통계 병합 → 최종 액션 선택
```

**장점**: 워커 간 동기화 불필요, 구현 단순  
**단점**: 각 워커가 독립 탐색하므로 중복 탐색 발생, 정보 공유 없어 비효율적  
**활용**: 탐색 다양성이 중요한 상황 (앙상블 효과)

---

### 3. Tree Parallelism (트리 병렬화)

**개념**: **공유된 단일 트리**에서 여러 스레드가 동시에 탐색. AlphaGo 계열이 채택한 방식.

```
Shared Tree
    ├── Thread 1: Select → Expand → Evaluate (GPU) → Backprop
    ├── Thread 2: Select → Expand → Evaluate (GPU) → Backprop
    └── Thread N: Select → Expand → Evaluate (GPU) → Backprop
```

**필요 기술**:
- **Local Mutex**: 동일 노드에 대한 동시 업데이트 방지
- **Virtual Loss**: 탐색 중인 노드에 임시 패널티 부여

**장점**: 정보 공유로 탐색 효율 높음, AlphaGo/AlphaZero에서 검증  
**단점**: Lock 경합, 동기화 오버헤드

---

### 4. Virtual Loss (핵심 기법)

트리 병렬화의 핵심 기술. 한 워커가 특정 경로를 탐색 중일 때, **해당 노드에 임시 부정적 가치(virtual loss)를 부여**하여 다른 워커가 다른 경로를 탐색하도록 유도.

```python
# 탐색 시작: virtual loss 부여
N(s,a) += 1         # 방문 횟수 임시 증가
W(s,a) -= lambda    # 가치 임시 감소 (virtual loss)

# 탐색 완료: 실제 값으로 복원
W(s,a) += lambda + actual_value  # virtual loss 제거 + 실제 보상 추가
```

**효과**: 여러 워커가 서로 다른 경로를 자동으로 탐색 → 탐색 다양성 확보  
**분석**: Virtual loss는 탐색의 분산을 증가시키는 효과가 있어, 최적값과 약간의 차이를 만들 수 있음 (품질 trade-off)

---

### 5. Transposition-Driven Scheduling (TDS)

**개념**: 트랜스포지션 테이블(동일 상태의 다른 경로 통한 도달)을 활용한 분산 스케줄링.

- 각 노드의 "담당 워커"를 해시 함수로 결정
- 노드 업데이트 요청을 해당 워커에게 비동기 전송
- Lock-free 분산 처리 가능

**장점**: 중복 계산 제거, 대규모 분산 환경에서 효율적  
**단점**: 통신 오버헤드, 구현 복잡성

---

### 6. Distributed Depth-First Scheduling (DDFS)

**개념**: 깊이 우선 탐색(DFS)의 특성을 활용한 분산 처리.

- DFS는 메모리 지역성이 높아 캐시 효율적
- 여러 DFS 트레이스를 분산 환경에서 병렬 실행
- WU-UCT: 불완전 업데이트(incomplete update)로 동기화 지연 허용

---

## 기법 비교 요약

| 기법 | 동기화 | 정보 공유 | 구현 복잡도 | 최적 환경 |
|---|---|---|---|---|
| Leaf Parallelism | 낮음 | 없음 | 낮음 | 깊은 rollout |
| Root Parallelism | 없음 | 없음 | 낮음 | 앙상블 탐색 |
| Tree Parallelism | 높음 | 완전 | 중간 | 단일 머신 멀티코어 |
| Virtual Loss | 낮음 | 완전 | 낮음 | 트리 병렬화와 함께 |
| TDS | 비동기 | 완전 | 높음 | 대규모 분산 |
| DDFS | 낮음 | 부분 | 높음 | 대규모 분산 |

---

## 핵심 결론

> **"단일 최선 방법은 없다."**

병렬 MCTS 기법의 선택은 다음 요소에 따라 달라진다:

1. **시스템 아키텍처**: 단일 머신 멀티코어 vs 분산 클러스터
2. **통신 비용**: 로컬 메모리 공유 vs 네트워크 통신
3. **탐색 품질 vs 속도**: Virtual loss의 품질 trade-off 허용 여부
4. **알고리즘 특성**: rollout 기반 vs 신경망 평가 기반 (MuZero 등)

MuZero처럼 신경망 평가가 필요한 경우, **GPU 배치 추론을 위한 워커 동기화**가 추가적인 설계 고려사항이 된다.

---

## 참고 링크

- [ICLR 2025 블로그 포스트](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-scalable-mcts-104/blog/scalable-mcts/)
- [ICLR 2025 포스터](https://iclr.cc/virtual/2025/poster/31350)
- [발표 영상 (SlidesLive)](https://slideslive.com/39033902/understanding-methods-for-scalable-mcts)
- [Virtual Loss 분석 논문](https://liacs.leidenuniv.nl/~plaata1/papers/paper_ICAART17.pdf)
