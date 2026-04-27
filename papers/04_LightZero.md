# LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios

**발표**: NeurIPS 2023 Spotlight (Datasets and Benchmarks Track)  
**논문**: [OpenReview](https://openreview.net/forum?id=oIUXpBnyjv) | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/765043fe026f7d704c96cec027f13843-Paper-Datasets_and_Benchmarks.pdf)  
**GitHub**: [opendilab/LightZero](https://github.com/opendilab/LightZero)  
**기관**: OpenDILab (상하이 AI Lab 등)

---

## TL;DR

MuZero 계열 MCTS 알고리즘 9종을 단일 프레임워크로 통합한 최초의 통합 벤치마크. Python(ptree)과 C++(ctree) 듀얼 MCTS 구현, 혼합 이종 컴퓨팅으로 병목 최적화. 20개 이상 환경에서 표준화된 비교 실험 제공.

---

## 배경 및 문제 의식

MCTS+RL 연구는 빠르게 발전하고 있지만 여러 문제가 있다:

- **분산된 구현**: AlphaZero, MuZero, EfficientZero 등 각 논문이 제각각의 코드베이스 사용
- **비교 어려움**: 동일한 조건에서 알고리즘 간 공정한 비교 불가
- **재현성 문제**: 하이퍼파라미터, 환경, 평가 방식이 논문마다 다름
- **접근성 장벽**: 각 구현의 학습 곡선이 높고 모듈화가 부족

LightZero는 이를 해결하는 통합 프레임워크를 제공한다.

---

## 방법론

### 프레임워크 아키텍처

3개의 핵심 모듈로 알고리즘과 시스템을 분리:

```
┌─────────────────────────────────────────────────┐
│                    LightZero                     │
├───────────┬────────────────────┬────────────────┤
│   Model   │      Policy        │      MCTS      │
│           │                    │                │
│ 네트워크   │ Learning           │ Python(ptree)  │
│ 구조 정의  │ Collecting         │ C++(ctree)     │
│ 초기화     │ Evaluation         │                │
│ Forward    │ 환경 상호작용       │ 트리 구조 및   │
│           │                    │ Policy 연동    │
└───────────┴────────────────────┴────────────────┘
```

### 혼합 이종 컴퓨팅 (Mixed Heterogeneous Computing)

MCTS에서 가장 시간이 많이 걸리는 부분을 선택적으로 가속:

- **C++ ctree**: MCTS 트리 연산 (Selection, Expansion, Backprop) — 낮은 레이턴시, 고성능
- **Python ptree**: 프로토타이핑 및 커스터마이징에 유리
- **GPU**: 신경망 forward pass (Representation, Dynamics, Prediction) 배치 처리
- **CPU 멀티프로세싱**: 환경 병렬화 (vectorized environments)

```
환경 (CPU, 병렬) → 관측값 배치 → GPU 신경망 추론
                                      ↓
                               C++ MCTS 트리 연산 (CPU)
                                      ↓
                               Policy 업데이트 (GPU)
```

### 지원 알고리즘 (9종)

| 알고리즘 | 특징 |
|---|---|
| MuZero | 기본 모델 기반 MCTS |
| EfficientZero | 데이터 효율성 향상 |
| Sampled MuZero | 연속 행동 공간 지원 |
| Gumbel MuZero | 시뮬레이션 수 감소 |
| AlphaZero | 완전 정보 게임 |
| MuZero Unplugged | 오프라인 RL |
| Stochastic MuZero | 확률적 환경 |
| UniZero | 스케일러블 세계 모델 |
| ReZero (2024) | 빠른 reanalyze |

### 지원 환경 (20개+)

Board games (Chess, Go, Gomoku), Atari, MuJoCo, MiniGrid, GoBigger 등

---

## 핵심 실험 결과

### NeurIPS 2023 Spotlight 주요 기여

- 9개 알고리즘을 동일한 코드베이스에서 공정하게 비교한 최초 연구
- 알고리즘별 sample efficiency vs. wall-clock 효율 trade-off 정량화
- 환경별 알고리즘 특성 분석 (보드게임, Atari, 연속 제어 등)

### 2024 확장: ReZero

**"ReZero: Boosting MCTS-based Algorithms by Just-in-Time and Speedy Reanalyze"** ([arXiv 2404.16364](https://arxiv.org/html/2404.16364v1))

MuZero의 reanalyze 단계를 "just-in-time" 방식으로 개선:
- 학습 속도 대폭 향상
- sample efficiency 유지
- LightZero 프레임워크에 통합

### 성능 특성

- C++ ctree가 Python ptree 대비 MCTS 연산에서 **약 5~10배** 빠름
- 배치 신경망 추론으로 GPU 활용률 최적화
- 멀티프로세스 환경 병렬화로 데이터 수집 병목 해소

---

## 의의 및 한계

**의의**
- MCTS+RL 연구의 표준 벤치마크로 자리잡음
- 새 알고리즘을 빠르게 프로토타입하고 기존 알고리즘과 비교 가능
- 오픈소스로 지속적으로 유지보수 및 확장 중

**한계**
- 분산 학습(multi-node) 지원은 제한적
- 아직 완전한 GPU-native MCTS 구현은 아님 (C++ CPU 기반 트리 연산)
- 일부 최신 알고리즘은 아직 통합 전

---

## 참고 링크

- [GitHub](https://github.com/opendilab/LightZero)
- [NeurIPS 2023 Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/765043fe026f7d704c96cec027f13843-Paper-Datasets_and_Benchmarks.pdf)
- [OpenReview](https://openreview.net/forum?id=oIUXpBnyjv)
- [ReZero 논문](https://arxiv.org/html/2404.16364v1)
- [UniZero 논문](https://arxiv.org/html/2406.10667v2)
