# mctx: Monte Carlo Tree Search in JAX (Google DeepMind)

**GitHub**: [google-deepmind/mctx](https://github.com/google-deepmind/mctx)  
**PyPI**: [mctx](https://pypi.org/project/mctx/)  
**공개**: Google DeepMind  
**프레임워크**: JAX

---

## TL;DR

Google DeepMind가 공개한 JAX 기반 MCTS 라이브러리. AlphaZero, MuZero, Gumbel MuZero를 모두 지원하며, JAX의 JIT 컴파일과 `vmap`을 통해 전체 MCTS를 GPU/TPU에서 완전히 벡터화하여 실행한다. 논문이 아닌 라이브러리이지만, DeepMind 내부 연구에서 검증된 프로덕션 수준의 구현이다.

---

## 배경

DeepMind의 AlphaZero/MuZero 연구는 대부분 내부 TensorFlow/JAX 구현을 사용했다. `mctx`는 이 구현을 오픈소스로 공개한 것으로, MCTS를 JAX 생태계에서 완전히 활용할 수 있도록 설계되었다.

**핵심 설계 원칙**:
- MCTS 전체 루프를 JAX `jit`으로 컴파일
- 배치 차원에서 `vmap`으로 여러 환경 동시 처리
- 사용자가 representation, dynamics, prediction 함수만 제공하면 됨

---

## 방법론 및 구조

### 핵심 컴포넌트

**1. RootFnOutput** — 루트 상태 표현
```python
RootFnOutput(
    prior_logits,  # 정책 네트워크 출력 (로짓)
    value,         # 루트 상태의 가치 추정
    embedding      # 환경 모델에 전달할 상태 임베딩
)
```

**2. recurrent_fn** — Dynamics + Prediction 모델
```python
def recurrent_fn(params, rng_key, action, embedding):
    # 환경 모델: (action, state) → (next_state, reward)
    return RecurrentFnOutput(
        reward,        # 전이 보상
        discount,      # 할인 계수
        prior_logits,  # 다음 상태의 정책 로짓
        value          # 다음 상태의 가치
    ), new_embedding
```

**3. 고수준 정책 함수**
```python
# MuZero 정책
output = mctx.muzero_policy(params, rng_key, root, recurrent_fn, num_simulations)

# Gumbel MuZero 정책 (권장)
output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn, num_simulations)
```

### 병렬화 전략

**JIT 컴파일**:
- MCTS 시뮬레이션 루프 전체를 `jax.jit`으로 컴파일
- 시뮬레이션 오케스트레이션(어떤 노드를 방문할지)을 사전 계산(precompute)하여 동적 분기 제거
- XLA 컴파일러가 GPU/TPU 최적 실행 코드 생성

**Batched Execution (`vmap`)**:
```python
# 여러 환경을 동시에 처리
batched_policy = jax.vmap(mctx.gumbel_muzero_policy, ...)
outputs = batched_policy(params, rng_keys, roots, recurrent_fn, num_simulations)
```
- B개 환경의 MCTS를 단일 GPU 커널로 동시 실행
- 신경망 추론도 배치로 처리 → GPU 활용률 극대화

### Gumbel MuZero (권장 알고리즘)

`mctx`가 기본적으로 권장하는 알고리즘. 핵심 특성:
- **정책 개선 보장**: action value가 올바르게 추정되면 수학적으로 정책 개선을 보장
- **시뮬레이션 수 감소**: 기존 MuZero보다 적은 시뮬레이션으로 동등한 성능
- **순차적 halving**: 탐색 자원을 유망한 액션에 집중

---

## 핵심 특성 및 성능

### 설계 장점

| 특성 | 내용 |
|---|---|
| 완전한 GPU/TPU 실행 | MCTS 루프 포함 전체 연산이 가속기에서 실행 |
| JIT 컴파일 | 반복 실행 시 오버헤드 제거 |
| vmap 배치 처리 | 여러 환경 동시 처리 |
| 낮은 API 복잡도 | representation/dynamics만 제공하면 사용 가능 |
| 프로덕션 검증 | DeepMind 내부 연구에 사용 |

### 사용 시 고려사항

- JAX 생태계에 종속 (TensorFlow/PyTorch 사용자는 포팅 필요)
- 동적 트리 크기 변경이 JIT와 충돌 가능 → 고정 크기 트리 사용
- TPU에서 특히 높은 성능 (JAX의 TPU 최적화 활용)

---

## 의의

- MCTS의 "GPU-native" 실행 가능성을 실용적으로 입증
- AlphaZero/MuZero/Gumbel MuZero를 단일 API로 통합
- JAX 기반 RL 연구의 사실상 표준 MCTS 라이브러리로 자리잡음
- LightZero의 PyTorch 생태계와 함께 MCTS 구현의 양대 기준점

---

## 참고 링크

- [GitHub](https://github.com/google-deepmind/mctx)
- [PyPI](https://pypi.org/project/mctx/)
- [DeepMind 공식 소개](https://www.deepmind.com/open-source/monte-carlo-tree-search-in-jax)
- [정책 개선 데모](https://github.com/google-deepmind/mctx/blob/main/examples/policy_improvement_demo.py)
