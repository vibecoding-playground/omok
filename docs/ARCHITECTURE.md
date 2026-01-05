# AlphaZero 스타일 오목 AI 구현 가이드

이 문서는 강화학습을 통해 오목을 두는 AI를 만드는 과정을 상세히 설명합니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [AlphaZero 알고리즘 이해하기](#2-alphazero-알고리즘-이해하기)
3. [게임 환경 구현](#3-게임-환경-구현)
4. [신경망 설계](#4-신경망-설계)
5. [MCTS (Monte Carlo Tree Search)](#5-mcts-monte-carlo-tree-search)
6. [학습 시스템](#6-학습-시스템)
7. [전체 흐름 요약](#7-전체-흐름-요약)

---

## 1. 프로젝트 개요

### 1.1 목표

**"스스로 오목을 배우는 AI 만들기"**

사람이 가르쳐주지 않아도, AI가 자기 자신과 대국하면서(Self-Play) 오목을 마스터하는 시스템을 구현합니다.

### 1.2 왜 AlphaZero 방식인가?

기존 게임 AI 접근법:

| 방식 | 설명 | 한계 |
|-----|------|-----|
| **규칙 기반** | 전문가가 "이런 상황엔 이렇게" 규칙 작성 | 모든 상황 커버 불가능 |
| **Minimax** | 모든 경우의 수 탐색 | 오목은 경우의 수가 너무 많음 |
| **지도학습** | 프로 기보로 학습 | 기보 데이터 필요, 프로 수준이 한계 |
| **AlphaZero** | 자가 대국으로 학습 | 데이터 없이 초인적 수준 도달 가능 |

AlphaZero는 2017년 DeepMind가 발표한 알고리즘으로, 바둑/체스/장기에서 인간 최고수를 압도했습니다.

### 1.3 핵심 아이디어

```
신경망 + 트리 탐색 = 강력한 의사결정
```

- **신경망**: "이 상황에서 어디가 좋아 보이는지" 직관적 판단
- **트리 탐색**: "실제로 몇 수 앞을 내다보며" 검증

둘을 결합하면, 직관으로 유망한 수를 좁히고 → 탐색으로 확인하는 효율적인 시스템이 됩니다.

---

## 2. AlphaZero 알고리즘 이해하기

### 2.1 세 가지 핵심 요소

```
┌─────────────────────────────────────────────────────┐
│                   AlphaZero                         │
├─────────────────────────────────────────────────────┤
│  1. Policy-Value Network (신경망)                   │
│     → 좋은 수 추천 + 승률 예측                       │
│                                                     │
│  2. MCTS (Monte Carlo Tree Search)                 │
│     → 여러 수 앞을 시뮬레이션해서 최선의 수 선택      │
│                                                     │
│  3. Self-Play (자가 대국)                           │
│     → 자기 자신과 대국하며 학습 데이터 생성          │
└─────────────────────────────────────────────────────┘
```

### 2.2 학습 사이클

```
         ┌──────────────────────────────────────┐
         │                                      │
         ▼                                      │
    ┌─────────┐      ┌─────────┐      ┌────────┴──┐
    │ 자가대국 │ ──→  │ 데이터  │ ──→  │   학습    │
    │(Self-Play)│     │  수집   │      │ (Training)│
    └─────────┘      └─────────┘      └───────────┘
         │                                      │
         │         신경망이 개선됨               │
         └──────────────────────────────────────┘
```

1. **자가대국**: 현재 신경망 + MCTS로 게임 진행
2. **데이터 수집**: 각 상황에서의 (상태, MCTS 결과, 승패) 저장
3. **학습**: 수집된 데이터로 신경망 업데이트
4. **반복**: 더 강해진 신경망으로 다시 자가대국

---

## 3. 게임 환경 구현

### 3.1 보드 표현 (`game/board.py`)

오목 보드를 어떻게 컴퓨터가 이해할 수 있는 형태로 바꿀까요?

```python
# 내부 저장: 9x9 정수 배열
#  0 = 빈 칸
#  1 = 흑돌
# -1 = 백돌

self._board = np.zeros((9, 9), dtype=np.int8)
```

### 3.2 신경망 입력 형태

신경망에는 "현재 플레이어 관점"에서 본 상태를 전달합니다:

```python
def get_state(self) -> np.ndarray:
    """(2, 9, 9) 텐서 반환"""
    state = np.zeros((2, 9, 9), dtype=np.float32)

    # 채널 0: 내 돌 위치 (1이면 내 돌 있음)
    state[0] = (self._board == self._current_player)

    # 채널 1: 상대 돌 위치 (1이면 상대 돌 있음)
    state[1] = (self._board == -self._current_player)

    return state
```

**왜 2채널인가?**

- 흑/백을 직접 구분하지 않고 "나/상대"로 표현
- 흑 차례든 백 차례든 신경망은 동일한 관점으로 학습
- 학습 효율이 2배 (흑 데이터 = 백 데이터)

### 3.3 승리 판정

```python
def _check_win(self, row: int, col: int) -> bool:
    """마지막 돌을 놓은 위치에서 5목 확인"""

    # 4방향 검사: 가로, 세로, 대각선, 역대각선
    directions = [(0,1), (1,0), (1,1), (1,-1)]

    for dr, dc in directions:
        count = 1  # 방금 놓은 돌

        # 양쪽 방향으로 같은 색 돌 세기
        for sign in [1, -1]:
            r, c = row + sign*dr, col + sign*dc
            while 유효한_위치(r, c) and 같은_색(r, c):
                count += 1
                r, c = r + sign*dr, c + sign*dc

        if count >= 5:
            return True

    return False
```

---

## 4. 신경망 설계

### 4.1 Policy-Value Network 구조

```
입력: 보드 상태 (2, 9, 9)
          │
          ▼
    ┌───────────┐
    │ Conv Block │  3x3 컨볼루션 → BatchNorm → ReLU
    └─────┬─────┘
          │
          ▼
    ┌───────────┐
    │ ResBlock  │ ×4  잔차 연결로 깊은 네트워크 학습 가능
    └─────┬─────┘
          │
     ┌────┴────┐
     │         │
     ▼         ▼
┌─────────┐ ┌─────────┐
│ Policy  │ │  Value  │
│  Head   │ │  Head   │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
  (81,)        (1,)
 확률분포      승률
```

### 4.2 왜 이런 구조인가?

#### Residual Block (잔차 블록)

```python
class ResBlock(nn.Module):
    def forward(self, x):
        residual = x
        out = conv → bn → relu → conv → bn
        return relu(out + residual)  # ← 잔차 연결!
```

**잔차 연결의 효과:**
- 깊은 네트워크에서 gradient vanishing 방지
- "변화량"만 학습하면 되므로 학습이 쉬움
- 입력을 그대로 전달하는 경로가 있어 정보 손실 적음

#### Dual Head (이중 출력)

```python
# Policy Head: "어디에 두면 좋을까?"
policy = nn.Linear(features, 81)  # 81개 위치별 점수
policy = softmax(policy)          # 확률로 변환

# Value Head: "현재 얼마나 유리한가?"
value = nn.Linear(features, 1)    # 단일 값
value = tanh(value)               # -1 ~ +1 범위
```

**왜 하나의 네트워크에서 두 출력?**
- 특징 추출 부분을 공유하므로 효율적
- Policy와 Value가 서로 보완하며 학습
- AlphaZero 논문의 핵심 설계

### 4.3 출력 해석

```python
# Policy 출력: [0.01, 0.02, ..., 0.15, ..., 0.03]  (81개)
#              각 위치에 둘 확률

# Value 출력: 0.35
#            현재 상태에서 이길 확률 약 67% ((0.35+1)/2)
```

---

## 5. MCTS (Monte Carlo Tree Search)

### 5.1 MCTS란?

**"여러 가지 미래를 시뮬레이션해서 가장 좋은 수 찾기"**

체스 같은 게임에서 모든 경우의 수를 탐색하는 것은 불가능합니다.
MCTS는 "유망해 보이는" 경로를 집중적으로 탐색하는 영리한 방법입니다.

### 5.2 MCTS의 4단계

```
┌─────────────────────────────────────────────────────────┐
│  1. Selection (선택)                                    │
│     루트에서 시작해 가장 유망한 자식 노드로 이동         │
│     UCB 점수가 가장 높은 노드 선택                       │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  2. Expansion (확장)                                    │
│     리프 노드에 도달하면 자식 노드들 생성                │
│     신경망의 Policy로 각 자식의 prior 초기화            │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  3. Evaluation (평가)                                   │
│     신경망의 Value로 현재 상태 평가                      │
│     (기존 MCTS는 랜덤 플레이아웃 사용)                   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  4. Backup (역전파)                                     │
│     평가값을 루트까지 전파                               │
│     지나온 모든 노드의 통계 업데이트                     │
└─────────────────────────────────────────────────────────┘
```

### 5.3 UCB (Upper Confidence Bound) 점수

어떤 노드를 선택할지 결정하는 핵심 공식:

```python
def ucb_score(self, parent_visits, c_puct):
    # Q: 이 수의 평균 가치 (exploitation)
    Q = self.value_sum / self.visit_count

    # U: 탐색 보너스 (exploration)
    U = c_puct * self.prior * sqrt(parent_visits) / (1 + self.visit_count)

    return Q + U
```

**직관적 이해:**

- **Q (exploitation)**: "지금까지 이 수가 얼마나 좋았나?"
- **U (exploration)**: "이 수를 충분히 탐색했나?"
  - prior가 높으면: 신경망이 좋다고 한 수 → 더 탐색
  - visit_count가 낮으면: 아직 덜 본 수 → 더 탐색

이 균형이 MCTS의 핵심입니다.

### 5.4 AlphaZero MCTS vs 기존 MCTS

| 구분 | 기존 MCTS | AlphaZero MCTS |
|-----|----------|----------------|
| 노드 초기화 | uniform prior | **신경망 Policy** |
| 평가 | random rollout | **신경망 Value** |
| 속도 | 느림 (끝까지 플레이) | 빠름 (즉시 평가) |

### 5.5 코드로 보는 MCTS

```python
def search(self, board, num_simulations=200):
    root = Node(prior=0)
    self._expand(root, board)  # 신경망으로 자식 노드 생성

    for _ in range(num_simulations):
        node = root
        scratch_board = board.copy()
        path = [node]

        # 1. Selection: 리프까지 내려가기
        while node.is_expanded and not scratch_board.is_game_over:
            action, node = node.select_child(c_puct=1.5)
            scratch_board.play(action)
            path.append(node)

        # 2-3. Expansion & Evaluation
        if scratch_board.is_game_over:
            value = 게임_결과
        else:
            value = self._expand(node, scratch_board)

        # 4. Backup
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # 상대 관점으로 전환

    # 방문 횟수 기반으로 최종 확률 반환
    return visit_counts / visit_counts.sum()
```

### 5.6 Dirichlet Noise

학습 시 탐색의 다양성을 위해 루트 노드에 노이즈 추가:

```python
def _add_dirichlet_noise(self, root_node):
    noise = np.random.dirichlet([0.3] * num_valid_moves)

    for action, child in root_node.children.items():
        child.prior = 0.75 * child.prior + 0.25 * noise[action]
```

**왜 필요한가?**
- 신경망이 확신하는 수만 두면 다양한 상황 학습 불가
- 노이즈로 가끔 다른 수도 탐색 → 더 robust한 학습

---

## 6. 학습 시스템

### 6.1 Self-Play (자가 대국)

```python
def play_game(self):
    board = Board()
    records = []

    while not board.is_game_over:
        # MCTS로 수 결정
        mcts_probs = mcts.search(board)

        # 학습용 데이터 저장
        records.append({
            'state': board.get_state(),
            'policy': mcts_probs,
            'player': board.current_player
        })

        # 수 두기
        action = sample(mcts_probs)
        board.play(action)

    # 게임 끝: 각 상태에 승패 레이블 부여
    winner = board.winner
    for record in records:
        if record['player'] == winner:
            record['value'] = +1
        elif winner is None:
            record['value'] = 0
        else:
            record['value'] = -1

    return records
```

### 6.2 학습 데이터 구조

```python
# 하나의 학습 샘플
{
    'state': np.array (2, 9, 9),   # 보드 상태
    'policy': np.array (81,),      # MCTS가 찾은 최적 정책
    'value': float,                # 실제 게임 결과 (-1, 0, +1)
}
```

### 6.3 손실 함수

```python
def compute_loss(model_output, target):
    log_policy, value = model_output
    target_policy, target_value = target

    # Policy Loss: 크로스 엔트로피
    # "MCTS가 찾은 분포와 비슷해져라"
    policy_loss = -sum(target_policy * log_policy)

    # Value Loss: MSE
    # "실제 승패를 예측해라"
    value_loss = (value - target_value)^2

    return policy_loss + value_loss
```

**두 손실의 의미:**

- **Policy Loss**: MCTS의 "집단 지성"을 배우기
  - MCTS는 수백 번 시뮬레이션한 결과
  - 신경망은 이걸 한 번에 예측하도록 학습

- **Value Loss**: 게임 결과 예측력 향상
  - 어떤 상태가 유리한지 직접 판단하는 능력

### 6.4 Replay Buffer

```python
class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = []
        self.max_size = max_size

    def add(self, samples):
        self.buffer.extend(samples)
        # 오래된 샘플 제거
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

**왜 Replay Buffer?**
- 최근 게임만 학습하면 "망각" 발생
- 다양한 시점의 데이터 섞어서 학습 → 더 일반화된 학습

### 6.5 학습 루프 전체

```python
for iteration in range(num_iterations):
    # 1. Self-Play: 데이터 생성
    for _ in range(games_per_iteration):
        game_data = self_play()
        replay_buffer.add(game_data)

    # 2. Training: 신경망 업데이트
    for epoch in range(epochs):
        batch = replay_buffer.sample(batch_size)
        loss = train_step(model, batch)

    # 3. 체크포인트 저장
    if iteration % save_interval == 0:
        save_model()
```

---

## 7. 전체 흐름 요약

### 7.1 추론 시 (게임 플레이)

```
보드 상태
    │
    ▼
┌─────────────────────────────────────────┐
│              MCTS Search                │
│  ┌─────────────────────────────────┐   │
│  │ 신경망으로 Policy/Value 얻기    │   │
│  │         ↓                       │   │
│  │ UCB로 유망한 노드 선택          │   │
│  │         ↓                       │   │
│  │ 200번 시뮬레이션               │   │
│  │         ↓                       │   │
│  │ 방문 횟수 → 확률 변환          │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    │
    ▼
최선의 수 선택
```

### 7.2 학습 시

```
┌────────────────────────────────────────────────────┐
│                  학습 반복 (iteration)              │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────┐                                  │
│  │  Self-Play   │ ← 현재 신경망으로 게임           │
│  │  (50 games)  │                                  │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │ Replay Buffer│ ← (상태, MCTS정책, 승패) 저장    │
│  │  (50000 max) │                                  │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  ┌──────────────┐                                  │
│  │   Training   │ ← 랜덤 배치 샘플링하여 학습      │
│  │  (5 epochs)  │                                  │
│  └──────┬───────┘                                  │
│         │                                          │
│         ▼                                          │
│  신경망 업데이트됨 → 다음 iteration               │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 7.3 파일 구조와 역할

```
omok/
├── game/
│   └── board.py          # 게임 규칙, 상태 표현
│
├── model/
│   └── network.py        # Policy-Value 신경망
│
├── mcts/
│   └── search.py         # MCTS 알고리즘
│
├── training/
│   ├── self_play.py      # 자가 대국 생성
│   └── trainer.py        # 학습 루프
│
├── main.py               # CLI (학습/대전)
├── train_colab.ipynb     # GPU 학습 (PyTorch)
└── train_colab_tpu.ipynb # TPU 학습 (JAX)
```

---

## 부록: 핵심 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `num_simulations` | 200 | MCTS 시뮬레이션 횟수 |
| `c_puct` | 1.5 | 탐색/활용 균형 상수 |
| `dirichlet_alpha` | 0.3 | 노이즈 집중도 (작을수록 집중) |
| `temperature` | 1.0→0.1 | 수 선택 무작위성 (학습 초반 높게) |
| `num_res_blocks` | 4 | ResNet 깊이 |
| `num_filters` | 128 | 컨볼루션 채널 수 |
| `batch_size` | 128 | 학습 배치 크기 |
| `learning_rate` | 1e-3 | 학습률 |

---

## 참고 자료

- [AlphaZero 논문](https://arxiv.org/abs/1712.01815) - Mastering Chess and Shogi by Self-Play
- [AlphaGo Zero 논문](https://www.nature.com/articles/nature24270) - Mastering Go without Human Knowledge
- [MCTS 튜토리얼](https://www.youtube.com/watch?v=UXW2yZndl7U) - Monte Carlo Tree Search 설명
