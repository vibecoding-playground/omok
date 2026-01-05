# Omok (오목) AI

AlphaZero 스타일 강화학습 기반 9x9 오목 AI입니다.

## 특징

- **AlphaZero 방식**: MCTS + Policy-Value Network
- **MPS 가속**: Apple Silicon 최적화
- **CLI 인터페이스**: 방향키로 조작하는 터미널 대전

## 설치

```bash
uv sync
```

## 사용법

### 학습
```bash
uv run python main.py train --iterations 50
```

옵션:
- `--iterations, -i`: 학습 반복 횟수 (기본: 50)
- `--checkpoint, -c`: 체크포인트에서 이어서 학습

### AI와 대전
```bash
uv run python main.py play
```

옵션:
- `--model, -m`: 모델 경로 (기본: `checkpoints/model_latest.pt`)
- `--sims, -s`: MCTS 시뮬레이션 횟수 (기본: 400)

### 조작법

| 키 | 동작 |
|---|------|
| ↑↓←→ | 커서 이동 |
| Enter/Space | 돌 놓기 |
| q | 종료 |

## 구조

```
omok/
├── main.py              # CLI 엔트리포인트
├── game/board.py        # 9x9 보드, 승리 판정
├── model/network.py     # ResNet Policy-Value Network
├── mcts/search.py       # MCTS (PUCT 알고리즘)
└── training/
    ├── self_play.py     # 자가 대국
    └── trainer.py       # 학습 루프
```
