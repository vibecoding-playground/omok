# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero-style reinforcement learning Omok (Gomoku) AI with 9x9 board.

## Commands

```bash
# Install dependencies
uv sync

# Train the model
uv run python main.py train --iterations 50

# Play against AI
uv run python main.py play

# Quick test
uv run python -c "from game.board import Board; print(Board())"
```

## Architecture

- **game/board.py**: Board state (9x9), move validation, win detection (5-in-a-row)
- **model/network.py**: ResNet-based Policy-Value network (~1.2M params), MPS/CUDA/CPU auto-detection
- **mcts/search.py**: MCTS with PUCT, neural network guided search
- **training/**: Self-play data generation + training loop with replay buffer
- **main.py**: CLI with curses (arrow key navigation)

## Key Design Decisions

- State representation: `(2, 9, 9)` tensor (current player stones, opponent stones)
- Action space: 81 (flattened 9x9 grid)
- Value head output: `[-1, 1]` (win probability from current player's perspective)
- Device priority: MPS > CUDA > CPU
