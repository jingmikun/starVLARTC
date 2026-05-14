from __future__ import annotations

import numpy as np


class RTCActionQueue:
    def __init__(
        self,
        action_chunk_size: int,
        action_dim: int,
        execution_horizon: int,
        min_queue_size: int = 2,
    ):
        self.action_chunk_size = int(action_chunk_size)
        self.action_dim = int(action_dim)
        self.execution_horizon = int(execution_horizon)
        self.min_queue_size = int(min_queue_size)
        self._queue: list[np.ndarray] = []

    def reset(self, normalized_chunk: np.ndarray, start_index: int = 0) -> None:
        chunk = self._validate_chunk(normalized_chunk)
        start = max(0, min(int(start_index), len(chunk)))
        self._queue = [action.copy() for action in chunk[start:]]

    def pop(self) -> np.ndarray:
        """
        Pop one normalized action [D].
        Raise RuntimeError if empty.
        """
        if not self._queue:
            raise RuntimeError("RTCActionQueue is empty.")
        return self._queue.pop(0).copy()

    def get_left_over(self) -> np.ndarray | None:
        """
        Return normalized leftover [R,D] from current queue.
        If no leftover, return None.
        """
        if not self._queue:
            return None
        return np.stack(self._queue, axis=0).astype(np.float32, copy=False)

    def merge(self, normalized_chunk: np.ndarray, skip_steps: int) -> None:
        """
        Merge new chunk after skipping stale prefix.
        normalized_chunk: [H,D]
        skip_steps: actual delay steps.
        First version can replace current queue with list(normalized_chunk[skip_steps:]).
        """
        chunk = self._validate_chunk(normalized_chunk)
        skip = max(0, int(skip_steps))
        if skip >= len(chunk):
            skip = len(chunk) - 1
        self._queue = [action.copy() for action in chunk[skip:]]

    def should_request(self) -> bool:
        """
        Return True when queue length <= execution_horizon or <= min_queue_size.
        """
        return len(self._queue) <= self.execution_horizon or len(self._queue) <= self.min_queue_size

    def __len__(self) -> int:
        return len(self._queue)

    def _validate_chunk(self, normalized_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(normalized_chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"Expected normalized action chunk [H,D], got {chunk.shape}")
        if chunk.shape[1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got chunk shape {chunk.shape}")
        if chunk.shape[0] == 0:
            raise ValueError("Normalized action chunk must contain at least one action.")
        if self.action_chunk_size > 0 and chunk.shape[0] > self.action_chunk_size:
            raise ValueError(f"Expected chunk length <= {self.action_chunk_size}, got {chunk.shape[0]}")
        return chunk
