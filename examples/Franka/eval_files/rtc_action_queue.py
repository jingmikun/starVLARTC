"""Franka RTC normalized action queue.

The implementation lives in ``starVLA.model.modules.action_model.rtc`` so that
Franka-like clients, including XTrainer later, can share the same behavior.
"""

from starVLA.model.modules.action_model.rtc import RTCActionQueue

__all__ = ["RTCActionQueue"]
