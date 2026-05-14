from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
import types
from collections import deque
from pathlib import Path

import numpy as np

from examples.DobotXtrainer.eval_files.model2dobotxtrainer_interface import DobotXtrainerModelClient
from examples.DobotXtrainer.eval_files.rtc_action_queue import RTCActionQueue


DEFAULT_OPENPI_ROOT = Path("/home/Xtrainer/ziyu/openpi05/openpi-main")
# Edit these reset joints directly in radians. They are converted to degrees right before ServoJ.
DEFAULT_LEFT_RESET_RAD = np.array([-1.4710289, 0.5990488, -2.1300774, 0.6807364, 1.4867578, 1.6206799], dtype=np.float32)
DEFAULT_RIGHT_RESET_RAD = np.array([np.pi / 2, 0.0, np.pi / 2, 0.0, -np.pi / 2, -np.pi / 2], dtype=np.float32)
DEFAULT_LEFT_RESET_GRIPPER = 1.0
DEFAULT_RIGHT_RESET_GRIPPER = 1.0


def ensure_optional_dm_env_stub() -> None:
    if "dm_env" in sys.modules:
        return
    try:
        import dm_env  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except ModuleNotFoundError:
        # Dobot RealEnv imports dm_env, but its dm_env return path is commented out.
        # Keep deployment independent from this unused optional package.
        sys.modules["dm_env"] = types.ModuleType("dm_env")


def import_real_env(openpi_root: Path):
    openpi_root = openpi_root.expanduser().resolve()
    if not openpi_root.exists():
        raise FileNotFoundError(f"OpenPI repo not found: {openpi_root}")
    if str(openpi_root) not in sys.path:
        sys.path.insert(0, str(openpi_root))
    ensure_optional_dm_env_stub()
    from examples.xtrainer_real.real_env import RealEnv  # pylint: disable=import-outside-toplevel

    return RealEnv


def wait_period(period_s: float, tic: float) -> None:
    remain = period_s - (time.time() - tic)
    if remain > 0:
        time.sleep(remain)


class RTCLatencyTracker:
    def __init__(self, control_period: float, maxlen: int = 10, mode: str = "p90"):
        self.control_period = float(control_period)
        self.mode = mode
        self._latencies: deque[float] = deque(maxlen=max(1, int(maxlen)))

    def add_latency(self, seconds: float) -> None:
        self._latencies.append(max(0.0, float(seconds)))

    def estimate_delay_steps(self) -> int:
        if not self._latencies:
            return 0
        values = np.asarray(self._latencies, dtype=np.float32)
        if self.mode == "max":
            latency = float(np.max(values))
        elif self.mode == "mean":
            latency = float(np.mean(values))
        elif self.mode == "p90":
            latency = float(np.percentile(values, 90))
        else:
            raise ValueError(f"Unsupported RTC latency mode: {self.mode}")
        return max(0, int(round(latency / self.control_period)))


def active_joint_indices(arms: str) -> np.ndarray:
    if arms == "left":
        return np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    if arms == "right":
        return np.array([7, 8, 9, 10, 11, 12], dtype=np.int64)
    return np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)


def qpos_rad_to_deg_grip(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    qpos_deg = qpos.copy()
    qpos_deg[:6] = np.rad2deg(qpos_deg[:6])
    qpos_deg[7:13] = np.rad2deg(qpos_deg[7:13])
    return qpos_deg


def freeze_inactive_arm(command_deg: np.ndarray, current_qpos: np.ndarray, arms: str) -> np.ndarray:
    command_deg = np.asarray(command_deg, dtype=np.float32).copy()
    current_deg = qpos_rad_to_deg_grip(current_qpos)
    if arms == "left":
        command_deg[7:] = current_deg[7:]
    elif arms == "right":
        command_deg[:7] = current_deg[:7]
    return command_deg


def build_default_reset_target_deg() -> np.ndarray:
    left_deg_grip = np.concatenate(
        [
            np.rad2deg(DEFAULT_LEFT_RESET_RAD.astype(np.float32, copy=False)),
            np.array([DEFAULT_LEFT_RESET_GRIPPER], dtype=np.float32),
        ]
    )
    right_deg_grip = np.concatenate(
        [
            np.rad2deg(DEFAULT_RIGHT_RESET_RAD.astype(np.float32, copy=False)),
            np.array([DEFAULT_RIGHT_RESET_GRIPPER], dtype=np.float32),
        ]
    )
    return np.concatenate([left_deg_grip, right_deg_grip]).astype(np.float32)


def move_to_reset_pose(env, arms: str, period_s: float, debug_step: bool = False) -> bool:
    obs = env.get_observation()
    current_qpos = np.asarray(obs["qpos"], dtype=np.float32)
    current_deg = qpos_rad_to_deg_grip(current_qpos)

    # Reset always drives both arms to the configured neutral pose, regardless of rollout mode.
    target_deg = build_default_reset_target_deg()
    left_steps = int(np.max(np.abs(target_deg[:6] - current_deg[:6])) / 2.0) + 1
    right_steps = int(np.max(np.abs(target_deg[7:13] - current_deg[7:13])) / 2.0) + 1
    steps = max(1, min(100, max(left_steps, right_steps)))
    active_indices = active_joint_indices("both")
    last_command = current_deg.copy()
    original_arms = env.arms

    try:
        env.arms = "both"
        for reset_idx, alpha in enumerate(np.linspace(0.0, 1.0, steps)):
            interp = current_deg + (target_deg - current_deg) * (1.0 - np.cos(np.pi * alpha)) / 2.0
            max_delta = float(np.max(np.abs(interp[active_indices] - last_command[active_indices])))
            if debug_step and not wait_for_debug_step(
                next_step=reset_idx + 1,
                local_idx=reset_idx,
                execute_len=steps,
                robot_command=interp,
                max_delta=max_delta,
                arms="both",
                phase="reset",
            ):
                return False

            tic = time.time()
            env.step(interp, single_arm=False)
            env.step_gripper(interp)
            wait_period(period_s, tic)
            last_command = interp
    finally:
        env.arms = original_arms
    return True


def get_images_in_train_order(obs: dict) -> list[np.ndarray]:
    images = obs["images"]
    return [
        images["cam_high"],
        images["cam_left_wrist"],
        images["cam_right_wrist"],
    ]


def format_deg_grip(vec: np.ndarray) -> str:
    return np.array2string(np.asarray(vec, dtype=np.float32), precision=3, suppress_small=True)


def describe_arm_command(label: str, current_deg: np.ndarray, target_deg: np.ndarray) -> list[str]:
    joint_delta = np.abs(target_deg[:6] - current_deg[:6])
    max_idx = int(np.argmax(joint_delta))
    return [
        f"[safety] {label} current(deg/grip): {format_deg_grip(current_deg)}",
        f"[safety] {label} target (deg/grip): {format_deg_grip(target_deg)}",
        f"[safety] {label} |delta| joints:  {format_deg_grip(joint_delta)}",
        f"[safety] {label} max joint delta: joint{max_idx + 1} current={current_deg[max_idx]:.2f} target={target_deg[max_idx]:.2f} delta={joint_delta[max_idx]:.2f}",
        f"[safety] {label} grip current={current_deg[6]:.3f} target={target_deg[6]:.3f} delta={abs(float(target_deg[6] - current_deg[6])):.3f}",
    ]


def build_safety_report(current_qpos: np.ndarray, robot_command: np.ndarray, arms: str) -> tuple[str, float, str, int]:
    current_deg = qpos_rad_to_deg_grip(current_qpos)
    target_deg = np.asarray(robot_command, dtype=np.float32)

    joint_deltas = np.abs(target_deg - current_deg)
    active_indices = active_joint_indices(arms)
    active_joint_deltas = joint_deltas[active_indices]
    max_local_idx = int(np.argmax(active_joint_deltas))
    max_joint_idx = int(active_indices[max_local_idx])
    max_delta = float(active_joint_deltas[max_local_idx])

    if max_joint_idx < 6:
        arm_label = "left"
        joint_name = f"joint{max_joint_idx + 1}"
    else:
        arm_label = "right"
        joint_name = f"joint{max_joint_idx - 6}"

    lines = [
        f"[safety] current left (deg/grip): {format_deg_grip(current_deg[:7])}",
        f"[safety] target  left (deg/grip): {format_deg_grip(target_deg[:7])}",
        f"[safety] current right(deg/grip): {format_deg_grip(current_deg[7:])}",
        f"[safety] target  right(deg/grip): {format_deg_grip(target_deg[7:])}",
    ]
    lines.extend(describe_arm_command("left ", current_deg[:7], target_deg[:7]))
    lines.extend(describe_arm_command("right", current_deg[7:], target_deg[7:]))
    lines.append(
        f"[safety] offending joint: {arm_label}.{joint_name} current={current_deg[max_joint_idx]:.2f} target={target_deg[max_joint_idx]:.2f} delta={max_delta:.2f}"
    )
    return "\n".join(lines), max_delta, arm_label, max_joint_idx


def wait_for_debug_step(
    next_step: int,
    local_idx: int,
    execute_len: int,
    robot_command: np.ndarray,
    max_delta: float,
    arms: str,
    phase: str = "policy",
) -> bool:
    print(
        f"\n[debug:{phase}] next_step={next_step:04d} chunk={local_idx + 1}/{execute_len} "
        f"arms={arms} max_joint_delta_deg={max_delta:.2f}"
    )
    print("[debug] left  target(deg/grip):", np.array2string(robot_command[:7], precision=3, suppress_small=True))
    print("[debug] right target(deg/grip):", np.array2string(robot_command[7:], precision=3, suppress_small=True))

    while True:
        try:
            user_input = input("[debug] Press Enter to execute exactly one step, or type q/quit/stop to exit: ")
        except EOFError:
            print("[debug] EOF received; exiting before executing this step.")
            return False
        command = user_input.strip().lower()
        if command == "":
            return True
        if command in {"q", "quit", "stop", "exit"}:
            print("[debug] Stop requested before executing this step.")
            return False
        print("[debug] Safety gate: only empty Enter executes; q/quit/stop exits.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy a trained StarVLA DobotXtrainer policy on the real robot.")
    parser.add_argument("--policy_ckpt_path", type=str, required=True, help="Path to the StarVLA checkpoint .pt/.safetensors file.")
    parser.add_argument("--task", type=str, required=True, help="Language instruction sent to the policy server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Policy server host.")
    parser.add_argument("--port", type=int, default=5694, help="Policy server port.")
    parser.add_argument("--unnorm_key", type=str, default=None, help="Optional dataset statistics key.")
    parser.add_argument("--openpi_root", type=str, default=str(DEFAULT_OPENPI_ROOT), help="Path to the referenced openpi-main repo.")
    parser.add_argument("--arms", type=str, choices=["both", "left", "right"], default="both", help="Which arm(s) to actuate.")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of executed low-level actions.")
    parser.add_argument("--actions_per_chunk", type=int, default=8, help="How many actions to execute from each predicted chunk. Set <=0 to use the full chunk.")
    parser.add_argument("--control_period", type=float, default=0.07, help="Seconds between consecutive low-level commands.")
    parser.add_argument("--warmup_seconds", type=float, default=2.0, help="Time to wait after RealEnv init so cameras can warm up.")
    parser.add_argument("--max_joint_delta_deg", type=float, default=30.0, help="Abort if a predicted command deviates from current joints by more than this amount.")
    parser.add_argument("--debug_step", "--debug", action="store_true", help="Debug safety mode: require pressing Enter before every low-level action; q/quit/stop exits before executing.")
    parser.add_argument("--skip_reset", action="store_true", help="Skip moving the robot to the default reset pose before rollout.")
    parser.add_argument("--no_gripper", action="store_true", help="Disable gripper control when constructing RealEnv.")
    parser.add_argument("--use_ddim", action="store_true", help="Pass DDIM flags through to the policy server.")
    parser.add_argument("--num_ddim_steps", type=int, default=4, help="DDIM step count passed to the policy server.")
    parser.add_argument("--use_rtc", action="store_true", help="Enable RTC-guided server sampling for background chunk refresh.")
    parser.add_argument("--rtc_execution_horizon", type=int, default=8, help="RTC prefix execution horizon.")
    parser.add_argument("--rtc_inference_delay_init", type=int, default=4, help="Initial latency estimate in control steps before measurements are available.")
    parser.add_argument(
        "--rtc_prefix_attention_schedule",
        type=str,
        choices=["exp", "linear", "ones", "zeros"],
        default="exp",
        help="RTC prefix attention schedule.",
    )
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=5.0, help="Maximum RTC guidance clipping weight.")
    parser.add_argument("--rtc_gripper_guidance_scale", type=float, default=0.0, help="RTC guidance scale for gripper dimensions.")
    parser.add_argument(
        "--rtc_latency_mode",
        type=str,
        choices=["max", "p90", "mean"],
        default="p90",
        help="Statistic used to estimate RTC inference delay.",
    )
    parser.add_argument("--rtc_latency_window", type=int, default=10, help="Number of recent latencies to track.")
    parser.add_argument("--rtc_min_queue_size", type=int, default=2, help="Minimum queue size before requesting another chunk.")
    parser.add_argument("--dry_run_predict_once", action="store_true", help="Request an initial chunk and one RTC chunk, then exit without executing robot commands.")
    parser.add_argument("--dry_run_queue_steps", type=int, default=0, help="Run queue and prediction logic for N steps without calling env.step.")
    return parser


def main(args: argparse.Namespace) -> None:
    RealEnv = import_real_env(Path(args.openpi_root))
    env = RealEnv(False, arms=args.arms, no_gripper=args.no_gripper)
    time.sleep(args.warmup_seconds)

    dry_run = bool(args.dry_run_predict_once or args.dry_run_queue_steps > 0)
    if not args.skip_reset and not dry_run:
        if not move_to_reset_pose(env, args.arms, args.control_period, debug_step=args.debug_step):
            return
    elif dry_run and not args.skip_reset:
        print("[dry-run] Skipping reset pose because dry run must not execute robot commands.")

    model = DobotXtrainerModelClient(
        policy_ckpt_path=args.policy_ckpt_path,
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )

    single_arm = args.arms != "both"
    executed_steps = 0
    if args.debug_step:
        print("[debug] Step-by-step safety mode enabled. One empty Enter executes one low-level action.")

    initial_obs = env.get_observation()
    tic = time.time()
    initial_chunk, initial_meta = model.predict_normalized_action_chunk(
        images=get_images_in_train_order(initial_obs),
        task_description=args.task,
        qpos_left_right=initial_obs["qpos"],
        use_rtc=False,
        use_ddim=args.use_ddim,
        num_ddim_steps=args.num_ddim_steps,
    )
    initial_latency = time.time() - tic
    print(
        f"[rtc] initial normalized chunk shape={initial_chunk.shape} "
        f"latency={initial_latency:.3f}s meta={initial_meta}"
    )

    action_queue = RTCActionQueue(
        action_chunk_size=model.action_chunk_size,
        action_dim=14,
        execution_horizon=args.rtc_execution_horizon,
        min_queue_size=args.rtc_min_queue_size,
    )
    action_queue.reset(initial_chunk)

    latency_tracker = RTCLatencyTracker(
        control_period=args.control_period,
        maxlen=args.rtc_latency_window,
        mode=args.rtc_latency_mode,
    )
    latency_tracker.add_latency(args.rtc_inference_delay_init * args.control_period)
    latency_tracker.add_latency(initial_latency)

    if args.dry_run_predict_once:
        prev_leftover = action_queue.get_left_over()
        rtc_delay = latency_tracker.estimate_delay_steps()
        tic = time.time()
        rtc_chunk, rtc_meta = model.predict_normalized_action_chunk(
            images=get_images_in_train_order(initial_obs),
            task_description=args.task,
            qpos_left_right=initial_obs["qpos"],
            use_rtc=True,
            prev_chunk_left_over=prev_leftover,
            inference_delay=rtc_delay,
            execution_horizon=args.rtc_execution_horizon,
            prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            max_guidance_weight=args.rtc_max_guidance_weight,
            gripper_guidance_scale=args.rtc_gripper_guidance_scale,
            use_ddim=args.use_ddim,
            num_ddim_steps=args.num_ddim_steps,
        )
        rtc_latency = time.time() - tic
        print(
            f"[dry-run] initial_shape={initial_chunk.shape} rtc_shape={rtc_chunk.shape} "
            f"leftover_shape={None if prev_leftover is None else prev_leftover.shape} "
            f"estimated_delay_steps={rtc_delay} latency={rtc_latency:.3f}s meta={rtc_meta}"
        )
        return

    queue_lock = threading.Lock()
    shared_lock = threading.Lock()
    request_event = threading.Event()
    stop_event = threading.Event()
    shared = {
        "latest_obs": initial_obs,
        "request_in_flight": False,
        "error": None,
        "last_rtc_meta": None,
        "last_latency": None,
    }

    def request_background_inference() -> None:
        with shared_lock:
            if shared["request_in_flight"]:
                return
            shared["request_in_flight"] = True
        request_event.set()

    def inference_worker() -> None:
        while not stop_event.is_set():
            if not request_event.wait(timeout=0.1):
                continue
            request_event.clear()
            if stop_event.is_set():
                break

            try:
                with shared_lock:
                    obs_snapshot = shared["latest_obs"]
                qpos_snapshot = np.asarray(obs_snapshot["qpos"], dtype=np.float32).copy()
                images_snapshot = [image.copy() for image in get_images_in_train_order(obs_snapshot)]
                with queue_lock:
                    prev_leftover = action_queue.get_left_over()

                inferred_delay = latency_tracker.estimate_delay_steps()
                tic = time.time()
                new_chunk, rtc_meta = model.predict_normalized_action_chunk(
                    images=images_snapshot,
                    task_description=args.task,
                    qpos_left_right=qpos_snapshot,
                    use_rtc=args.use_rtc,
                    prev_chunk_left_over=prev_leftover,
                    inference_delay=inferred_delay,
                    execution_horizon=args.rtc_execution_horizon,
                    prefix_attention_schedule=args.rtc_prefix_attention_schedule,
                    max_guidance_weight=args.rtc_max_guidance_weight,
                    gripper_guidance_scale=args.rtc_gripper_guidance_scale,
                    use_ddim=args.use_ddim,
                    num_ddim_steps=args.num_ddim_steps,
                )
                latency = time.time() - tic
                actual_skip_steps = max(0, int(round(latency / args.control_period)))
                with queue_lock:
                    action_queue.merge(new_chunk, actual_skip_steps)
                    queue_len = len(action_queue)
                latency_tracker.add_latency(latency)
                with shared_lock:
                    shared["request_in_flight"] = False
                    shared["last_rtc_meta"] = rtc_meta
                    shared["last_latency"] = latency
                print(
                    f"[rtc] merged chunk shape={new_chunk.shape} skip={actual_skip_steps} "
                    f"latency={latency:.3f}s queue={queue_len} meta={rtc_meta}"
                )
            except Exception:
                with shared_lock:
                    shared["error"] = traceback.format_exc()
                    shared["request_in_flight"] = False
                stop_event.set()

    worker = threading.Thread(target=inference_worker, name="rtc-inference-worker", daemon=True)
    worker.start()

    max_steps = args.dry_run_queue_steps if args.dry_run_queue_steps > 0 else args.max_steps
    try:
        while executed_steps < max_steps:
            with shared_lock:
                if shared["error"] is not None:
                    raise RuntimeError(f"Background RTC inference failed:\n{shared['error']}")

            loop_tic = time.time()
            current_obs = env.get_observation()
            with shared_lock:
                shared["latest_obs"] = current_obs

            with queue_lock:
                if len(action_queue) == 0:
                    raise RuntimeError("RTC action queue became empty before a new chunk arrived.")
                normalized_action = action_queue.pop()
                queue_len_after_pop = len(action_queue)
                should_request = action_queue.should_request()

            if should_request:
                request_background_inference()

            raw_state_model = model.reorder_env_state_to_model(current_obs["qpos"])
            action_model = model.normalized_action_to_model_action(normalized_action, raw_state_model)
            robot_command = model.model_action_to_robot_command(action_model)
            robot_command = freeze_inactive_arm(robot_command, current_obs["qpos"], args.arms)
            safety_report, max_delta, arm_label, joint_idx = build_safety_report(
                current_obs["qpos"],
                robot_command,
                arms=args.arms,
            )

            if max_delta > args.max_joint_delta_deg:
                print(safety_report)
                raise RuntimeError(
                    f"Refusing to execute unsafe action at step {executed_steps}: "
                    f"offending_joint={arm_label}.joint{joint_idx + 1 if joint_idx < 6 else joint_idx - 6} "
                    f"max_joint_delta_deg={max_delta:.2f} > {args.max_joint_delta_deg:.2f}"
                )

            if args.debug_step and not dry_run and not wait_for_debug_step(
                next_step=executed_steps + 1,
                local_idx=0,
                execute_len=1,
                robot_command=robot_command,
                max_delta=max_delta,
                arms=args.arms,
            ):
                return

            if not dry_run:
                env.step(robot_command, single_arm=single_arm)
                if not args.no_gripper:
                    env.step_gripper(robot_command)
                model.commit_executed_action(action_model)

            executed_steps += 1
            print(
                f"[step {executed_steps:04d}] queue={queue_len_after_pop} "
                f"max_joint_delta_deg={max_delta:.2f}"
            )
            wait_period(args.control_period, loop_tic)
    finally:
        stop_event.set()
        request_event.set()
        worker.join(timeout=2.0)


if __name__ == "__main__":
    try:
        main(build_argparser().parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted by user; exiting.")
