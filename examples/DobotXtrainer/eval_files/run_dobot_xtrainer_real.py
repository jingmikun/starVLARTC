from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np

from examples.DobotXtrainer.eval_files.model2dobotxtrainer_interface import DobotXtrainerModelClient


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
    return parser


def main(args: argparse.Namespace) -> None:
    RealEnv = import_real_env(Path(args.openpi_root))
    env = RealEnv(False, arms=args.arms, no_gripper=args.no_gripper)
    time.sleep(args.warmup_seconds)

    if not args.skip_reset:
        if not move_to_reset_pose(env, args.arms, args.control_period, debug_step=args.debug_step):
            return

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

    while executed_steps < args.max_steps:
        obs = env.get_observation()
        action_chunk_model = model.predict_action_chunk(
            images=get_images_in_train_order(obs),
            task_description=args.task,
            qpos_left_right=obs["qpos"],
            use_ddim=args.use_ddim,
            num_ddim_steps=args.num_ddim_steps,
        )

        if action_chunk_model.ndim != 2 or action_chunk_model.shape[1] != 14:
            raise RuntimeError(f"Expected action chunk shape [T, 14], got {action_chunk_model.shape}")

        chunk_len = len(action_chunk_model)
        execute_len = chunk_len if args.actions_per_chunk <= 0 else min(chunk_len, args.actions_per_chunk)

        for local_idx in range(execute_len):
            if executed_steps >= args.max_steps:
                break

            current_obs = env.get_observation()
            action_model = action_chunk_model[local_idx]
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

            if args.debug_step and not wait_for_debug_step(
                next_step=executed_steps + 1,
                local_idx=local_idx,
                execute_len=execute_len,
                robot_command=robot_command,
                max_delta=max_delta,
                arms=args.arms,
            ):
                return

            tic = time.time()
            env.step(robot_command, single_arm=single_arm)
            if not args.no_gripper:
                env.step_gripper(robot_command)
            wait_period(args.control_period, tic)

            model.commit_executed_action(action_model)
            executed_steps += 1
            print(
                f"[step {executed_steps:04d}] chunk={local_idx + 1}/{execute_len} "
                f"max_joint_delta_deg={max_delta:.2f}"
            )


if __name__ == "__main__":
    try:
        main(build_argparser().parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted by user; exiting.")
