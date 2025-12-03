# websocietysimulator/agent/run_experiments.py

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Type

# Ensure repository root is importable when running as a script (python path/to/run_experiments.py)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from websocietysimulator.agent.my_reflective_agent import BaselineUserAgent, ReflectiveUserAgent
from websocietysimulator.agent.simulation_agent import SimulationAgent
from websocietysimulator.simulator import Simulator


RESULTS_DIR = Path("results_logs")
RESULTS_DIR.mkdir(exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline vs reflective user agents on WebSocietySimulator tasks."
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("WSS_DATA_DIR"),
        help="Processed dataset directory containing item.json, review.json, user.json",
    )
    parser.add_argument(
        "--task-dir",
        default=os.getenv("WSS_TASK_DIR"),
        help="Directory with task_*.json files",
    )
    parser.add_argument(
        "--groundtruth-dir",
        default=os.getenv("WSS_GROUNDTRUTH_DIR"),
        help="Directory with groundtruth_*.json files",
    )
    parser.add_argument("--num-tasks", type=int, default=200, help="Number of tasks to run (None runs all)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"], help="Device for evaluation")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use LMDB-backed cache interaction tool (saves RAM, slightly more I/O).",
    )
    parser.add_argument(
        "--no-threading",
        action="store_true",
        help="Disable multi-threaded simulation (useful for debugging).",
    )
    parser.add_argument("--max-workers", type=int, default=8, help="Thread pool size when threading is enabled.")
    return parser.parse_args()


def _validate_paths(args: argparse.Namespace) -> None:
    missing = [name for name, path in [
        ("data_dir", args.data_dir),
        ("task_dir", args.task_dir),
        ("groundtruth_dir", args.groundtruth_dir),
    ] if not path or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing or invalid paths for: {', '.join(missing)}. "
            "Pass them via CLI flags or WSS_* environment variables."
        )


def run_agent(agent_cls: Type[SimulationAgent], run_name: str, args: argparse.Namespace) -> dict:
    """Build a Simulator, execute a given agent class with groundtruth feedback, evaluate, and persist logs."""
    print(f"\n=== Running {run_name} ===")
    sim = Simulator(data_dir=args.data_dir, device=args.device, cache=args.cache)
    sim.set_task_and_groundtruth(task_dir=args.task_dir, groundtruth_dir=args.groundtruth_dir)
    sim.set_agent(agent_cls)

    task_limit = None if args.num_tasks is None else args.num_tasks
    tasks = sim.tasks[:task_limit]
    groundtruth = sim.groundtruth_data[: len(tasks)]

    agent = agent_cls(llm=None)
    agent.set_interaction_tool(sim.interaction_tool)

    agent_outputs = []
    for idx, (task, gt) in enumerate(zip(tasks, groundtruth)):
        agent.insert_task(task)
        output = agent.workflow()
        agent_outputs.append({"task": task.to_dict(), "output": output})

        if hasattr(agent, "observe_groundtruth"):
            try:
                agent.observe_groundtruth(task=task.to_dict(), groundtruth=gt, prediction=output)
            except TypeError:
                # Fallback for older signatures
                agent.observe_groundtruth(task.to_dict(), gt, output)

        if (idx + 1) % 100 == 0:
            print(f"{run_name}: completed {idx + 1} / {len(tasks)} tasks")

    metrics_obj = sim.simulation_evaluator.calculate_metrics(
        simulated_data=[entry["output"] for entry in agent_outputs],
        real_data=groundtruth,
    )
    metrics = {
        "type": "simulation",
        "metrics": metrics_obj.__dict__,
        "data_info": {
            "evaluated_count": len(groundtruth),
            "original_simulation_count": len(agent_outputs),
            "original_ground_truth_count": len(sim.groundtruth_data),
        },
    }

    (RESULTS_DIR / f"{run_name}_outputs.json").write_text(json.dumps(agent_outputs, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"{run_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"{run_name} metrics:\n{json.dumps(metrics, indent=2)}")
    return metrics


def main():
    args = _parse_args()
    _validate_paths(args)

    baseline_metrics = run_agent(BaselineUserAgent, "baseline", args)
    reflective_metrics = run_agent(ReflectiveUserAgent, "reflective", args)

    print("\n=== Summary ===")
    print("Baseline:", json.dumps(baseline_metrics, indent=2))
    print("Reflective:", json.dumps(reflective_metrics, indent=2))


if __name__ == "__main__":
    main()
