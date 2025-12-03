WebSocietySimulator quick guide
===============================

What changed
------------
- Updated `websocietysimulator/agent/my_reflective_agent.py` with priors, stabilized reflection, and cleaner star formatting.
- Updated `websocietysimulator/agent/run_experiments.py` to run sequentially and feed groundtruth back to the agent.

Setup
-----
- Python 3.10+ recommended. Install deps: `pip install -r requirements.txt` (or `poetry install`).
- Ensure `data_process.py` is available in the repo root.

Process Yelp data
-----------------
Requires raw Yelp JSON files:
- `yelp_academic_dataset_business.json`
- `yelp_academic_dataset_user.json`
- `yelp_academic_dataset_review.json`

Run:
```
python data_process.py --input /path/to/raw_yelp --output data/yelp_processed
```

Run experiments
---------------
Example (Yelp track1 tasks/groundtruth):
```
python -m websocietysimulator.agent.run_experiments \
  --data-dir data/yelp_processed \
  --task-dir example/track1/yelp/tasks \
  --groundtruth-dir example/track1/yelp/groundtruth \
  --num-tasks 400
```
- Logs and metrics are written to `results_logs/baseline_*` and `results_logs/reflective_*`.

Notes
-----
- Stars are quantized to 0.1–0.2 steps in outputs for realism.
- Reflection uses groundtruth feedback each step; priors are built from `data_dir/review.json`.
