"""Write jobs.txt (one line of run_one.py arguments per outstanding run),
canary_jobs.txt (three fixed real jobs for a pre-farm HTCondor smoke test),
and the two HTCondor submit files.

Completed runs (their results/runs/<tag>.json exists) are skipped, so
resubmission after failures is safe and cheap.

Argument lines use the run_one.py CLI's own --arm/--T/--density/--seed
flags (not bare positional values): HTCondor's `arguments = $(args)` passes
each jobs.txt line straight through wrapper.sh to run_one.py, and
run_one.py's CLI takes named flags -- as fixed by the local smoke-test
invocation this study was validated with (`run_one.py --arm unweighted
--T 7 --density 5 --seed 0 --smoke`) -- so the jobs file must speak the same
syntax.
"""
from pathlib import Path

import training

HERE = Path(__file__).resolve().parent

DENSITIES = (5, 80, 320)
HORIZONS = (7, 14, 27, 40)
SEEDS = (0, 1, 2)
ARMS = ("unweighted", "causal")

CANARY = (
    ("unweighted", 7, 5, 0),
    ("causal", 7, 5, 0),
    ("unweighted", 40, 320, 0),      # stresses N = 12,800 in the cheap arm
)

SUBMIT_TEMPLATE = """\
universe                = vanilla
executable              = wrapper.sh
arguments               = $(args)
output                  = condor/logs/$(Cluster).$(Process).out
error                   = condor/logs/$(Cluster).$(Process).err
log                     = condor/logs/$(Cluster).log
request_cpus            = 1
request_memory          = 3072
should_transfer_files   = NO
getenv                  = False
+UseOS                  = "el9"
+JobCategory            = "medium"
queue args from {jobs_file}
"""


def line(arm: str, horizon: float, density: float, seed: int) -> str:
    return f"--arm {arm} --T {horizon:g} --density {density:g} --seed {seed}"


if __name__ == "__main__":
    (HERE / "condor" / "logs").mkdir(parents=True, exist_ok=True)

    lines, done = [], 0
    for arm in ARMS:
        for horizon in HORIZONS:
            for density in DENSITIES:
                for seed in SEEDS:
                    tag = f"{arm}_T{horizon:g}_d{density:g}_s{seed}"
                    if (training.RESULTS / f"{tag}.json").exists():
                        done += 1
                        continue
                    lines.append(line(arm, horizon, density, seed))
    (HERE / "jobs.txt").write_text("\n".join(lines) + "\n" if lines else "")

    canary_lines = [line(arm, horizon, density, seed)
                    for arm, horizon, density, seed in CANARY]
    (HERE / "canary_jobs.txt").write_text("\n".join(canary_lines) + "\n")

    (HERE / "collocation.sub").write_text(
        SUBMIT_TEMPLATE.format(jobs_file="jobs.txt"))
    (HERE / "canary.sub").write_text(
        SUBMIT_TEMPLATE.format(jobs_file="canary_jobs.txt"))

    print(f"{len(lines)} jobs to run, {done} already done -> jobs.txt")
    print(f"{len(canary_lines)} canary jobs -> canary_jobs.txt")
    print("-> collocation.sub, canary.sub")
