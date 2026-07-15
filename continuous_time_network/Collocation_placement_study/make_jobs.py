"""Write jobs.txt (outstanding runs only) and the HTCondor submit file.
The latin baseline is NOT rerun (it exists in ../Causal_weighting and
../Initial_pass at density 20 and is imported at analysis time), so the
sweep is: {uniform, anchored} x {unweighted, causal} x T in {14, 27} x
seeds {0, 1, 2} = 24 runs."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "runs"

SUBMIT = """\
universe                = vanilla
executable              = wrapper.sh
arguments               = $(args)
output                  = condor/logs/$(Cluster).$(Process).out
error                   = condor/logs/$(Cluster).$(Process).err
log                     = condor/logs/$(Cluster).log
request_cpus            = 1
request_memory          = 2048
should_transfer_files   = NO
getenv                  = False
+UseOS                  = "el9"
+JobCategory            = "medium"
queue args from jobs.txt
"""

if __name__ == "__main__":
    (HERE / "condor" / "logs").mkdir(parents=True, exist_ok=True)
    lines, done = [], 0
    for arm in ("causal", "unweighted"):          # slow causal jobs first
        for horizon in (27.0, 14.0):
            for placement in ("anchored", "uniform"):
                for seed in (0, 1, 2):
                    tag = f"{arm}_T{horizon:g}_{placement}_s{seed}"
                    if (RESULTS / f"{tag}.json").exists():
                        done += 1
                        continue
                    lines.append(f"--arm {arm} --T {horizon:g} "
                                 f"--placement {placement} --seed {seed}")
    (HERE / "jobs.txt").write_text("\n".join(lines) + "\n" if lines else "")
    (HERE / "placement.sub").write_text(SUBMIT)
    print(f"{len(lines)} jobs to run, {done} already done -> jobs.txt + placement.sub")
