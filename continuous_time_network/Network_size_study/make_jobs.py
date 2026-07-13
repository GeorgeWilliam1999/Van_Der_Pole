"""Write jobs.txt (one line of run_one.py arguments per outstanding run) and
the HTCondor submit file. Completed runs (their results/runs/<tag>.json
exists) are skipped, so resubmission after failures is safe and cheap."""
from pathlib import Path

import capacity

HERE = Path(__file__).resolve().parent

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
    for variant in capacity.VARIANTS:
        for horizon in capacity.HORIZONS:
            for depth in capacity.DEPTHS:
                for width in capacity.WIDTHS:
                    for s in range(len(capacity.starts())):
                        tag = f"{variant}_T{horizon:g}_d{depth}_w{width}_s{s}"
                        if (capacity.RESULTS / f"{tag}.json").exists():
                            done += 1
                            continue
                        lines.append(f"{variant} {horizon:g} {depth} {width} {s}")
    (HERE / "jobs.txt").write_text("\n".join(lines) + "\n" if lines else "")
    (HERE / "capacity.sub").write_text(SUBMIT)
    print(f"{len(lines)} jobs to run, {done} already done "
          f"-> jobs.txt + capacity.sub")
