#!/bin/sh
# One tiny job to prove the batch nodes can run the sweep before 135 jobs go in:
# the shared disk must be mounted and the TE conda python must import torch.
echo "host: $(hostname)"
echo "os:   $(cat /etc/os-release | grep PRETTY | cut -d= -f2)"
ls /data/bfys/gscriven/Van_Der_Pole/discrete_time_network/model.py \
    && echo "DATA_MOUNT_OK"
/data/bfys/gscriven/conda/envs/TE/bin/python - <<'PY'
import torch, numpy, scipy
torch.set_num_threads(1)
print("torch", torch.__version__, "| numpy", numpy.__version__,
      "| scipy", scipy.__version__)
x = torch.linspace(0, 1, 10, dtype=torch.float64)
print("TORCH_OK", float((x * x).sum()))
PY
