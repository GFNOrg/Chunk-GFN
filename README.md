chunk-gfn
---------

**Installation**

This project requires `python>=3.10`. To install, we recommend first setting up
a virtual environment of your choice, and then pip installing this package:

`pip install -e .`

**Running Experiments**

Experiment runs can be found in `sbatch_scripts/`. Runs are run via `main.py` and all
options are handled by `hydra`. See below for an example.

```bash
python main.py seed=42 data=bit_sequence gfn=tb_gfn trainer.max_epochs=1000 data.max_len=128 gfn.replay_buffer.cutoff_distance=25 gfn.reward_temperature=0.3333 logger.wandb.name="prioritized-len-128"
```


**Datasets**

To make some datasets available, make sure to add this to your environment.

```bash
#!/bin/bash
export CHUNKGFN_DATA="/path/to/code/chunk-gfn/data"
```

to download those datasets, look in
`/path/to/code/chunk-gfn/data/${dataset}/download.sh`.

**Logs**

The logging directory is determined in `configs/paths/default.yaml` it is by default
`log_dir: ${oc.env:PROJECT_DIR}/logs/` and could be changed if to any location in
your environment if desired.

When using `SLURM`, the system will automatically define the following environment variables and our code expects them to be defined. When not using slurm, `SLURM_JOB_ID` and `SLURM_JOB_NAME` will be automatically generated. This will determine the log directory.
