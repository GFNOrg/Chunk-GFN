#!/bin/bash
unset SLURM_PROCID
exec "$@"