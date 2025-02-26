#!/bin/bash

bash sbatch_scripts/bit_sequence/shortparse.sh
bash sbatch_scripts/bit_sequence/sac.sh
bash sbatch_scripts/bit_sequence/random_sampler.sh
bash sbatch_scripts/bit_sequence/maxent.sh
bash sbatch_scripts/bit_sequence/gfn.sh
bash sbatch_scripts/bit_sequence/oc.sh
#bash sbatch_scripts/fractalgrid/a2c.sh
#bash sbatch_scripts/fractalgrid/gfn.sh
#bash sbatch_scripts/fractalgrid/random_sampler.sh
#bash sbatch_scripts/fractalgrid/sac.sh
#bash sbatch_scripts/fractalgrid/oc.sh
#bash sbatch_scripts/rna_binding/a2c.sh
#bash sbatch_scripts/rna_binding/gfn_loss.sh
#bash sbatch_scripts/rna_binding/gfn.sh
#bash sbatch_scripts/rna_binding/maxent.sh
#bash sbatch_scripts/rna_binding/random_sampler.sh
#bash sbatch_scripts/rna_binding/sac.sh
#bash sbatch_scripts/rna_binding/shortparse.sh
#bash sbatch_scripts/rna_binding/oc.sh

