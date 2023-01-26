#!/bin/csh
#SBATCH -N 1
#SBATCH -J PNS_141e1MeV
#SBATCH -t 23:30:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -A cbronze
#SBATCH -D /g/g20/condon3/PNS/2022-08-22_1047

echo '=================Job diagnostics================='
date
echo -n 'This machine is ';hostname
echo -n 'My jobid is '; echo $SLURM_JOBID
echo 'My path is:'
echo $PATH
echo 'My job info:'
squeue -j $SLURM_JOBID
echo 'Machine info'
sinfo -s

echo '=================Job Starting================='
echo 'Job_id = $SLURM_JOBID'
set echo
setenv DATAPATH /usr/gapps/MCNP_DATA/620
srun -n36 -k /usr/apps/mcnp/bin/mcnp6.mpi nam=PNS_141e1MeV mct=mctal_PNS_141e1MeV o=out_PNS_141e1MeV runtpe=r_PNS_141e1MeV

wait
echo 'Done'