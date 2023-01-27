#!/bin/csh
#SBATCH -N 84
#SBATCH -J Condon_PNS2022-08-30_1358
#SBATCH -t 23:30:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -A cbronze
#SBATCH -D /g/g20/condon3/PNS/2022-08-30_1358

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
srun -N1 -n36 mcnp6 c r=r_PNS_1e-9MeV o=out_PNS_1e-9MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e-9MeV o=out_PNS_158e-9MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e-9MeV o=out_PNS_251e-9MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e-9MeV o=out_PNS_398e-9MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e-9MeV o=out_PNS_631e-9MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e-8MeV o=out_PNS_1e-8MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e-8MeV o=out_PNS_158e-8MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e-8MeV o=out_PNS_251e-8MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e-8MeV o=out_PNS_398e-8MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e-8MeV o=out_PNS_631e-8MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e-7MeV o=out_PNS_1e-7MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e-7MeV o=out_PNS_158e-7MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e-7MeV o=out_PNS_251e-7MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e-7MeV o=out_PNS_398e-7MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e-7MeV o=out_PNS_631e-7MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e-6MeV o=out_PNS_1e-6MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e-6MeV o=out_PNS_158e-6MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e-6MeV o=out_PNS_251e-6MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e-6MeV o=out_PNS_398e-6MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e-6MeV o=out_PNS_631e-6MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e-5MeV o=out_PNS_1e-5MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e-5MeV o=out_PNS_158e-5MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e-5MeV o=out_PNS_251e-5MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e-5MeV o=out_PNS_398e-5MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e-5MeV o=out_PNS_631e-5MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e-4MeV o=out_PNS_1e-4MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e-4MeV o=out_PNS_158e-4MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e-4MeV o=out_PNS_251e-4MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e-4MeV o=out_PNS_398e-4MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e-4MeV o=out_PNS_631e-4MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e-3MeV o=out_PNS_1e-3MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e-3MeV o=out_PNS_158e-3MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e-3MeV o=out_PNS_251e-3MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e-3MeV o=out_PNS_398e-3MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e-3MeV o=out_PNS_631e-3MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e-2MeV o=out_PNS_1e-2MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e-2MeV o=out_PNS_158e-2MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e-2MeV o=out_PNS_251e-2MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e-2MeV o=out_PNS_398e-2MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e-2MeV o=out_PNS_631e-2MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e-1MeV o=out_PNS_1e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_126e-1MeV o=out_PNS_126e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e-1MeV o=out_PNS_158e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_2e-1MeV o=out_PNS_2e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e-1MeV o=out_PNS_251e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_316e-1MeV o=out_PNS_316e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e-1MeV o=out_PNS_398e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_501e-1MeV o=out_PNS_501e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e-1MeV o=out_PNS_631e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_794e-1MeV o=out_PNS_794e-1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e0MeV o=out_PNS_1e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_112e0MeV o=out_PNS_112e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_126e0MeV o=out_PNS_126e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_141e0MeV o=out_PNS_141e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e0MeV o=out_PNS_158e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_178e0MeV o=out_PNS_178e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_2e0MeV o=out_PNS_2e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_224e0MeV o=out_PNS_224e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e0MeV o=out_PNS_251e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_282e0MeV o=out_PNS_282e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_316e0MeV o=out_PNS_316e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_355e0MeV o=out_PNS_355e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e0MeV o=out_PNS_398e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_447e0MeV o=out_PNS_447e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_501e0MeV o=out_PNS_501e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_562e0MeV o=out_PNS_562e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e0MeV o=out_PNS_631e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_708e0MeV o=out_PNS_708e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_794e0MeV o=out_PNS_794e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_891e0MeV o=out_PNS_891e0MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e1MeV o=out_PNS_1e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_112e1MeV o=out_PNS_112e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_126e1MeV o=out_PNS_126e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_141e1MeV o=out_PNS_141e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_158e1MeV o=out_PNS_158e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_178e1MeV o=out_PNS_178e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_2e1MeV o=out_PNS_2e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_251e1MeV o=out_PNS_251e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_316e1MeV o=out_PNS_316e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_398e1MeV o=out_PNS_398e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_501e1MeV o=out_PNS_501e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_631e1MeV o=out_PNS_631e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_794e1MeV o=out_PNS_794e1MeV_cont &
srun -N1 -n36 mcnp6 c r=r_PNS_1e2MeV o=out_PNS_1e2MeV_cont &

wait
echo 'Done'