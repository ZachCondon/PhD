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
srun -N1 -n36 mcnp6 i=PNS_1e-9MeV o=out_PNS_1e-9MeV runtpe=r_PNS_1e-9MeV &
srun -N1 -n36 mcnp6 i=PNS_158e-9MeV o=out_PNS_158e-9MeV runtpe=r_PNS_158e-9MeV &
srun -N1 -n36 mcnp6 i=PNS_251e-9MeV o=out_PNS_251e-9MeV runtpe=r_PNS_251e-9MeV &
srun -N1 -n36 mcnp6 i=PNS_398e-9MeV o=out_PNS_398e-9MeV runtpe=r_PNS_398e-9MeV &
srun -N1 -n36 mcnp6 i=PNS_631e-9MeV o=out_PNS_631e-9MeV runtpe=r_PNS_631e-9MeV &
srun -N1 -n36 mcnp6 i=PNS_1e-8MeV o=out_PNS_1e-8MeV runtpe=r_PNS_1e-8MeV &
srun -N1 -n36 mcnp6 i=PNS_158e-8MeV o=out_PNS_158e-8MeV runtpe=r_PNS_158e-8MeV &
srun -N1 -n36 mcnp6 i=PNS_251e-8MeV o=out_PNS_251e-8MeV runtpe=r_PNS_251e-8MeV &
srun -N1 -n36 mcnp6 i=PNS_398e-8MeV o=out_PNS_398e-8MeV runtpe=r_PNS_398e-8MeV &
srun -N1 -n36 mcnp6 i=PNS_631e-8MeV o=out_PNS_631e-8MeV runtpe=r_PNS_631e-8MeV &
srun -N1 -n36 mcnp6 i=PNS_1e-7MeV o=out_PNS_1e-7MeV runtpe=r_PNS_1e-7MeV &
srun -N1 -n36 mcnp6 i=PNS_158e-7MeV o=out_PNS_158e-7MeV runtpe=r_PNS_158e-7MeV &
srun -N1 -n36 mcnp6 i=PNS_251e-7MeV o=out_PNS_251e-7MeV runtpe=r_PNS_251e-7MeV &
srun -N1 -n36 mcnp6 i=PNS_398e-7MeV o=out_PNS_398e-7MeV runtpe=r_PNS_398e-7MeV &
srun -N1 -n36 mcnp6 i=PNS_631e-7MeV o=out_PNS_631e-7MeV runtpe=r_PNS_631e-7MeV &
srun -N1 -n36 mcnp6 i=PNS_1e-6MeV o=out_PNS_1e-6MeV runtpe=r_PNS_1e-6MeV &
srun -N1 -n36 mcnp6 i=PNS_158e-6MeV o=out_PNS_158e-6MeV runtpe=r_PNS_158e-6MeV &
srun -N1 -n36 mcnp6 i=PNS_251e-6MeV o=out_PNS_251e-6MeV runtpe=r_PNS_251e-6MeV &
srun -N1 -n36 mcnp6 i=PNS_398e-6MeV o=out_PNS_398e-6MeV runtpe=r_PNS_398e-6MeV &
srun -N1 -n36 mcnp6 i=PNS_631e-6MeV o=out_PNS_631e-6MeV runtpe=r_PNS_631e-6MeV &
srun -N1 -n36 mcnp6 i=PNS_1e-5MeV o=out_PNS_1e-5MeV runtpe=r_PNS_1e-5MeV &
srun -N1 -n36 mcnp6 i=PNS_158e-5MeV o=out_PNS_158e-5MeV runtpe=r_PNS_158e-5MeV &
srun -N1 -n36 mcnp6 i=PNS_251e-5MeV o=out_PNS_251e-5MeV runtpe=r_PNS_251e-5MeV &
srun -N1 -n36 mcnp6 i=PNS_398e-5MeV o=out_PNS_398e-5MeV runtpe=r_PNS_398e-5MeV &
srun -N1 -n36 mcnp6 i=PNS_631e-5MeV o=out_PNS_631e-5MeV runtpe=r_PNS_631e-5MeV &
srun -N1 -n36 mcnp6 i=PNS_1e-4MeV o=out_PNS_1e-4MeV runtpe=r_PNS_1e-4MeV &
srun -N1 -n36 mcnp6 i=PNS_158e-4MeV o=out_PNS_158e-4MeV runtpe=r_PNS_158e-4MeV &
srun -N1 -n36 mcnp6 i=PNS_251e-4MeV o=out_PNS_251e-4MeV runtpe=r_PNS_251e-4MeV &
srun -N1 -n36 mcnp6 i=PNS_398e-4MeV o=out_PNS_398e-4MeV runtpe=r_PNS_398e-4MeV &
srun -N1 -n36 mcnp6 i=PNS_631e-4MeV o=out_PNS_631e-4MeV runtpe=r_PNS_631e-4MeV &
srun -N1 -n36 mcnp6 i=PNS_1e-3MeV o=out_PNS_1e-3MeV runtpe=r_PNS_1e-3MeV &
srun -N1 -n36 mcnp6 i=PNS_158e-3MeV o=out_PNS_158e-3MeV runtpe=r_PNS_158e-3MeV &
srun -N1 -n36 mcnp6 i=PNS_251e-3MeV o=out_PNS_251e-3MeV runtpe=r_PNS_251e-3MeV &
srun -N1 -n36 mcnp6 i=PNS_398e-3MeV o=out_PNS_398e-3MeV runtpe=r_PNS_398e-3MeV &
srun -N1 -n36 mcnp6 i=PNS_631e-3MeV o=out_PNS_631e-3MeV runtpe=r_PNS_631e-3MeV &
srun -N1 -n36 mcnp6 i=PNS_1e-2MeV o=out_PNS_1e-2MeV runtpe=r_PNS_1e-2MeV &
srun -N1 -n36 mcnp6 i=PNS_158e-2MeV o=out_PNS_158e-2MeV runtpe=r_PNS_158e-2MeV &
srun -N1 -n36 mcnp6 i=PNS_251e-2MeV o=out_PNS_251e-2MeV runtpe=r_PNS_251e-2MeV &
srun -N1 -n36 mcnp6 i=PNS_398e-2MeV o=out_PNS_398e-2MeV runtpe=r_PNS_398e-2MeV &
srun -N1 -n36 mcnp6 i=PNS_631e-2MeV o=out_PNS_631e-2MeV runtpe=r_PNS_631e-2MeV &
srun -N1 -n36 mcnp6 i=PNS_1e-1MeV o=out_PNS_1e-1MeV runtpe=r_PNS_1e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_126e-1MeV o=out_PNS_126e-1MeV runtpe=r_PNS_126e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_158e-1MeV o=out_PNS_158e-1MeV runtpe=r_PNS_158e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_2e-1MeV o=out_PNS_2e-1MeV runtpe=r_PNS_2e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_251e-1MeV o=out_PNS_251e-1MeV runtpe=r_PNS_251e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_316e-1MeV o=out_PNS_316e-1MeV runtpe=r_PNS_316e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_398e-1MeV o=out_PNS_398e-1MeV runtpe=r_PNS_398e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_501e-1MeV o=out_PNS_501e-1MeV runtpe=r_PNS_501e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_631e-1MeV o=out_PNS_631e-1MeV runtpe=r_PNS_631e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_794e-1MeV o=out_PNS_794e-1MeV runtpe=r_PNS_794e-1MeV &
srun -N1 -n36 mcnp6 i=PNS_1e0MeV o=out_PNS_1e0MeV runtpe=r_PNS_1e0MeV &
srun -N1 -n36 mcnp6 i=PNS_112e0MeV o=out_PNS_112e0MeV runtpe=r_PNS_112e0MeV &
srun -N1 -n36 mcnp6 i=PNS_126e0MeV o=out_PNS_126e0MeV runtpe=r_PNS_126e0MeV &
srun -N1 -n36 mcnp6 i=PNS_141e0MeV o=out_PNS_141e0MeV runtpe=r_PNS_141e0MeV &
srun -N1 -n36 mcnp6 i=PNS_158e0MeV o=out_PNS_158e0MeV runtpe=r_PNS_158e0MeV &
srun -N1 -n36 mcnp6 i=PNS_178e0MeV o=out_PNS_178e0MeV runtpe=r_PNS_178e0MeV &
srun -N1 -n36 mcnp6 i=PNS_2e0MeV o=out_PNS_2e0MeV runtpe=r_PNS_2e0MeV &
srun -N1 -n36 mcnp6 i=PNS_224e0MeV o=out_PNS_224e0MeV runtpe=r_PNS_224e0MeV &
srun -N1 -n36 mcnp6 i=PNS_251e0MeV o=out_PNS_251e0MeV runtpe=r_PNS_251e0MeV &
srun -N1 -n36 mcnp6 i=PNS_282e0MeV o=out_PNS_282e0MeV runtpe=r_PNS_282e0MeV &
srun -N1 -n36 mcnp6 i=PNS_316e0MeV o=out_PNS_316e0MeV runtpe=r_PNS_316e0MeV &
srun -N1 -n36 mcnp6 i=PNS_355e0MeV o=out_PNS_355e0MeV runtpe=r_PNS_355e0MeV &
srun -N1 -n36 mcnp6 i=PNS_398e0MeV o=out_PNS_398e0MeV runtpe=r_PNS_398e0MeV &
srun -N1 -n36 mcnp6 i=PNS_447e0MeV o=out_PNS_447e0MeV runtpe=r_PNS_447e0MeV &
srun -N1 -n36 mcnp6 i=PNS_501e0MeV o=out_PNS_501e0MeV runtpe=r_PNS_501e0MeV &
srun -N1 -n36 mcnp6 i=PNS_562e0MeV o=out_PNS_562e0MeV runtpe=r_PNS_562e0MeV &
srun -N1 -n36 mcnp6 i=PNS_631e0MeV o=out_PNS_631e0MeV runtpe=r_PNS_631e0MeV &
srun -N1 -n36 mcnp6 i=PNS_708e0MeV o=out_PNS_708e0MeV runtpe=r_PNS_708e0MeV &
srun -N1 -n36 mcnp6 i=PNS_794e0MeV o=out_PNS_794e0MeV runtpe=r_PNS_794e0MeV &
srun -N1 -n36 mcnp6 i=PNS_891e0MeV o=out_PNS_891e0MeV runtpe=r_PNS_891e0MeV &
srun -N1 -n36 mcnp6 i=PNS_1e1MeV o=out_PNS_1e1MeV runtpe=r_PNS_1e1MeV &
srun -N1 -n36 mcnp6 i=PNS_112e1MeV o=out_PNS_112e1MeV runtpe=r_PNS_112e1MeV &
srun -N1 -n36 mcnp6 i=PNS_126e1MeV o=out_PNS_126e1MeV runtpe=r_PNS_126e1MeV &
srun -N1 -n36 mcnp6 i=PNS_141e1MeV o=out_PNS_141e1MeV runtpe=r_PNS_141e1MeV &
srun -N1 -n36 mcnp6 i=PNS_158e1MeV o=out_PNS_158e1MeV runtpe=r_PNS_158e1MeV &
srun -N1 -n36 mcnp6 i=PNS_178e1MeV o=out_PNS_178e1MeV runtpe=r_PNS_178e1MeV &
srun -N1 -n36 mcnp6 i=PNS_2e1MeV o=out_PNS_2e1MeV runtpe=r_PNS_2e1MeV &
srun -N1 -n36 mcnp6 i=PNS_251e1MeV o=out_PNS_251e1MeV runtpe=r_PNS_251e1MeV &
srun -N1 -n36 mcnp6 i=PNS_316e1MeV o=out_PNS_316e1MeV runtpe=r_PNS_316e1MeV &
srun -N1 -n36 mcnp6 i=PNS_398e1MeV o=out_PNS_398e1MeV runtpe=r_PNS_398e1MeV &
srun -N1 -n36 mcnp6 i=PNS_501e1MeV o=out_PNS_501e1MeV runtpe=r_PNS_501e1MeV &
srun -N1 -n36 mcnp6 i=PNS_631e1MeV o=out_PNS_631e1MeV runtpe=r_PNS_631e1MeV &
srun -N1 -n36 mcnp6 i=PNS_794e1MeV o=out_PNS_794e1MeV runtpe=r_PNS_794e1MeV &
srun -N1 -n36 mcnp6 i=PNS_1e2MeV o=out_PNS_1e2MeV runtpe=r_PNS_1e2MeV &

wait
echo 'Done'