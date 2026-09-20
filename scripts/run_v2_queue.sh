#!/bin/bash
# 以最多 N 个并发进程运行 v2 配置。
N=${1:-6}
RUN="env -u LD_LIBRARY_PATH OMP_NUM_THREADS=3 PYTHONPATH=src"; PY=${PY:-python}
mkdir -p results/v2_logs
for f in $(cat ${JOBS:-configs/v2/jobs.txt}); do
  while [ $(pgrep -fc run_benchmark_v2.py) -ge $N ]; do sleep 20; done
  name=$(basename $f .yaml)
  $RUN nohup $PY scripts/run_benchmark_v2.py --config $f > results/v2_logs/$name.log 2>&1 &
  sleep 2
done
wait
echo ALL DONE
