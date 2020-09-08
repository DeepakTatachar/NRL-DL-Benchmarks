import subprocess
import multiprocessing
import time
import os

def monitor(pid, run):
    log_file = open('log.txt', 'w')
    log_file.write(str(pid))
    while run.value:
        gpu_util_cmd = 'nvidia-smi pmon -d 2 -s um -c 1'
        io_util_cmd = 'pidstat -dl 1 2'
        gpuutil = subprocess.check_output(gpu_util_cmd, shell=True)
        log_file.write(gpuutil.decode("utf-8"))
        log_file.write('-'*50 + '\n')
        ioutil = subprocess.check_output(io_util_cmd, shell=True)
        log_file.write(ioutil.decode("utf-8"))
        log_file.write('-'*50 + '\n')
        time.sleep(0.05)
    
    log_file.close()

def run_benchmark(signal):
    benchmark = 'CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset=imagenet --arch=torch_resnet18 --epochs=0 --train_batch_size=64 --test_batch_size=64 --parallel=True'
    os.system(benchmark)
    print('Benchmark Finished')
    signal.value = 0

signal = multiprocessing.Value('i', 1)
benchmark = multiprocessing.Process(target=run_benchmark, args=[signal])
benchmark.start()
monitor = multiprocessing.Process(target=monitor, args=[benchmark.pid, signal])
monitor.start()

benchmark.join()
monitor.join()