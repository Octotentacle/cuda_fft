import matplotlib.pyplot as plt
import math
import os

begin_deg = 10
end_deg = 26

def run_tests():
    for deg in range(begin_deg, end_deg):
        os.system(f"./compare_fft ./inputs/input_{deg}.txt > ./outputs/output_{deg}.txt")
        print(f"{1<<deg} passed!")


def process_outputs_1():
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 10)
    cooley_tukey_time = []
    cuda_time = []
    for deg in range(begin_deg, end_deg):
        with open(f"./outputs/output_{deg}.txt") as inp:
            line = inp.readline()
            cooley_tukey_time.append(int(line.split(' ')[0]) / 1000)
            cuda_time.append(int(line.split(' ')[1]) / 1000)
    ax.plot(list(range(begin_deg, end_deg)), cooley_tukey_time, label='2-radix decimation-in-time Cooley-Tukey iterative')
    ax.plot(list(range(begin_deg, end_deg)), cuda_time, label='CUDA implementation')
    ax.scatter(list(range(begin_deg, end_deg)), cooley_tukey_time)
    ax.scatter(list(range(begin_deg, end_deg)), cuda_time)
    ax.legend()
    ax.set_xlabel("$log_2(input\_size)$")
    ax.set_ylabel("time, seconds")
    plt.savefig('plot1.png')


def process_outputs_2():
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 10)
    cooley_tukey_time = []
    cuda_time = []
    for deg in range(begin_deg, end_deg):
        with open(f"./outputs/output_{deg}.txt") as inp:
            line = inp.readline()
            cooley_tukey_time.append(math.log2(int(line.split(' ')[0]) / 1000))
            cuda_time.append(math.log2(int(line.split(' ')[1]) / 1000))
    ax.plot(list(range(begin_deg, end_deg)), cooley_tukey_time, label='2-radix decimation-in-time Cooley-Tukey iterative')
    ax.plot(list(range(begin_deg, end_deg)), cuda_time, label='CUDA implementation')
    ax.scatter(list(range(begin_deg, end_deg)), cooley_tukey_time)
    ax.scatter(list(range(begin_deg, end_deg)), cuda_time)
    ax.legend()
    ax.set_xlabel("$log_2(input\_size)$")
    ax.set_ylabel("$log_2(time)$, seconds")
    plt.savefig('plot2.png')


def process_outputs_3():
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 10)
    rel = []
    for deg in range(begin_deg, end_deg):
        with open(f"./outputs/output_{deg}.txt") as inp:
            line = inp.readline()
            rel.append(int(line.split(' ')[0]) / 1000)
            rel[-1] /= int(line.split(' ')[1]) / 1000
    ax.plot(list(range(begin_deg, end_deg)), rel, label='alg1/alg2')
    ax.scatter(list(range(begin_deg, end_deg)), rel)
    ax.legend()
    ax.set_xlabel("$log_2(input\_size)$")
    plt.savefig('plot3.png')


def process_outputs_4():
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 10)
    rel = []
    for deg in range(begin_deg, end_deg):
        with open(f"./outputs/output_{deg}.txt") as inp:
            line = inp.readline()
            rel.append(int(line.split(' ')[0]) / 1000)
            rel[-1] /= int(line.split(' ')[1]) / 1000
            rel[-1] = math.log2(rel[-1])
    ax.plot(list(range(begin_deg, end_deg)), rel, label='$log_2(alg1/alg2)$')
    ax.scatter(list(range(begin_deg, end_deg)), rel)
    ax.legend()
    ax.set_xlabel("$log_2(input\_size)$")
    plt.savefig('plot4.png')


if __name__ == "__main__":
    # run_tests()
    process_outputs_1()
    process_outputs_2()
    process_outputs_3()
    process_outputs_4()
