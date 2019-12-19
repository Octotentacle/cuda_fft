#include <bits/stdc++.h>
#include <unistd.h>
#include <cuda.h>

template <typename Iter>
void cooley_tukey(Iter first, Iter last) {
    auto size = last - first;
    if (size >= 2) {
        auto temp = std::vector<std::complex<double>>(size / 2);
        for (int i = 0; i < size / 2; ++i) {
            temp[i] = first[i * 2 + 1];
            first[i] = first[i * 2];
        }
        for (int i = 0; i < size / 2; ++i) {
            first[i + size / 2] = temp[i];
        }
        auto split = first + size / 2;
        cooley_tukey(first, split);
        cooley_tukey(split, last);
        for (int k = 0; k < size / 2; ++k) {
            auto w = std::exp(std::complex<double>(0, -2.0 * M_PI * k / size));
            auto& bottom = first[k];
            auto& top = first[k + size / 2];
            top = bottom - w * top;
            bottom -= top - bottom;
        }
    }
}

void mod_cooley_tukey(std::complex<double>* data, size_t left, size_t right) {
    auto size = right - left;
    if (size >= 2) {
        auto temp = std::vector<std::complex<double>>(size / 2);
        for (size_t i = 0; i < size / 2; ++i) {
            temp[i] = data[left + i * 2 + 1];
            data[left + i] = data[left + i * 2];
        }
        for (size_t i = 0; i < size / 2; ++i) {
            data[left + i + size / 2] = temp[i];
        }
        auto split = left + size / 2;
        mod_cooley_tukey(data, left, split);
        mod_cooley_tukey(data, split, right);
        for (size_t k = 0; k < size / 2; ++k) {
            auto w = std::exp(std::complex<double>(0, -2. * M_PI * k / size));
            auto& bottom = data[k + left];
            auto& top = data[k + size / 2 + left];
            top = bottom - w * top;
            bottom -= top - bottom;
        }
    }
}

template <typename T>
void bit_reversal(std::vector<T>& data, size_t left, size_t right) {
    auto size = right - left;
    if (size >= 2) {
        auto temp = std::vector<T>(size / 2);
        for (size_t i = 0; i < size / 2; ++i) {
            temp[i] = data[left + i * 2 + 1];
            data[left + i] = data[left + i * 2];
        }
        for (size_t i = 0; i < size / 2; ++i) {
            data[left + i + size / 2] = temp[i];
        }
        auto split = left + size / 2;
        bit_reversal(data, left, split);
        bit_reversal(data, split, right);
    }
}


void iterative_cooley_tukey(std::vector<std::complex<double>>& data, size_t left, size_t right) {
    bit_reversal(data, left, right);
    for (size_t iter = 2; iter <= right - left; iter <<= 1u) {
        for (size_t base_pos = 0; base_pos < right; base_pos += iter) {
            for (size_t k = 0; k < iter / 2; ++k) {
                auto w = std::exp(std::complex<double>(0, -2. * M_PI * k / iter));
                data[k + iter / 2 + base_pos] = data[k + base_pos] - w * data[k + iter / 2 + base_pos];
                data[k + base_pos] -= data[k + iter / 2 + base_pos] - data[k + base_pos];
            }
        }
    }
}

static __device__ __host__ inline size_t rev_num(size_t num, size_t deg) {
    size_t reverse_num = 0;
    int i;
    for (i = 0; i < deg; i++) {
        if((num & (1 << i))) {
            reverse_num |= 1 << ((deg - 1) - i);
        }
    }
    return reverse_num;
}


__global__ void bit_reversed_order(double2* __restrict__ input, double2* output, size_t deg) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < (1 << deg); i += stride) {
        size_t new_ind = rev_num(i, deg);
        output[new_ind] = input[i];
    }
}

static __device__ __host__ inline double2 CplxSub(double2 a, double2 b) {
    double2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

static __device__ __host__ inline double2 CplxMul(double2 a, double2 b) {
    double2 c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

__global__ void new_cooley_tukey_iteration(double2* data, size_t len) {
    size_t pos = blockDim.x * blockIdx.x + threadIdx.x;
    size_t k = pos % len;
    size_t base_pos = pos - k;
    if (k < len / 2) {
        double2 w;
        double phi = -2. * M_PI * k / len;
        w.x = cos(phi);
        w.y = sin(phi);
        data[k + len / 2 + base_pos] = CplxSub(data[k + base_pos], CplxMul(w, data[k + len / 2 + base_pos]));
        data[k + base_pos] = CplxSub(data[k + base_pos], CplxSub(data[k + len / 2 + base_pos], data[k + base_pos]));
    }
}

void cuda_fft(std::vector<std::complex<double>>& data) {
    double2* inp_data, *bit_rev_data;
    cudaMallocManaged((void**)&inp_data, data.size() * sizeof(double2));
    cudaMallocManaged((void**)&bit_rev_data, data.size() * sizeof(double2));
    for (int i = 0; i < data.size(); ++i) {
        inp_data[i].x = data[i].real();
        inp_data[i].y = data[i].imag();
    }
    size_t deg = 0, sz = data.size();
    while (sz != 1) {
        ++deg;
        sz >>= 1;
    }
    bit_reversed_order<<<512, 256>>>(inp_data, bit_rev_data, deg);
    cudaDeviceSynchronize();
    cudaFree(inp_data);
    for (size_t blockSize = 2; blockSize <= data.size(); blockSize <<= 1u) {
        size_t threads = std::min(static_cast<size_t>(1024), blockSize);
        new_cooley_tukey_iteration<<<data.size() / threads, threads>>>(bit_rev_data, blockSize);
        cudaDeviceSynchronize();
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i].real(bit_rev_data[i].x);
        data[i].imag(bit_rev_data[i].y);
    }
    cudaFree(bit_rev_data);
}


template <typename Iter>
void inversed_fft(Iter first, Iter last) {
    cooley_tukey(first, last);
    auto it = first;
    auto size = last - first;
    while (it != last) {
        *it /= size;
        ++it;
    }
    std::reverse(first + 1, last);
}

const int kRuns = 1;

void test_speed_and_correctness(std::string filename) {
    std::ifstream fin(filename);
    std::vector<std::complex<double>> data;
    while (!fin.eof()) {
        double real, imag;
        if (fin >> real >> imag) {
            data.emplace_back(real, imag);
        }
    }
    auto data1 = data;
    for (int k = 0; k < kRuns; ++k) {
        {
            auto start = std::chrono::high_resolution_clock::now();
            iterative_cooley_tukey(data, 0, data.size());
            auto finish = std::chrono::high_resolution_clock::now();
            auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
            std::cout << milliseconds.count() << ' ';
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            cuda_fft(data1);
            auto finish = std::chrono::high_resolution_clock::now();
            auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
            std::cout << milliseconds.count() << ' ';
        }
        double max_diff = 0, min_diff = 1E9;
        for (size_t i = 0; i < data.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(data[i] - data1[i]));
            min_diff = std::min(min_diff, std::abs(data[i] - data1[i]));
        }
        std::cout << max_diff << ' ' << min_diff << '\n';
    }
}

int main(int argc, char* argv[]) {
    test_speed_and_correctness(argv[1]);
    return 0;
}