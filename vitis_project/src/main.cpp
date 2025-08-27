#include <iostream>
#include <fstream>
#include <vector>

void multiply_arrays(const std::vector<int>& in1, const std::vector<int>& in2, std::vector<int>& out) {
    for (size_t i = 0; i < in1.size(); ++i) {
        out[i] = in1[i] * in2[i];
    }
}

int main() {
    std::ifstream infile("test1.txt");
    if (!infile) {
        std::cerr << "File failure";
        return 1;
    }

    int size;
    infile >> size;

    std::vector<int> in1(size), in2(size), out(size);

    for (int i = 0; i < size; ++i) infile >> in1[i];
    for (int i = 0; i < size; ++i) infile >> in2[i];

    multiply_arrays(in1, in2, out);

    std::cout << "Output:\n";
    for (int val : out) std::cout << val << " ";
    std::cout << "\n";

    return 0;
}

