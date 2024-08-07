#include <iostream>
#include <argparse/argparse.hpp>

#include "src/graph/graph.h"
#include "src/tc/tc.cuh"


int main(int argc, char* argv[]) {

    argparse::ArgumentParser parser("triangle counting");

    parser.add_argument("--gpu")
            .help("GPU Device ID (must be a positive integer)")
            .default_value(0)
            .action([](const std::string &value) { return std::stoi(value); });

    parser.add_argument("--graph")
            .help("Graph file path")
            .default_value("/")
            .action([](const std::string &value) { return value; });

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cout << parser << std::endl;
        exit(EXIT_FAILURE);
    }

    auto device_count = 0;
    auto device_id = 0;

    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "error: no gpu device found" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (parser.is_used("--gpu")) {
        device_id = parser.get<int>("--gpu");
        if (device_id >= device_count) {
            std::cerr << "error: invalid gpu device id" << std::endl;
            exit(EXIT_FAILURE);
        }
        cudaSetDevice(device_id);
    }

    if (parser.is_used("--graph")) {
        auto dataset = parser.get<std::string>("--graph");
        auto g = Graph(dataset);

        // then aglorithm
        tc(&g);
    }

}
