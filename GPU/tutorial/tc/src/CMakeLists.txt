add_library(src
        # graph folder
        graph/graph.h
        graph/graph.cpp
        tc/tc.cu tc/tc.cuh)

# Link libraries (if needed)
target_link_libraries(src PUBLIC argparse)

# Include directories
target_include_directories(src PUBLIC
        ${argparse_SOURCE_DIR}
        )