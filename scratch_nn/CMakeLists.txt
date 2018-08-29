cmake_minimum_required (VERSION 2.6)
project (scratch_nn)
add_executable (scratch_nn scratch_nn.cpp)
target_link_libraries(scratch_nn armadillo)
