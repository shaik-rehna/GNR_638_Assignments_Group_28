cmake_minimum_required(VERSION 3.10)
project(my_framework)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(my_framework
    core/tensor.cpp
    core/autograd.cpp
    layers/linear.cpp
    layers/conv2d.cpp
    layers/maxpool2d.cpp
    ops/activation.cpp
    ops/flatten.cpp
    loss/cross_entropy.cpp
    optim/sgd.cpp
    bindings.cpp)