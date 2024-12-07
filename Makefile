CXX=g++
CXXFLAGS=-std=c++11 -O3 -march=native -Wall -I$(SRC_DIR) $(shell python3 -m pybind11 --includes)
PY_LDFLAGS=$(shell python3-config --ldflags) -lopenblas -shared -fPIC
LDFLAGS=-lopenblas
SRC_DIR=swish/src
MODULE_DIR=swish/python

swish: $(MODULE_DIR)/bindings.o $(SRC_DIR)/Swish.o
	$(CXX) $^ -o $(MODULE_DIR)/swish_binding`python3-config --extension-suffix` $(PY_LDFLAGS) $(CXXFLAGS)

$(MODULE_DIR)/bindings.o: $(MODULE_DIR)/bindings.cpp $(SRC_DIR)/Swish.hpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(SRC_DIR)/Swish.o: $(SRC_DIR)/Swish.cpp $(SRC_DIR)/Swish.hpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@