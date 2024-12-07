#include "Swish.hpp"
#include <vector>
#include <cstdlib>
#include <bits/stdc++.h>

std::vector<double> Swish::Swish(const std::vector<double>& arr) {
	std::size_t size = arr.size();
	std::vector<double> output(size);
	for (std::size_t i = 0; i < size; ++i) {
		double num = arr[i];
		output[i] = num / (1 + std::exp(-num));
	}
	return output;
}