#pragma once
/**
 * Author: Turner J Robbins
 **/
#include <vector>
#include <armadillo>
#include "layer.h"

class Network
{
	std::vector<Layer> *m_layers;
	std::vector<arma::mat> *m_weights;
public:
	void addLayer();
	void insertLayer();
	void computeOutput();
};
