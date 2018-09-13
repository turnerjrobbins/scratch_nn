#pragma once
/**
 * Author: Turner J Robbins
 **/
#include <vector>
#include <armadillo>
#include "layer.h"

class Network
{
	std::vector<Layer *> m_layers;
	std::vector<arma::mat *> m_weights;
	void updateMatrices();
public:
	void addLayer(Layer *newLayer);
	void insertLayer(Layer *newLayer);
	double computeOutput(arma::mat input, int expectedOutput);
	const std::vector<arma::mat *>* getWeights() {return &m_weights;}
};
