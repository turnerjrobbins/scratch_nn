#pragma once
/**
 * Author - Turner J Robbins
 **/
#include <armadillo>
#include <vector>
#include "layer.h"

class Layer 
{
	//List of neurons
	int m_size;
public:
	Layer(int layerSize)
	{
		this->m_size = layerSize;	
	}

	//Inputs:
	//    weights: matrix representing weights of inputs
	//    biases: matrix representing biases
	//Output:
	//    layer_output: matrix representing the output of every neuron in this layer
	double activate(arma::mat inputs);

	//Input:
	//    Lambda representing activation function
	void setActivationFunction();

	int getLayerSize() {return this->m_size;}
};
