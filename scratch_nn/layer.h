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
	Layer *m_nextLayer;
	Layer *m_prevLayer;
public:
	Layer()
	{
		
	}

	Layer(Layer prev, Layer next)
	{

	}

	//Inputs:
	//    weights: matrix representing weights of inputs
	//    biases: matrix representing biases
	//Output:
	//    layer_output: matrix representing the output of every neuron in this layer
	void activate();

	//Input:
	//    Lambda representing activation function
	void setActivationFunction();
};
