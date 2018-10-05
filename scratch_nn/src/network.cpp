#include <cmath>
#include "network.h"
void Network::addLayer(Layer *newLayer)
{
	this->m_layers.push_back(newLayer);
	this->updateMatrices();
}
void Network::updateMatrices()
{
	using namespace arma;
	if(this->m_weights.size() != this->m_layers.size() -1) 
	{
		int r = this->m_layers[this->m_layers.size()-1]->getLayerSize();
		int c = this->m_layers[this->m_layers.size()-2]->getLayerSize();
		this->m_weights.push_back(new mat(r,c,fill::randu));
	}
}
double Network::computeOutput(arma::mat input, int expectedOutput) 
{
//	std::cout << "Input: " << input << std::endl;
	arma::mat o;
	input.for_each([] (arma::mat::elem_type &x) {x = x / (1+abs(x));});
//	std::cout << "Activated: " << input << std::endl;
	arma::mat *m = this->m_weights[0];
//	std::cout << "Weights: " << *m << std::endl;
	input = input*(*m);
//	std::cout << "Output: " << input << std::endl;
	input.for_each([] (arma::mat::elem_type &x) {x = x / (1+abs(x));});
//	std::cout << "Activated: " << input << std::endl;
	m = this->m_weights[1];
//	std::cout << "Weights: " << *m << std::endl;
	input = input % (*m);
//	std::cout << "Output: " << input << std::endl;
	double output = accu(input);
	output = output / (1+abs(output));
//	std::cout << "Final Output: " << output << std::endl;
	return output;
}
