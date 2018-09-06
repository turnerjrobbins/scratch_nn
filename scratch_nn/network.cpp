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
