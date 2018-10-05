#include <cstdlib>
#include <iostream>
#include <armadillo>
#include "layer.h"
#include "network.h"

using namespace arma;

struct individual {
	Network *nn;
	double error;
};

int selectIndividual(std::vector<individual *> *pop, 
		std::default_random_engine *generator) {
	double dev = pop->size() / 2.5;
	std::normal_distribution<double> dist(0.0, dev);
	int ind = abs(dist(*generator));
	return ind;
}
individual * breedIndividuals(individual *a, individual *b) {
	individual *n = new individual;
	n->nn = new Network();
	n->error = 0.0;
	n->nn->addLayer(new Layer(2));
	n->nn->addLayer(new Layer(2));
	n->nn->addLayer(new Layer(1));
}
/** basic implementation of a neural network **/

int main()
{
	std::cout << "hello world\n";
	const double select_from = 0.25;
	const double mutation_rate = 0.01;
	const int pop_size = 100;
	const int batch_size = 100;

	//Generate 100 NNS
	std::vector<individual *> pop;	
	for (int i = 0; i < pop_size; i++)
	{
		individual *newInd = new individual;
		newInd->nn = new Network();
		newInd->error = 0.0;
		Network *nn = newInd->nn;
		nn->addLayer(new Layer(2));
		nn->addLayer(new Layer(2));
		nn->addLayer(new Layer(1));
		pop.push_back(newInd);
	}
	std::cout << "Population Generated." << std::endl;
	//Run NNs with samples using loss as fitness
	struct testData {
		arma::mat *input = new arma::mat(1,2);
		bool b_expected;
	} ;
	std::vector<testData *> batch;
	for(int i =0; i < batch_size; i++)
	{
		testData *newData = new testData;
		newData->input->at(0) = std::rand() % 2;//generate random true/false
		newData->input->at(1) = std::rand() % 2;//generate random true/false
		newData->b_expected = newData->input->at(0) 
			&& newData->input->at(1);
		batch.push_back(newData);
	}
	std::cout << "Test Data Created" << std::endl;	
	for(individual *p : pop) {
		for(testData *t : batch) {
			p->error += p->nn->computeOutput(*(t->input), t->b_expected);
		}
	}
	std::cout << "Evalutation Complete" << std::endl;

	std::sort(pop.begin(), pop.end(),
			[] (individual *a, individual*b) {return a->error < b->error;});

	std::cout << "Lowest error: " << pop[0]->error << std::endl;
	
	std::default_random_engine g;	
	std::vector<individual *> newPop;
	for(int i=0; i <pop_size; i++) {
		int a = selectIndividual(&pop, &g);
		int b = selectIndividual(&pop, &g);
		newPop.push_back(breedIndividuals(pop[a], pop[b]));
	}
	//Selection - select from the top quartile
	//
	//Breeding strategy - select 1st matrix from parent 1, 2nd from parent 2
	//
	//Mutate weights up or down
	
}
