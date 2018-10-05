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
    std::vector<mat *> *weights = n->nn->getWeights();
    std::vector<mat *> *a_weights = a->nn->getWeights();
    std::vector<mat *> *b_weights = b->nn->getWeights();
    weights->at(0) = a_weights->at(0);
    weights->at(1) = b_weights->at(1);
    return n;
}

void mutateIndividual(individual *ind, double mutRate) { 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0,1.0);
    
    //std::cout << "Accessing Weights" << std::endl;
    std::vector<mat *> *weights = ind->nn->getWeights(); 
    //std::cout << "Accessing Matrix" << std::endl;
    //std::cout << "Doing foreach" << std::endl;
    for(mat *mutMatx : *weights) { 
       mutMatx->for_each([&](mat::elem_type& val) {
           if(dis(gen) < mutRate) {
               val += rand() % 2 ? .1 : -.1;
               std::cout << "Pop mutated" << std::endl;
           }
       } );
    }

}
/** basic implementation of a neural network **/

int main()
{
	std::cout << "Neural Network\n";

    //Define GA constants
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
    
    train:	
    //EVALUATION
    for(individual *p : pop) {
		for(testData *t : batch) {
			p->error += std::abs(p->nn->computeOutput(*(t->input), t->b_expected));
		}
	}
	std::cout << "Evalutation Complete" << std::endl;
	std::sort(pop.begin(), pop.end(),
			[] (individual *a, individual*b) {return a->error < b->error;});

	std::cout << "Lowest error: " << pop[0]->error << std::endl;
    if(std::abs(pop[0]->error) < 0.01) {
        exit(0);
    }	
	
    //SELECTION AND BREEDING
	std::default_random_engine g;	
	std::vector<individual *> newPop;
	for(int i=0; i <pop_size; i++) {
		int a = selectIndividual(&pop, &g);
		int b = selectIndividual(&pop, &g);
		newPop.push_back(breedIndividuals(pop[a], pop[b]));
	}
    pop = newPop;
    for(auto *ind : pop) {
        mutateIndividual(ind, mutation_rate );
    } 
    std::cout << "Mutation Complete" << std::endl;
    //Mutate weights up or down
    goto train;
}
