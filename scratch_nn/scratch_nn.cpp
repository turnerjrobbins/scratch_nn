#include <iostream>
#include <armadillo>
#include "layer.h"
#include "network.h"

using namespace arma;

/** basic implementation of a neural network **/

int main()
{
	std::cout << "hello world\n";
	mat A = randu<mat>(4,5);
	mat B = randu<mat>(4,5);
	std::cout << A*B.t() << std::endl;
	exit(0);
	//Load training examples
	
	//Train the Network
		//Compute Output
		//Compute Gradients
		//Update parameters
		//repeat
	
	//Validate the networke
}
