#include <iostream>
#include "NeuralNetwork.hpp"
#include <fstream>
#include "Misc.hpp"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

int main()
{
	fstream configFile("pconfig.json");

	string str((istreambuf_iterator<char>(configFile)),
                istreambuf_iterator<char>());

	auto config = json::parse(str);

    string predectionFile  = config["predectionData"];
    string weightFile    = config["weightsFile"];
    vector<int> topology = config["topology"];
    int hiddenactivationtype = config["hiddenactivationtype"];
    int outputactivationtype = config["outputactivationtype"];
    int costfunctiontype     = config["costfunctiontype"];
    
    vector<vector<double> > predectionData =
    utils::Misc::fetchData(predectionFile);

	NeuralNetwork* nn = new NeuralNetwork(
										topology,
										hiddenactivationtype,
										outputactivationtype,
										costfunctiontype
										);

	nn->loadWeights(weightFile);

	int right_prediction = 0;

	//cout<<predectionData.size();

	for(int k = 0; k < predectionData.size(); k++){

	vector<double> input;

	input = predectionData[k];

	input.erase(input.begin() + 0);

	nn->setCurrentInput(input);

	nn->feedForward();

	int result = nn->layers[nn->layers.size()-1]->matrixifyActivatedVals()->getmax();

	nn->layers[nn->layers.size()-1]->matrixifyActivatedVals()->printToConsole();

	//cout<<"\n"<<predectionData[k][0]<<" : "<<result<<endl;
	
	if(result == predectionData[k][0])
		right_prediction++;

	}

	cout<<right_prediction<<" / "<<predectionData.size();
	topology.clear();
	return 0;
}