#include <iostream>
#include "NeuralNetwork.hpp"
#include <fstream>
#include "Misc.hpp"
#include "json.hpp"

using namespace std;
using json = nlohmann::json;

int main()
{
	ofstream fout("error.out");
	fstream configFile("config.json");

	string str((istreambuf_iterator<char>(configFile)),
                istreambuf_iterator<char>());

	auto config = json::parse(str);

    string trainingFile  = config["trainingData"];
    string weightFile    = config["weightsFile"];
    vector<int> topology = config["topology"];
    int epoch            = config["epoch"];
    double bias          = config["bias"];
    double momentum      = config["momentum"];
    double learningRate  = config["learningRate"];

	vector<vector<double> > trainingData =
    utils::Misc::fetchData(trainingFile);

	NeuralNetwork* nn = new NeuralNetwork(topology);

	nn->loadWeights(weightFile);

	cout<<"click Enter to let me learn...\n";

	getchar();

	for(int i = 0; i < epoch; i++){

	double ll = double(i);

	double per = (ll+1) / epoch;

	double cal = per * 100;

	for(int j = 0; j < 10000; j++){

	vector<double> input;

	vector<double> target;

	input = trainingData[j];

    input.erase(input.begin() + 0);

    double temp = trainingData[j][0];

    for(int h = 0; h < 10; h++)
    {
    	if(temp == h)
    		target.push_back(1);
    	else
    		target.push_back(0);
    }

	nn->setCurrentInput(input);

	nn->setCurrentTarget(target);

	nn->feedForward();

	//nn->layers[1]->matrixifyActivatedVals()->printToConsole();

	nn->setErrors();

	nn->backPropagation();
	cout<<"data : "<<j<<endl;
	}
	fout<<nn->error<<endl;
	cout<<cal<<" % learned."<<endl;
	}
	
	cout<<endl<<endl<<endl;

	cout<<"saving weights to "<<weightFile<<" file ...\nplease wait\n";

	nn->saveWeights(weightFile);

	cout<<"saving done\n\nyou may close the cmd prompt :):)\n\n";

	nn->loadWeights(weightFile);

	int right_prediction = 0;

	//cout<<trainingData.size();

	/*for(int k = 0; k < trainingData.size(); k++){

	vector<double> input;

	input = trainingData[k];

	input.erase(input.begin() + 0);

	nn->setCurrentInput(input);

	fout<<nn->error<<endl;

	nn->feedForward();

	int result = nn->layers[1]->matrixifyActivatedVals()->getmax();

	cout<<trainingData[k][0]<<" : "<<result<<endl;
	
	if(result == trainingData[k][0])
		right_prediction++;

	}

	cout<<right_prediction<<" / 10000";*/
	topology.clear();
	return 0;
}