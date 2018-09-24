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

    string trainingFile      = config["trainingData"];
    string weightFile        = config["weightsFile"];
    int hiddenactivationtype = config["hiddenactivationtype"];
    int outputactivationtype = config["outputactivationtype"];
    int costfunctiontype     = config["costfunctiontype"];
    vector<int> topology     = config["topology"];
    int epoch                = config["epoch"];
    double bias              = config["bias"];
    double momentum          = config["momentum"];
    double learningRate      = config["learningRate"];

    vector<vector<double> > trainingData =
    utils::Misc::fetchData(trainingFile);

    NeuralNetwork* nn = new NeuralNetwork(
                                        topology , 
                                        hiddenactivationtype,
                                        outputactivationtype,
                                        costfunctiontype,
                                        bias,
                                        learningRate,
                                        momentum
                                        );

    nn->loadWeights(weightFile);

    cout<<"Weights are loaded successfully for improving accuracy..\n";

    cout<<"Now click Enter to let me learn...\n";

    getchar();

    for(int i = 0; i < epoch; i++){

    double ll = double(i);

    double per = (ll+1) / epoch;

    double cal = per * 100;

    for(int j = 0; j < trainingData.size(); j++){

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
    }
    
    fout<<nn->error<<endl;
    cout<<cal<<" % learned."<<endl;
    }
    
    cout<<endl<<endl<<endl;

    cout<<"saving weights to "<<weightFile<<" file ...\nplease wait\n";

    nn->saveWeights(weightFile);

    cout<<"saving done\n\nyou may close the cmd prompt :):)\n\n";
    
    topology.clear();
    return 0;
}