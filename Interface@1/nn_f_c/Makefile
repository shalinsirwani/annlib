output: continue_training.o Neuron.o Layer.o Matrix.o Math.o NeuralNetwork.o feedForward.o Backpropogation.o setErrors.o Constructor.o Misc.o Classifier.o SaveWeights.o
	g++ -std=c++17 -Wall -fexceptions continue_training.o Neuron.o Layer.o Matrix.o Math.o NeuralNetwork.o feedForward.o Backpropogation.o setErrors.o Constructor.o Misc.o Classifier.o SaveWeights.o -o output

continue_training.o: continue_training.cpp
	g++ -std=c++17 -Wall -fexceptions -c continue_training.cpp

Neuron.o: Neuron.cpp Neuron.hpp
	g++ -std=c++17 -Wall -fexceptions -c Neuron.cpp

Layer.o: Layer.cpp Layer.hpp
	g++ -std=c++17 -Wall -fexceptions -c Layer.cpp

Matrix.o: Matrix.cpp Matrix.hpp
	g++ -std=c++17 -Wall -fexceptions -c Matrix.cpp

Math.o: Math.cpp Math.hpp
	g++ -std=c++17 -Wall -fexceptions -c Math.cpp

NeuralNetwork.o: NeuralNetwork.cpp NeuralNetwork.hpp  
	g++ -std=c++17 -Wall -fexceptions -c NeuralNetwork.cpp  

Backpropogation.o: Backpropogation.cpp NeuralNetwork.hpp
	g++ -std=c++17 -Wall -fexceptions -c Backpropogation.cpp

feedForward.o: feedForward.cpp NeuralNetwork.hpp
	g++ -std=c++17 -Wall -fexceptions -c feedForward.cpp

setErrors.o: setErrors.cpp NeuralNetwork.hpp
	g++ -std=c++17 -Wall -fexceptions -c setErrors.cpp

Constructor.o: Constructor.cpp NeuralNetwork.hpp
	g++ -std=c++17 -Wall -fexceptions -c Constructor.cpp

Misc.o: Misc.cpp Misc.hpp
	g++ -std=c++17 -Wall -fexceptions -c Misc.cpp

Classifier.o: Classifier.cpp NeuralNetwork.hpp json.hpp
	g++ -std=c++17 -Wall -fexceptions -c Classifier.cpp

SaveWeights.o: SaveWeights.cpp NeuralNetwork.hpp json.hpp
	g++ -std=c++17 -Wall -fexceptions -c SaveWeights.cpp

clean:
	rm *.o