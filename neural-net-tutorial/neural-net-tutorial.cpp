#include<vector>
#include<cmath>
#include<iostream>
#include<cstdlib>
#include<cassert>
#include<fstream>
#include<sstream>
#include<string>


using namespace std;

class TrainingData {

public:
	TrainingData(const string filename);
	~TrainingData();
	bool isEof(void) { return m_trainingDataFile.eof(); }
	void getTopology(vector<unsigned>& topology);

	unsigned getNextInput(vector<double>& inputVals);
	unsigned getTargetOutputs(vector<double>& targetOutputVals);

private:
	ifstream m_trainingDataFile;
	
};

unsigned TrainingData::getNextInput(vector<double>& inputVals) {
	string line; 
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;

	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}	
	}
	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double>& targetOutputVals) {
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);

	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}
	return targetOutputVals.size();
}


TrainingData::TrainingData(const string filename) {
	m_trainingDataFile.open(filename);
}

TrainingData::~TrainingData(){

}

void TrainingData::getTopology(vector<unsigned>& topology) {
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);
	ss >> label;

	if (line.empty() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

}


struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	~Neuron();
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer& prevLayer);
	void calcOutputGradients(double const targetVal);
	void calHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer& prevLayer);


private:
	static double eta;
	static double alpha;
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer& nextLayer) const;
	double m_outputVal;
	vector<Connection> m_outpoutWeights;
	unsigned m_myIndex;
	double m_gradient;
	 

};

void Neuron::updateInputWeights(Layer& prevLayer) {
	
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outpoutWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient
								+ alpha * oldDeltaWeight;

	}
}


double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

double Neuron::sumDOW(const Layer& nextLayer) const {

	double sum = 0.0;
	for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
		sum += m_outpoutWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calHiddenGradients(const Layer& nextLayer) {
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);

};

void Neuron::calcOutputGradients(double const targetVal) {
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
};

double Neuron::transferFunction(double x) {
	// Tanh - output range(-1.0 ... 1.0)
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer& prevLayer) {

	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size(); n++) {
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outpoutWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
	
};


Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {

	for (unsigned c = 0; c < numOutputs; c++) {
		m_outpoutWeights.push_back(Connection());
		m_outpoutWeights.back().weight = randomWeight();

	}

	m_myIndex = myIndex;
};

Neuron::~Neuron() {
};




class Net {
public:
	Net(const vector<unsigned> &topology);
	~Net();
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};

void Net::getResults(vector<double>& resultVals) const {

	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; n++) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}

}

void Net::backProp(const vector<double>& targetVals) {

	// Calculate overall net error (RMS of output neuron errors)
	Layer& outputLayer = m_layers.back();
	m_error = 0.0;
	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error = sqrt((1 / (outputLayer.size()-1)) * (m_error));

	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) 
							/ (m_recentAverageSmoothingFactor + 1.0);

	// Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size()-1; n++) {
		outputLayer[n].calcOutputGradients(targetVals[n]);

	}

	// Calculate gradients on hidden layers
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) {
		Layer& hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); n++) {
			hiddenLayer[n].calHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// Updata connection weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; n++) {
			layer[n].updateInputWeights(prevLayer);
		}
	 }
}

void Net::feedForward(const vector<double>& inputVals) {
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign input values
	for (unsigned i = 0; i < inputVals.size(); i++) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer& prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}

	}


}

Net::Net(const vector<unsigned>& topology) {
	unsigned numLayers = topology.size();

	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == numLayers - 1 ? 0 : topology[layerNum + 1];

			for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
				m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			}

	}

}

Net::~Net() {

}

void showVectorVals(string label, vector<double>& v) {
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); i++) {

		cout << v[i] << " ";
	}

	cout << endl;

}

int main() {


	TrainingData trainData("trainingData.txt");

	vector<unsigned> topology;
	trainData.getTopology(topology);

	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;
	int correctNum = 0;

	while (!trainData.isEof()) {
		inputVals.clear();
		targetVals.clear();
		resultVals.clear();

		trainingPass++;
		std::cout << endl << "Pass " << trainingPass;

		if (trainData.getNextInput(inputVals) != topology[0]) {
			break;
		}
		showVectorVals(": Inputs:", inputVals);
		myNet.feedForward(inputVals);

		myNet.getResults(resultVals);
		showVectorVals("Output:", resultVals);
	
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);
	
		std::cout << "Net recent average error: "
			<< myNet.getRecentAverageError() << endl;

		if (abs(targetVals.back() - resultVals.back()) < 0.1)
			correctNum++;

	}

	std::cout << endl;
	std::cout << "Correct Number: " << correctNum << endl;
	std::cout << endl << "Done" << endl;


}