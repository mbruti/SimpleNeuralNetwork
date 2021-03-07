package simpleNeuralNetwork;
import java.util.ArrayList;

public class NeuralNetwork {
	ArrayList<Neuron> neuralNetwork; 
	int numLayers;
	int[] countNeuronsPerLayer=new int[100];
	
	NeuralNetwork() {
		neuralNetwork=new ArrayList<Neuron>();
		numLayers=0;
		for(int i=0;i<100;i++) 
			countNeuronsPerLayer[i]=0;
	}
	
	void addNeuron(Neuron neuron) {
		neuralNetwork.add(neuron);
		if (numLayers<(neuron.layer+1)) numLayers=neuron.layer+1;
		countNeuronsPerLayer[neuron.layer]+=1;
		neuron.setIndex(countNeuronsPerLayer[neuron.layer]);
	}

	int getNumLayers() {
		return numLayers;
	}
	
	int getNumNeurons(int layer) {
		return countNeuronsPerLayer[layer];
	}
	
	int getTotalNumNeurons() {
		int sumNeurons=0;
		for (int i=0;i<numLayers;i++) {
			sumNeurons+=countNeuronsPerLayer[i];
		}
		return sumNeurons;
	}
	
	void addInputLayer(int nNeurons, double inputs[],double dflt) {
		double inputValue;
		for (int i=0;i<nNeurons;i++) {
			if (i>(inputs.length-1))
				inputValue=dflt;
			else
				inputValue=inputs[i];
			Neuron n=new Neuron(0,inputValue);
			addNeuron(n);
		}
	}
	
	void addLayer(int nNeurons, int layer, String activationFunctionString) {
		for (int i=0;i<nNeurons;i++) {
			Neuron n=new Neuron(layer,activationFunctionString);
			addNeuron(n);
		}
	}
	
	Neuron getNeuronByIndex(int layer, int neuronIndex) {
		for (Neuron n : neuralNetwork) {
			if ((n.getIndex()==neuronIndex) && (n.getLayer()==layer))
				return n;
		}
		return null;
	}
	
	double getNeuronResultByIndex(int layer, int neuronIndex) {
		return getNeuronByIndex(layer,neuronIndex).getResult();
	}
	
	double getNeuronActivationValueByIndex(int layer, int neuronIndex) {
		return getNeuronByIndex(layer,neuronIndex).getActivationValue();
	}
	
	void connect(int startLayer,int startNeuronIndex,int endLayer,int endNeuronIndex, double weight) {
		Neuron startNeuron=getNeuronByIndex(startLayer,startNeuronIndex);
		Neuron endNeuron=getNeuronByIndex(endLayer,endNeuronIndex);
		endNeuron.addInConnection(startNeuron, weight);
	}
	
	void connectDense(int startLayer, int endLayer, double weights[][],int dflt) {
		double weight;
		for (int i=0;i<countNeuronsPerLayer[startLayer];i++) {
			for (int j=0;j<countNeuronsPerLayer[endLayer];j++) {
				if (i<weights.length) {
					if (j<weights[i].length) {
						weight=weights[i][j];
					} else {
						weight=dflt;
					}
				} else {
					weight=dflt;
				}
				connect(startLayer,i+1,endLayer,j+1,weight);
			}
		}
	}
	
	void compute(boolean verbose) {
		ArrayList<Double>[] results=new ArrayList[100];
		for(int i=0;i<100;i++)
			results[i]=new ArrayList<Double>();
		if (verbose) {
			System.out.println("COMPUTING NEURAL NETWORK...");
			System.out.println("Tot neurons: "+getTotalNumNeurons());
			for (int i=0;i<getNumLayers();i++) 
				System.out.println("\tNumber Neurons Layer "+i+": "+getNumNeurons(i));
			System.out.println("\nLayer 0 (Input):");
			for (Neuron n : neuralNetwork) {
				if (n.getLayer()==0) {
					System.out.println("\tInput Neuron "+n.getIndex()+" = "+n.getActivationValue());
				}
			}
			System.out.println();
		}
		for(int i=1;i<numLayers;i++) {
			if (verbose) {
				if (i==(numLayers-1)) 
					System.out.println("Output Layer "+i+":");
				else
					System.out.println("Hidden Layer "+i+":");
			}
			for (Neuron n : neuralNetwork) {
				if (n.getLayer()==i) {
					double result=0;
					for (InConnection inc : n.inConnections) {
						result=result+inc.weight*inc.neuron.getActivationValue();
					}
					n.setResult(result);
					results[i].add(result);
				}
			}
			for (Neuron n : neuralNetwork) {
				if (n.getLayer()==i) {
					n.setActivationValue(n.computeActivationFunction(results[i]));
					if (verbose) {
						System.out.print("\tNeuron "+n.getIndex()+" ActFunction="+n.getActivationFunction());
						System.out.println("  Sum(i*w)="+n.getResult()+" => f(sum(i*w))="+n.getActivationValue());
					}
				}
			}
			if (verbose) System.out.println();
		}
	}
}
