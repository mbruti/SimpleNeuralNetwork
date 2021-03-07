package simpleNeuralNetwork;
import java.util.ArrayList;
import java.util.Iterator;

enum ActivationFunction { 
	BINARY("binary"),
	RELU("relu"),
	SOFTMAX("softmax");
	private String afs;
	ActivationFunction(String afs) {
		this.afs=afs;
	}
	public String getActivationFunctionString() {
		return afs;
	}
}

class InConnection {
	Neuron neuron;
	double weight; 
	InConnection(Neuron neuron, double weight) {
		this.neuron=neuron;
		this.weight=weight;
	}
}

public class Neuron {
	ArrayList<Neuron> outConnections;
	ArrayList<InConnection> inConnections;
	ArrayList<Double> weights;
	double threshold;
	double value=0;
	double result=0;
	int layer,index;
	boolean isActive;
	ActivationFunction activationFunction=ActivationFunction.BINARY;
	
	Neuron(int layer) {
		outConnections=new ArrayList<Neuron>();
		inConnections=new ArrayList<InConnection>();
		this.layer=layer;
	}
	
	Neuron(int layer, double value) {
		this(layer);
		this.value=value;
	}
		
	Neuron(int layer, String af) {
		this(layer);
		switch (af) {
		case "relu" : this.activationFunction=ActivationFunction.RELU;break;
		case "softmax" : this.activationFunction=ActivationFunction.SOFTMAX;break;
		default : this.activationFunction=ActivationFunction.BINARY;
		}
	}
	
	String getActivationFunction() {
		return activationFunction.getActivationFunctionString();
	}
	
	double binaryActivationFunction(double input) {
		if (input>0.1) 
			return 1.0;
		else 
			return 0.0;
	}
	
	double reluActivationFunction(double input) {
		if (input>=0) 
			return input;
		else 
			return 0.0;
	}
	
	double softmaxActivationFunction(ArrayList<Double> results) {
		double accumulator=0;
		double res;
		Iterator<Double> it = results.iterator();
		while (it.hasNext()) {
			res=it.next();
			accumulator=accumulator+Math.exp(res);
		}
		return Math.exp(result)/accumulator;
	}
		
	double computeActivationFunction(ArrayList<Double> results) {
		switch (activationFunction) {
		case RELU : 
			return reluActivationFunction(result);
		case SOFTMAX : 
			return softmaxActivationFunction(results);
		default : return binaryActivationFunction(result);
		}
	}
	
	void addOutConnection(Neuron neuron) {
		outConnections.add(neuron);
	}
	
	void addInConnection(Neuron neuron, double weight) {
		InConnection inConnection=new InConnection(neuron,weight);
		inConnections.add(inConnection);
		neuron.addOutConnection(this);
	}
	
	void setActivationValue(double value) {
		this.value=value;
	}
	double getActivationValue() {
		return value;
	}
	
	void setResult(double result) {
		this.result=result;
	}
	double getResult() {
		return result;
	}
	
	int getLayer() {
		return layer;
	}
	
	void setIndex(int index) {
		this.index=index;
	}
	int getIndex() {
		return index;
	}
}