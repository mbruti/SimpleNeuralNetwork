package simpleNeuralNetwork;
import java.util.Scanner;
public class NeuralNetworkDemo {
	public NeuralNetworkDemo() {
		NeuralNetwork nn=new NeuralNetwork();
		double[] inputs=new double[2];
		double weights01[][]= {{1,-1},{-1,1}};
		System.out.println("NEURAL NETWORK XOR DEMO");
		Scanner scan=new Scanner(System.in);
		System.out.print("Enter first value  0/1: ");
		inputs[0]=scan.nextDouble();
		System.out.print("Enter second value 0/1: ");
		inputs[1]=scan.nextDouble();
		scan.close();
		nn.addInputLayer(2, inputs,0);
		nn.addLayer(2, 1, "relu");
		nn.addLayer(1,2,"relu");
		nn.connectDense(0,1,weights01,1);
		nn.connect(1,1,2,1,1);
		nn.connect(1,2,2,1,1);
		nn.compute(true);
		System.out.println("*** INPUT LAYER ***");
		for (int i=1;i<=nn.getNumNeurons(0);i++) {
			System.out.printf("%f\t",nn.getNeuronActivationValueByIndex(0,i));
		}
		System.out.println();
		for (int i=1;i<nn.getNumLayers()-1;i++) {
			System.out.println("*** HIDDEN LAYER ***");
			for (int j=1;j<=nn.getNumNeurons(i);j++) 
				System.out.printf("%f\t",nn.getNeuronActivationValueByIndex(i,j));
			System.out.println();
		}
		System.out.println("*** OUTPUT LAYER ***");
		for (int i=1;i<=nn.getNumNeurons(nn.getNumLayers()-1);i++) {
			System.out.printf("%f\t",nn.getNeuronActivationValueByIndex(nn.getNumLayers()-1,i));	
		}
		System.out.println("\n\nRisultato="+nn.getNeuronByIndex(2,1).getActivationValue());
	}
}
