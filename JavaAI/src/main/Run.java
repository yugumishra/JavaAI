package main;

import java.util.Scanner;

import ann.Activation;
import ann.ActivationFunction;
import ann.Ann;
import ann.Dense;
import ann.Input;
import ann.Shape;
import ann.Tensor;

public class Run {
	public static void main(String[] args) {
		train();
		loadTest();
	}
	
	public static void train() {

		Scanner user = new Scanner(System.in);
		
		boolean cpu = true;
		boolean valid = false;
		do {
			System.out.println("Would you like this to run on the CPU or GPU?");
			String resp = user.nextLine();
			
			if(resp.equals("CPU")) {
				valid = true;
				cpu = true;
			}
			if(resp.equals("GPU")) {
				valid = true;
				cpu = false;
			}
			
			if(!valid) System.out.println("Sorry, please enter your request again. Enter 'CPU' for CPU training or 'GPU' for GPU training.");
		}while(!valid);
		
		
		Input input = new Input(new Shape(784));
		
		Dense hidden1 = new Dense(input, new Shape(20));
		Activation act1 = new Activation(hidden1, ActivationFunction.RELU);
		
		Dense hidden2 = new Dense(act1, new Shape(10));
		Activation act2 = new Activation(hidden2, ActivationFunction.SOFTMAX);

		Ann ann = new Ann(input, act2);
		
		ann.printSummary();
		
		
		System.out.println("\nWhat's the size of a training batch?");
		int batchSize = user.nextInt();
		
		System.out.println("\nHow many epochs would you like to train for?");
		int numEpochs = user.nextInt();

		user.close();
		
		
		Tensor[] trainSet = Utility.readSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
		System.out.println("\nTraining beginning!");
		ann.train(trainSet[0], trainSet[1], numEpochs, batchSize, 5e-2f);
		
		ann.save("MNIST-0.0.1");
		
		Tensor[] testSet = Utility.readSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
		System.out.println("Validation Set Accuracy: " + 100*ann.test(testSet[0], testSet[1]));
	}
	
	public static void loadTest() {
		Tensor[] testSet = Utility.readSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
		Ann ann = Ann.load("MNIST-0.0.1");
		ann.printSummary();
		
		float acc = ann.test(testSet[0], testSet[1]);
		
		System.out.println("Validation Set Accuracy: " + 100*acc);
	}
	
	
}
