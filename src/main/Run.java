package main;

import java.util.Scanner;

import ann.*;

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
		
		Dense hidden1 = new Dense(input, new Shape(20), false);
		BatchNormalization batch1 = new BatchNormalization(hidden1);
		Activation act1 = new Activation(batch1, ActivationFunction.RELU);
		Dropout drop1 = new Dropout(act1, 0.5f);
		
		Dense hidden2 = new Dense(drop1, new Shape(10), false);
		BatchNormalization batch2 = new BatchNormalization(hidden2);
		Activation act2 = new Activation(batch2, ActivationFunction.SOFTMAX);

		Ann ann = new Ann(input, act2);
		
		ann.printSummary();
		
		
		System.out.println("\nWhat's the size of a training batch?");
		int batchSize = user.nextInt();
		
		System.out.println("\nHow many epochs would you like to train for?");
		int numEpochs = user.nextInt();

		user.close();
		
		
		Tensor[] trainSet = Utility.readSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
		
		//create optimizer object for training hyperparams
		Optimizer optimizer = new AdamOptimizer(3e-2f, 0.9f, 0.9f, 0.999f, 0.001f);

		//create metric tracker for tracking metrics
		Metrics tracker = new Metrics(true, true, true);
		
		System.out.println("\nTraining beginning!");
		ann.train(trainSet, 0.1f, numEpochs, batchSize, true, optimizer, tracker);
		
		ann.save("MNIST-0.1.1");
	}
	
	public static void loadTest() {
		Tensor[] testSet = Utility.readSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
		Ann ann = Ann.load("MNIST-0.1.1");
		ann.printSummary();
		
		float acc = ann.test(testSet[0], testSet[1]);
		
		System.out.println("Test Set Accuracy: " + 100*acc);
	}
	
	
}
