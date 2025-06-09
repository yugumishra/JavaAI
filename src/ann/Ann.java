package ann;

import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

//encapsulates an entire neural network
//includes references to the input and output of the network
public class Ann {
	
	//ann encoding version number
	public static int VERSION_NUMBER = 2;
	
	// references to layers
	Layer input;
	Layer output;

	int numLayers;

	boolean gpuAccelerated;
	boolean training = true;

	public Ann(Layer input, Layer output) {
		this(input, output, true);
	}
	
	private Ann(Layer input, Layer output, boolean weightSet) {
		this.input = input;
		this.output = output;

		input.ann = this;
		output.ann = this;

		this.gpuAccelerated = false;

		// intialize the data tensors right now
		// as well as the gradient
		if (!gpuAccelerated) {
			numLayers = 1;
			// loop through each layer
			// input is guaranteed to be an input layer here, so skipping it anyway is good
			// initialize current layer (with learnable weights) as the next layer
			Layer current = input.getNext();
			while (current != null) {
				current.ann = this;

				if(weightSet == true) {
					Tensor[] w = current.weightInit();
					current.weights = w;
				}

				numLayers++;

				current = current.getNext();
			}
		}
	}

	public Ann(Ann c) {
		
	}

	// prints a summary of each layer
	public void printSummary() {
		// iterate through
		Layer current = input;
		while (current != null) {
			// print
			System.out.println(current);
			current = current.getNext();
		}
		System.out.println();
	}
	
	public Layer getInput() {
		return input;
	}

	public Tensor predict(Tensor in) {
		training = false;
		return forward(in);
	}

	private Tensor forward(Tensor in) {
		Layer current = input.getNext();
		// avoid destroying the reference
		Tensor curr = in;
		while (current != null) {
			curr = current.forward(curr);
			current = current.getNext();
		}
		return curr;
	}

	// assume in has shape (batch size, input shape)
	// and expected has shape (batch size, output shape)
	public void backprop(Tensor in, Tensor expected, Optimizer optimizer, Metrics tracker) {
		// transform input to output
		training = true;
		Tensor out = forward(in);

		// subtract the ground truth to get error (cross-entropy if softmax)
		out.sub(expected);

		if(tracker.doLoss()) {
			//add cross entropy loss here
			float crossEntropy = out.MSEsum();
			tracker.addLoss(crossEntropy);
		}

		// backprop through the layers using the backward method (backprops layer
		// errors)
		Layer curr = output;
		while (curr != null) {
			//before the weight update, add l2 loss if present
			if(optimizer.l2) {
				Tensor[] weights = curr.getWeights();
				for(int i =0 ; i< weights.length; i++) {
					tracker.addLoss(weights[i].MSEsum());
				}
			}
			//backpropagate the error
			out = curr.backprop(out, optimizer);
			//advance (backwards)
			curr = curr.getPrev();
		}
	}

	public void train(Tensor[] dataset, int numEpochs, int batchSize, boolean timeTaken, Optimizer optimizer, Metrics tracker) {
		train(dataset, 0.0f, numEpochs, batchSize, timeTaken, optimizer, tracker);
	}

	public void train(Tensor[] dataset, float validationProportion, int numEpochs, int batchSize, boolean timeTaken, Optimizer optimizer, Metrics tracker) {
		if (gpuAccelerated)
			/* later */return;

		Tensor in = dataset[0];
		Tensor expected = dataset[1];

		//get dataset size
		int datasetSize = in.shape.getDim(0);
		
		//transform into training set size by setting aside validationProportion amount of the dataset
		boolean validating = validationProportion > 0.0f;
		int trainingSetSize = ((int) ((1.0f-validationProportion) * datasetSize));

		if(!validating) {
			trainingSetSize = datasetSize;
		}

		ArrayList<Integer> allIndices = new ArrayList<>();
		for (int i = 0; i < datasetSize; i++)
			allIndices.add(i);
		
		//shuffle it once
		Collections.shuffle(allIndices);

		//split the dataset into training and validation
		List<Integer> trainingIndices = allIndices.subList(0, trainingSetSize);
		List<Integer> validationIndices = allIndices.subList(trainingSetSize, datasetSize);

		//create the validation tensors
		Tensor validationSet = null, validationAnswers = null;
		if(validating) {
			validationSet = new RandomAccessTensor(in, validationIndices, true);
			validationAnswers = new RandomAccessTensor(expected, validationIndices, true);
		}
		//create the views into trainingIndices for each batch
		int numBatches = trainingSetSize / batchSize;
		ArrayList<List<Integer>> subLists = new ArrayList<>();
		for (int i = 0; i < numBatches; i++) {
			int start = i * batchSize;
			int end = (i + 1) * batchSize;
			subLists.add(trainingIndices.subList(start, end));
		}

		//iterate through each epoch
		for (int epoch = 1; epoch <= numEpochs; epoch++) {
			// mark start time
			long start = System.currentTimeMillis();

			// shuffle sampling order (sublists maintain view)
			Collections.shuffle(trainingIndices);

			// start batches
			for (int batch = 0; batch < numBatches; batch++) {
				// get the batch's dataset
				RandomAccessTensor batchSet = new RandomAccessTensor(in, subLists.get(batch), true);
				RandomAccessTensor batchAnswers = new RandomAccessTensor(expected, subLists.get(batch), true);

				//backpropagate errors (important part of the learning)
				backprop(batchSet, batchAnswers, optimizer, tracker);

				//batch is done, increment batch counter in the tracker (for averaging)
				tracker.batchDone();
			}
			if(validating) {
				//test on validation set (post epoch)
				float acc = 100 * test(validationSet, validationAnswers);
				tracker.setAccuracy(acc);
			}
			//timing end
			long end = System.currentTimeMillis() - start;

			// print time taken
			if(timeTaken) System.out.println("\nEpoch " + (epoch) + " took " + (((int) end) / 1000.0f) + " seconds.");
			
			//print out metrics 
			tracker.printMetrics();

			//steps the optimizer's decay routine
			//must be done every epoch (for accurate learning rate scheduling)
			optimizer.decay();
		}
	}
	
	//tests this ann on a validation set and reports the accuracy metric
	public float test(Tensor in, Tensor expected) {
		//PREDICT the whole batch 
		//important to use predict here, not forward
		//we are in inference mode, so we want training to be false (only done in predict method)
		Tensor out = predict(in);
		
		//get accuracy
		float accuracy = calculateAccuracy(out, expected);
		
		return accuracy;
	}

	// calculates the classification accuracy of a classification network given its
	// output of class probabilities
	// and one hot encoded answers in expected
	// done per batch
	// output shape = expected shape = (batch, num_classes number of logits)
	private float calculateAccuracy(Tensor output, Tensor expected) {
		// use flat iteration instead of at since its faster
		int num_classes = output.strides[1];
		int batchSize = output.data.length / num_classes;

		int correct = 0;
		// iterate over each sample in batch
		for (int vec = 0; vec < output.data.length; vec += num_classes) {
			// argmax indices (start at vec since thats the "0" of each sample)
			int maxO = vec, maxE = vec;

			// iterate over each classification class to argmax
			for (int i = vec + 1; i < vec + num_classes; i++) {
				// argmax on output
				if (output.data[i] > output.data[maxO])
					maxO = i;

				// argmax on expected
				//shift index for random access tensor
				int index = i;
				int mNdex = maxE;
				if(expected.getClass() == RandomAccessTensor.class) { 
					index = expected.calcIndex(expected.calcInverseIndex(i));
					mNdex = expected.calcIndex(expected.calcInverseIndex(maxE));
				}
				if (expected.at(index) > expected.at(mNdex))
					maxE = i;
			}

			// can compare like this since it does not matter if indices are displaced by
			// vec or not
			if (maxO == maxE) {
				correct++;
			}
		}

		return ((float) correct / batchSize);
	}

	// encodes the ANN in this object into a bin file for saving
	// includes metadata for how to read
	// formatted like this
	// file encoding version number
	// numLayers
	// layer metadata (layer type, tensor shape)
	// data (in order of tensor appearance per layer)
	public void save(String name) {
		// create file with .bin ending
		FileOutputStream fos = null;
		try {
			fos = new FileOutputStream(name + ".bin");
		} catch (FileNotFoundException e) {
			System.err.println("File Output Stream failed to be created!");
			e.printStackTrace();
		}
		
		// use dataoutputstream class to send data
		DataOutputStream dos = new DataOutputStream(fos);
		try {
			//version number
			dos.writeInt(VERSION_NUMBER);
		
			//num layers
			dos.writeInt(numLayers);
			
			//layer metadata
			Layer curr = input;
			while(curr != null) {
				//layer type
				dos.writeInt(LayerEnum.get(curr.getClass()));
				
				//write layer shape
				Shape sh = curr.getShape();
				//write dimension of this tensor
				dos.writeInt(sh.dims.length);
				//now write each dimension
				for(int dim = 0; dim < sh.dims.length; dim++) {
					dos.writeInt(sh.dims[dim]);
				}
				
				//if activation, write the activation function id
				if(curr.getClass() == Activation.class) {
					dos.writeInt(ActivationFunction.get(((Activation) curr).func));
				}
				
				//tensor metadata
				Tensor[] weights = curr.weights;
				//write number of tensors
				int numTensors = (weights == null) ? (0): (weights.length);
				if(curr.getClass() == BatchNormalization.class) {
					numTensors += 2;
				}
				dos.writeInt(numTensors);
				if(weights != null) {
					for(int i = 0; i< weights.length; i++) {
						//write parameter type
						dos.writeInt(weights[i].type.get());

						//write shape
						Shape s = weights[i].shape;
						//write dimension of this tensor
						dos.writeInt(s.dims.length);
						//now write each dimension
						for(int dim = 0; dim < s.dims.length; dim++) {
							dos.writeInt(s.dims[dim]);
						}
						
						//now write data
						for(int ah = 0; ah< weights[i].data.length; ah++) {
							dos.writeFloat(weights[i].data[ah]);
						}
					}
					if(curr.getClass() == BatchNormalization.class) {
						BatchNormalization bn = (BatchNormalization) curr;

						//write param type
						dos.writeInt(bn.runningMean.type.get());
						//write shape
						Shape s = bn.runningMean.shape;
						//write dimension of this tensor
						dos.writeInt(s.dims.length);
						//now write each dimension
						for(int dim = 0; dim < s.dims.length; dim++) {
							dos.writeInt(s.dims[dim]);
						}
						
						//now write data
						for(int ah = 0; ah< bn.runningMean.data.length; ah++) {
							dos.writeFloat(bn.runningMean.data[ah]);
						}

						//write param type
						dos.writeInt(bn.runningVar.type.get());
						//write shape
						s = bn.runningVar.shape;
						//write dimension of this tensor
						dos.writeInt(s.dims.length);
						//now write each dimension
						for(int dim = 0; dim < s.dims.length; dim++) {
							dos.writeInt(s.dims[dim]);
						}
						
						//now write data
						for(int ah = 0; ah< bn.runningVar.data.length; ah++) {
							dos.writeFloat(bn.runningVar.data[ah]);
						}
					}
				}
				//move to next layer
				curr = curr.getNext();
			}
			
		}catch (IOException e) {
			System.err.println("There was an error writing the ANN!");
			e.printStackTrace();
		}
		
		try {
			dos.close();
			fos.close();
		}catch (IOException e) {
			System.err.println("There was an error closing the file!");
			e.printStackTrace();
		}
	}
	
	//loads an ann from a file
	public static Ann load(String name) {
	    FileInputStream fis = null;
	    try {
	    	try {
	    		fis = new FileInputStream(name + ".bin");
	    	}catch (FileNotFoundException e) {
	    		System.err.println("The ANN being loaded does not exist!");
	    		e.printStackTrace();
	    	}
	    	//data reader
	        ByteBuffer buff = ByteBuffer.wrap(fis.readAllBytes());
	        
	        //version number check
	        int num = buff.getInt();
	        if(num != VERSION_NUMBER) {
	        	System.err.println("Old ANN encoding, incompatible with loading!");
	        	throw new IOException("Old Version Number");
	        }
	        
	        //read num layers (know how much to iterate for)
	        int numLayers = buff.getInt();
	        
	        Layer in = null, out = null;
	        Layer prev = null;
	        Layer curr = in;

	        //read metadata for each layer
	        for (int i = 0; i < numLayers; i++) {
	        	//read layer type & class
	            int layerType = buff.getInt();
	            Class<? extends Layer> clas = LayerEnum.get(layerType);
	            
	            //read layer shape
	            int rrank = buff.getInt();
	            int[] dimms = new int[rrank];
	            for(int d = 0; d < dimms.length; d++) {
	            	dimms[d] = buff.getInt();
	            }
	            Shape layerShape = new Shape(dimms);
	            
	            //create the layer
	            if(clas == Input.class) { 
	            	//create layer and advance
	            	curr = new Input(layerShape);
	            	in = curr;
	            	prev = curr;
	            	curr = null;
	            }else if(clas == Dense.class) {
	            	//create layer and advance
	            	curr = new Dense(prev, layerShape, true);
	            	prev = curr;
	            	curr = null;
	            }else if(clas == Activation.class) { 
	            	//read activation function
	            	ActivationFunction func = ActivationFunction.get(buff.getInt());
	            	//create layer and advance
	            	curr = new Activation(prev, func);
	            	prev = curr;
	            	curr = null;
	            }else if (clas == BatchNormalization.class) {
					curr = new BatchNormalization(prev);
					prev = curr;
					curr = null;
				}
	            
	            //read tensor shapes
	            int numTensors = buff.getInt();
				if(numTensors == 1 && clas == Dense.class) {
					((Dense) prev).setBias(false);
				}
	            if(numTensors == 0) continue;
	            Tensor[] tensors = new Tensor[numTensors];

	            for (int t = 0; t < numTensors; t++) {
					//read parameter type
					ParameterType type = ParameterType.get(buff.getInt());

	                //read the rank
	            	int rank = buff.getInt();
	                int[] dims = new int[rank];
	                //read the shape
	                for (int d = 0; d < dims.length; d++) {
	                    dims[d] = buff.getInt();
	                }
	                //create new tensor
	                //allocate tensors with shape
	                tensors[t] = new Tensor(new Shape(dims)); 
					//set type
					tensors[t].type = type;
	                
	                //read tensor data
	                tensors[t].init();
	                for (int ah = 0; ah < tensors[t].data.length; ah++) {
	                    tensors[t].data[ah] = buff.getFloat();
	                    
	                }
	            }
	            
	            prev.weightSet(tensors);
	        }
	        //set out to the final layer created
	        out = prev;
	        
	        //instantiate and return
	        return new Ann(in, out, false);

	    } catch (IOException e) {
	    	System.err.println("Something went wrong reading the ANN!");
	        e.printStackTrace();
	        return null;
	    } finally {
	        try {
	            if (fis != null) fis.close();
	        } catch (IOException e) {
	            e.printStackTrace();
	        }
	    }
	    
	    
	}

}
