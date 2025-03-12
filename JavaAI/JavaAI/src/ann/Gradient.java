package ann;

//container for gradients for each of the trainable parameters of an ann
public class Gradient {
	
	//an array of tensors (each tensor representing the gradients for the learnable params of that layer)
	//same size and shape as ann, but holds gradients instead
	Tensor[] gradient;
	
	int tensorIndex;
	
	int batchSize;
	
	public Gradient(Ann ann, int numTensors) {
		gradient = new Tensor[numTensors];
		//safely skip input layer
		Layer current = ann.input.getNext();
		
		tensorIndex = 0;
		
		//iterate through
		while(current != null) {
			// 0 init tensors
			Tensor[] tensors = current.weightInit(false);
			if(tensors != null) {
				//place
				for(int i = 0; i< tensors.length; i++) {
					gradient[tensorIndex++] = tensors[i];
				}	
			}
			
			//advance
			current = current.getNext();
		}
		
		//initBackprop();
		batchSize= 0;
	}
	
	
	public void initBackprop() {
		tensorIndex = gradient.length-1;
	}
	
	public void finishBackprop() {
		for(Tensor t: gradient) t.zero();
	}
	
	//apply an update to the gradients stored in gradient (assumes you go in backward order)
	//perhaps apply adam gradient
	public void updateGrad(Tensor update) {
		//right now just SGD
		update.mul(1.0f / ((float) batchSize));
		gradient[tensorIndex--].add(update);
	}
}
