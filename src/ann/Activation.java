package ann;

//represents the logits of a previous layer
//only works on vectors
public class Activation extends Layer {
	
	ActivationFunction func;
	
	//needed for backprop
	Tensor logits;
	
	//no shape (its the same as the previous layer's)
	public Activation(Layer prev, ActivationFunction func) {
		super(prev, null);
		this.func = func;
		
		logits = null;
	}
	
	@Override
	public Shape getShape() {
		return super.getPrev().getShape();
	}
	
	@Override
	public int numTrainableParams() {
		return 0;
	}
	
	@Override
	public Tensor[] getWeights() {
		return null;
	}
	
	@Override
	public Tensor forward(Tensor in) {
		//store logits for this layer
		logits = new Tensor(in);
		
		switch(func) {
		case RELU:
			//yay fall through on purpose (so easy man)
		case SIGMOID:
			//element wise
			for(int i = 0; i< in.data.length; i++) in.data[i] = func.activate(in.data[i]);
			break;
		case SOFTMAX:
			//last dimension gets softmaxxed
			//represents the length of each rank 1 tensor (what we are softmaxxing on)
			int len = 0;
			if(in.rank > 1) {
				len = in.strides[1];
			}else {
				len = in.data.length;
			}
			
			//represents the number of softmaxxes to do (or how many rank 1 tensors are in the tensor)
			int numMaxes = in.data.length / len;
			for(int softmax = 0; softmax< numMaxes; softmax++) {
				//calc offset from start
				int start = softmax * len;
				
				//find max element
				float max = Float.NEGATIVE_INFINITY;
				for(int i = start; i < start+len; i++) {
					if (in.data[i] > max) {
		                max = in.data[i];
		            }
				}
				
				//exp each element
				//sum the exp'd versions of the elements in the vector
				float sumExp = 0.0f;
				for(int i = start; i < start+len; i++) {
					in.data[i] = (float) Math.exp(in.data[i] - max);
					sumExp += in.data[i];
				}
				
				//normalize it by the sum
				for(int i = start; i< start+len; i++) {
					in.data[i] /= sumExp;
				}
			}
			break;
		default:
			break;
		}
		
		return in;
	}

	@Override
	public Tensor backprop(Tensor in, Optimizer optimizer) {
		switch(func) {
		case RELU:
			//fall through on purpose
		case SIGMOID:
			for(int i = 0; i< logits.data.length; i++) logits.data[i] = func.activateddx(logits.data[i]);
			break;
		case SOFTMAX:
			//error already flowed (nice gradient calculation)
			return in;
		default:
			break;
		}
		
		//element wises multiply for activation gradient
		in.elementWiseMultiply(logits);
		
		//return
		return in;
	}
}
