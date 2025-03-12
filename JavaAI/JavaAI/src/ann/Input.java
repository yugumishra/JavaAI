package ann;

//layer representing input to the ann
//same as layer, except there is no previous layer (because its the first layer)
public class Input extends Layer {
	public Input(Shape initialShape) {
		super(null, initialShape);
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
		return in;
	}
	
	@Override
	public Tensor backprop(Gradient accumulator, Tensor in) {
		//not needed
		return null;
	}
	
	@Override
	public void gradientUpdate(Tensor grad, int index, float learningRate) {
		//not needed, can't change input
		return;
	}
}
