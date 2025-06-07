package ann;

//representation of a dense layer
//this is represented as a matrix multiply

//strictly outputs a vector (so shape will only be 2 dims)
public class Dense extends Layer {
	//shape representing the matrix multiply (also bias shape, but that's trivial)
	Shape matrix;

	//represents whether or not we have a bias in this dense layer
	boolean bias;
	
	public Dense(Layer prev, Shape shape, boolean bias) {
		super(prev, shape);

		this.bias = bias;
		
		//calculate the required shape of this matrix
		calculateShape();
	}

	public void setBias(boolean b) {
		this.bias = b;
	}
	
	public void calculateShape() {
		//every other case (output = vector)
		int vectorLength = super.getPrev().getShape().getDim(0);
		if(vectorLength != -1) {
			matrix = new Shape(vectorLength, super.getShape().getDim(0));
		}
	}
	
	@Override
	public int numTrainableParams() {
		return (matrix.getDim(0) + ((bias) ? (1) : (0))) * matrix.getDim(1);
	}
	
	@Override
	public Tensor[] weightInit() {
		Tensor[] weights = new Tensor[(bias) ? (2) : (1)];
		
		weights[0] = new Tensor(matrix);
		weights[0].type = ParameterType.DENSE_WEIGHT;
		if(bias) {
			weights[1] = new Tensor(getShape());
			weights[1].type = ParameterType.DENSE_BIAS;
		}
		float glorotConstant = (float) Math.sqrt(2.0 / (getShape().getDim(0) + getShape().getDim(1)));
		weights[0].randomInit(glorotConstant);
			
		if(bias) weights[1].randomInit(1.0f);
		
		return weights;
	}
	
	@Override
	public Tensor forward(Tensor in) {
		//store activations in error (for backprop)
		this.activation = new Tensor(in);
		
		//forward prop
		Tensor out = Tensor.multiply(weights[0], in, false, false);

		if(bias) {
			//implicit broadcast
			weights[1].shape.batch();
			out.add(weights[1]);
			weights[1].shape.debatch();
		}
		return out;
	}
	
	

	@Override
	public Tensor backprop(Tensor in, Optimizer optimizer) {
		Tensor[] weightGradients = new Tensor[(bias) ? (2) : (1)];
		//form the gradients for the bias and weight parameters
		weightGradients[0] = Tensor.multiply(in, this.activation, false, true);
		//accumulate layer errors across the batch axis for bias error
		if(bias) weightGradients[1] = in.sum(0);

		//create this levels error
		Tensor error = Tensor.multiply(weights[0], in, true, false);
		
		//first gradients are from last step
		Tensor[] updatedGrads = optimizer.update(weights, gradients, weightGradients, this.activation.shape.getDim(0));
		gradients = updatedGrads;

		//return
		in = error;
		return error;
	}
}
