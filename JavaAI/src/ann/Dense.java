package ann;

//representation of a dense layer
//this is represented as a matrix multiply

//strictly outputs a vector (so shape will only be 2 dims)
public class Dense extends Layer {
	//shape representing the matrix multiply (also bias shape, but that's trivial)
	Shape matrix;
	
	public Dense(Layer prev, Shape shape) {
		super(prev, shape);
		
		//calculate the required shape of this matrix
		calculateShape();
	}
	
	public void calculateShape() {
		int id = LayerEnum.get(super.getPrev());;
		//convolutional case (going from convolutional layer to fully connected)
		if(id == 2) {
			return;
		}
		
		//every other case (output = vector)
		int vectorLength = super.getPrev().getShape().getDim(0);
		if(vectorLength != -1) {
			matrix = new Shape(vectorLength, super.getShape().getDim(0));
		}
	}
	
	@Override
	public int numTrainableParams() {
		return (matrix.getDim(0) + 1) * matrix.getDim(1);
	}
	
	@Override
	public Tensor[] weightInit(boolean random) {
		System.out.println("am i getting aclled");
		Tensor[] weights = new Tensor[2];
		
		weights[0] = new Tensor(matrix);
		weights[1] = new Tensor(getShape());
		if(random) {
			float glorotConstant = (float) Math.sqrt(2.0 / (getShape().getDim(0) + getShape().getDim(1)));
			weights[0].randomInit(glorotConstant);
			
			weights[1].randomInit(1.0f);
		}else {
			for(int i = 0; i< 2; i++) weights[i].init();
		}
		
		return weights;
	}
	
	@Override
	public Tensor forward(Tensor in) {
		//store activations in error (for backprop)
		this.activation = new Tensor(in);
		
		//forward prop
		Tensor out = Tensor.multiply(weights[0], in, false, false);

		//implicit broadcast
		weights[1].shape.batch();
		out.add(weights[1]);
		weights[1].shape.debatch();

		return out;
	}
	
	

	@Override
	public Tensor backprop(Tensor in, float lr) {
		//form the gradients for the bias and weight parameters
		Tensor weightGrad = Tensor.multiply(in, this.activation, false, true);
		//accumulate layer errors across the batch axis for bias error
		Tensor biasGrad = in.sum(0);
		
		biasGrad.mul(lr / ((float) this.activation.shape.getDim(0)));
		weightGrad.mul(lr / ((float) this.activation.shape.getDim(0)));
		
		//do the gradient descent
		weights[0].sub(weightGrad);
		weights[1].sub(biasGrad);
		
		//create this levels error
		Tensor error = Tensor.multiply(weights[0], in, true, false);
		in = error;
		return error;
	}
}
