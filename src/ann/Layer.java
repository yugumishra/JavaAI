package ann;

//defines the base class for layer
//all neural networks are composed of layers
//so this class defines what a layer is required to have
public abstract class Layer {
	//the reference to the previous layer
	private Layer prev;
	//the reference to the next layer
	private Layer next;
	
	//shape describing the tensor's shape AFTER the operation at hand (post layer op)
	private Shape shape;
	
	//tensor holding this layer's learnable parameters
	Tensor[] weights;
	//tensor holding the gradient to this layer's params
	Tensor[] gradients;
	
	//tensor holding this layer's activation
	Tensor activation;
	
	//reference to parent ann
	Ann ann;
	
	protected Layer(Layer previous, Shape shape) {
		//reference to previous
		prev = previous;
		
		//earmark the previous layers' next right here
		if(previous != null) previous.next = this;
		
		//null initializer (when this layer is passed into other layers, it will get set)
		this.next = null;
		
		//shape init
		this.shape = shape;
		
		
		weights = null;
		activation = null;
		ann = null;
		gradients = null;
	}
	
	public Tensor[] weightInit() {
		return null;
	}
	
	//by reference copy
	public void weightSet(Tensor[] weights) {
		this.weights = weights;
	}
	
	public Shape getShape() {
		return shape;
	}
	
	public Layer getNext() {
		return next;
	}
	
	public Layer getPrev() {
		return prev;
	}
	
	public Tensor[] getWeights() {
		return weights;
	}
	
	//has each layer store the activation of itself in a variable (for the backprop pass to use)
	public abstract Tensor forward(Tensor in);
	
	public abstract Tensor backprop(Tensor in, Optimizer optimizer);
	
	public abstract int numTrainableParams();
	
	@Override
	public String toString() {
		return LayerEnum.toStr(this) + " Layer, " + this.getShape().toString() + ", trainable parameters: " + numTrainableParams();
	}
}
