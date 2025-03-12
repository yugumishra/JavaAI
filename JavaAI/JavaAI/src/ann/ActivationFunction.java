package ann;

public enum ActivationFunction {
	RELU(0), SIGMOID(1), SOFTMAX(2);
	
	private int id;
	
	private ActivationFunction(int id) {
		this.id = id;
	}
	
	public int get() {
		return id;
	}
	
	@Override
	public String toString() {
		switch(id) {
		case 0:
			return "ReLU";
		case 1:
			return "Sigmoid";
		case 2:
			return "Softmax";
		default:
			return "Invalid Activation Function";
		}
	}
	
	public static int get(ActivationFunction func) {
		return func.id;
	}
	
	static float activate(ActivationFunction func, float x) {
		switch(func) {
		case RELU:
			return (x>0) ? (x) : (0);
		case SIGMOID:
			float exp = (float) Math.exp(-x);
			return 1.0f / (1.0f + exp);
		default:
			return 0.0f;
		}
	}
	
	float activate(float x) {
		switch(this) {
		case RELU:
			return (x>0) ? (x) : (0);
		case SIGMOID:
			float exp = (float) Math.exp(-x);
			return 1.0f / (1.0f + exp);
		default:
			return 0.0f;
		}
	}
	
	static float activateddx(ActivationFunction func, float x) {
		switch(func) {
		case RELU:
			return (x>0) ? (1) : (0);
		case SIGMOID:
			float exp = (float) Math.exp(-x);
			return exp / ((1.0f + exp) * (1.0f + exp));
		default:
			return 0.0f;
		}
	}
	
	float activateddx(float x) {
		switch(this) {
		case RELU:
			return (x>0) ? (1) : (0);
		case SIGMOID:
			float exp = (float) Math.exp(-x);
			return exp / ((1.0f + exp) * (1.0f + exp));
		default:
			return 0.0f;
		}
	}
}
