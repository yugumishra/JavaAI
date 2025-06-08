package ann;

public enum LayerEnum {
	INVALID(-1), INPUT(0), DENSE(1), CONVOLUTION(2), BATCHNORM(3), ACTIVATION(4), DROPOUT(5);
	
	private int id;
	
	LayerEnum(int id) {
		this.id = id;
	}
	
	public int get() {
		return id;
	}
	
	public static int get(Class<?> classType) {
		if(classType == Input.class) return 0;
		if(classType == Dense.class) return 1;
		if(classType == Input.class) return 2;
		if(classType == BatchNormalization.class) return 3;
		if(classType == Activation.class) return 4;
		if(classType == Dropout.class) return 5;
		return -1;
	}
	
	public static Class<? extends Layer> get(int id) {
		if(id == 0) return Input.class;
		if(id == 1) return Dense.class;
		if(id == 2) return Input.class;
		if(id == 3) return BatchNormalization.class;
		if(id == 4) return Activation.class;
		if(id == 5) return Dropout.class;
		return null;
	}
	
	public static LayerEnum getEnum(Class<?> classType) {
		if(classType == Input.class) return INPUT;
		if(classType == Dense.class) return DENSE;
		if(classType == Input.class) return CONVOLUTION;
		if(classType == BatchNormalization.class) return BATCHNORM;
		if(classType == Activation.class) return ACTIVATION;
		if(classType == Dropout.class) return DROPOUT;
		return INVALID;
	}
	
	public static int get(Layer l) {
		return get(l.getClass());
	}
	
	public static LayerEnum getEnum(Layer l) {
		return getEnum(l.getClass());
	}
	
	public static String toStr(Layer l) {
		switch(get(l.getClass())) {
		case 0:
			return "Input";
		case 1:
			return "Dense";
		case 2:
			return "Convolutional";
		case 3:
			return "Batch Normalization";
		case 4:
			String str = "Activation (";
			String act = ((Activation) l).func.toString();
			str +=  act+ ")";
			return str;
		case 5:
			return "Dropout";
		default:
			return "Invalid ";
		}
	}
	
	public static int numTensors(Layer l) {
		switch(getEnum(l.getClass())) {
		case INPUT:
			return 0;
		case DENSE:
			return 2;
		case CONVOLUTION:
			return 0;
		case BATCHNORM:
			return 0;
		case ACTIVATION:
			return 0;
		case DROPOUT:
			return 0;
		default:
			return -1;
		}
	}
	public static int numTensors(Class<? extends Layer> clazz) {
		switch(getEnum(clazz)) {
		case INPUT:
			return 0;
		case DENSE:
			return 2;
		case CONVOLUTION:
			return 0;
		case BATCHNORM:
			return 0;
		case ACTIVATION:
			return 0;
		default:
			return -1;
		}
	}
}
