package ann;

//this class defines the shape of a tensor (symbolic)
//so these are defined by users to create tensors of any size (up to 4th order)
//the ann system then uses these to shape the neural network
//and are further used for training & computation/usage
public class Shape {
	//constant size array of maximum size 4
	//usually the first dimension is the batch dimension
	//the next 2 are either generic data (for one dimensional data)
	//or 2 dimensions representing the dimensions of the data (for images)
	//the last dimension is usually channel size
	int[] dims;
	
	//initializer for image
	public Shape(int x, int y, int channels) {
		dims = new int[] {x, y, channels};
	}
	
	//another constructor, but for 1d data (vector)
	//constructor uses -2 sentinel value to indicate batch will be provided at runtime
	public Shape(int dim) {
		dims = new int[] {dim};
	}
	
	//matrix representation for matrices in dense layers
	public Shape(int x, int y) {
		dims = new int[] {x, y};
	}
	
	public Shape(int batch, Shape s, boolean shapeContainsBatchAsWell) {
		if(!shapeContainsBatchAsWell) {
			int[] newDims = new int[s.rank() + 1];
			newDims[0] = batch;
			for(int i = 0; i< s.rank(); i++) {
				newDims[i+1] = s.dims[i];
			}
			dims = newDims;
		}else {
			int[] newDims = new int[s.rank()];
			newDims[0] = batch;
			for(int i = 1; i< s.rank(); i++) {
				newDims[i] = s.dims[i];
			}
			dims = newDims;
		}
	}
	
	public Shape(int[] dims) {
		this.dims = new int[dims.length];
		for(int i = 0; i < dims.length; i++) {
			this.dims[i] = dims[i];
		}
	}

	public Shape(Shape s) {
		this(s.dims);
	}
	
	//transform symbolic to datamoving shape dsecriptor
	public void batch() {
		batch(1);
	}
	
	public void batch(int batchSize) {
		int[] newDims = new int[rank() + 1];
		newDims[0] = batchSize;
		for(int i = 0; i< rank(); i++) {
			newDims[i+1] = dims[i];
		}
		dims = newDims;
	}
	
	public void debatch() {
		int[] newDims = new int[rank() - 1];
		for(int i = 1; i< rank(); i++) {
			newDims[i-1] = dims[i];
		}
		dims = newDims;
	}
	
	public void removeAxis(int axis) {
		int[] newDims = new int[rank() - 1];
		int index = 0;
		for(int i =0; i< rank(); i++) {
			if(i == axis) continue;
			newDims[index++] = dims[i];
		}
		dims = newDims;
	}
	
	//returns the number of dimensions
	public int rank() {
		return dims.length;
	}
	
	//gets the max at dimension i
	public int getDim(int i) {
		if(i < 0 || i > dims.length - 1) return -1;
		return dims[i];
	}
	
	//returns the presence of ones (indicates broadcasting)
	public int containsOnes() {
		for(int i = 0; i< dims.length; i++) {
			if(dims[i] == 1) return i;
		}
		return -1;
	}
	
	//to string for print summaries
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("shape=(");
		for(int i = 0;i< dims.length; i++) {
			sb.append(dims[i]);
			if(i != dims.length - 1) sb.append(", ");
		}
		sb.append(")");
		return sb.toString();
	}
	
	//shape comparison
	@Override
	public boolean equals(Object obj) {
		if(obj.getClass() != Shape.class) return false;
		Shape other = (Shape) obj;
		if(other.rank() != rank()) return false;
		for(int i = 0; i< other.rank(); i++) {
			if(other.dims[i] != dims[i]) return false;
		}
		return true;
	}
}
