package ann;

import java.util.Random;

//n dimensional array
//generalization of vectors and matrices
//actual data container though (not just the shape)
public class Tensor {
	Shape shape;
	float[] data;

	int[] strides;
	int rank;

	public Tensor(Shape shape) {
		this.shape = shape;
	}

	public Tensor(Tensor src) {
		this.shape = src.shape;
		init();
		if (src.getClass() != RandomAccessTensor.class) {
			for (int i = 0; i < data.length; i++) {
				data[i] = src.data[i];
			}
		} else {
			for (int i = 0; i < data.length; i++) {
				data[i] = src.at(src.calcInverseIndex(i));
			}
		}
	}

	public void init() {
		calcStrides();

		// alloc
		data = new float[strides[rank - 1] * shape.getDim(0)];
	}

	public void zero() {
		for (int i = 0; i < data.length; i++)
			data[i] = 0.0f;
	}

	public void init(float[] in) {
		data = in;
	}

	public void calcStrides() {
		rank = shape.rank();
		strides = new int[rank];

		// start with the last stride being 1
		strides[0] = 1;

		// compute strides by successive multiplications
		for (int i = 1; i < rank; i++) {
			strides[i] = strides[i - 1] * shape.getDim(rank - i);
		}
	}

	// initializes every element in the tensor to a random gaussian subject to a
	// skew
	public void randomInit(float skew) {
		init();

		Random random = new Random();
		// iterate through the tensor
		for (int i = 0; i < data.length; i++) {
			data[i] = (float) (random.nextGaussian()) * skew;
		}
	}

	// same as above but no skew
	public void randomInit() {
		randomInit(1.0f);
	}

	// takes the dimensional index (relative to each axis) and converts to flat
	// index
	int calcIndex(int[] ind) {
		int index = ind[0];
		for (int i = 1; i < strides.length; i++) {
			index += strides[i] * ind[i];
		}
		return index;
	}

	// takes the flat index and converts to a dimensional index
	int[] calcInverseIndex(int flatIndex) {
		int[] ind = new int[strides.length];
		int remainingIndex = flatIndex;

		// Iterate over the dimensions in reverse order (starting from the last
		// dimension)
		for (int i = strides.length-1; i >= 0; i--) {
			// Divide the remaining index by the stride for the current dimension
			ind[i] = remainingIndex / strides[i];
			// Update the remaining index for the next iteration
			remainingIndex %= strides[i];
		}

		return ind;
	}
	
	//speed ups for 2x2 matrices
	int calcIndex(int i, int j) {
		return i + strides[1] * j;
	}

	// below are methods to update/access tensor values
	public float at(int[] ind) {
		return data[calcIndex(ind)];
	}
	
	public float at(int i) {
		return data[i];
	}

	public void add(int[] ind, float n) {
		data[calcIndex(ind)] += n;
	}

	public void mul(int[] ind, float n) {
		data[calcIndex(ind)] *= n;
	}

	public void set(int[] ind, float n) {
		data[calcIndex(ind)] = n;
	}

	// global operations on tensor
	public void mul(float s) {
		for (int i = 0; i < data.length; i++)
			data[i] *= s;
	}

	// an element wise multiply (not to be confused with tensor multiply)
	public void elementWiseMultiply(Tensor a) {
		int ones = a.shape.containsOnes();
		if (ones == -1 && a.shape.equals(this.shape) == false) {
			// invalid add
			System.err.println("invalid mul on tensors");
			return;
		} else if (ones != -1) {
			// broadcasting
			for (int i = 0; i < data.length; i++) {
				int[] ind = calcInverseIndex(i);
				ind[ones] = 0;
				data[i] *= a.at(ind);
			}
			return;
		}

		for (int i = 0; i < data.length; i++) {
			data[i] *= a.data[i];
		}
	}

	// element wise add on tensor (only defined when shapes the same)
	public void add(Tensor a) {
		int ones = a.shape.containsOnes();
		if (ones == -1 && a.shape.equals(this.shape) == false) {
			// invalid add
			System.err.println("invalid add on tensors " + this.shape + ", " + a.shape);
			return;
		} else if (ones != -1) {
			// broadcasting
			for (int i = 0; i < data.length; i++) {
				int[] ind = calcInverseIndex(i);
				ind[ind.length - ones - 1] = 0;
				data[i] += a.at(ind);
			}
			return;
		}

		for (int i = 0; i < data.length; i++) {
			data[i] += a.data[i];
		}
	}

	// same as above but for sub
	public void sub(Tensor a) {
		int ones = a.shape.containsOnes();
		if (ones == -1 && a.shape.equals(this.shape) == false) {
			// invalid add
			System.err.println("invalid sub on tensors " + this.shape + ", " + a.shape);
			return;
		} else if (ones != -1) {
			// broadcasting
			for (int i = 0; i < data.length; i++) {
				int[] ind = calcInverseIndex(i);
				ind[ones] = 0;
				data[i] -= a.at(ind);
			}
			return;
		}
		for (int i = 0; i < data.length; i++) {
			float val = (a.getClass() == RandomAccessTensor.class) ? (a.at(a.calcInverseIndex(i))) : (a.data[i]);
			data[i] -= val;
		}
	}

	// tensor sums (reduce the rank of the tensor)
	// creates new tensor
	public Tensor sum(int axis) {
		// check if the axis along which the tensor will be reduced is valid
		if (axis < 0 || axis >= this.shape.rank()) {
			System.err.println(
					"Invalid tensor sum, tensor of rank " + this.shape.rank() + " is too small for axis: " + axis);
			return null;
		}

		// calc the stride
		int stride = strides[strides.length - 1 - axis];

		Shape resultant = new Shape(this.shape.dims);
		resultant.removeAxis(axis);
		Tensor result = new Tensor(resultant);
		result.init();

		for (int i = stride; i < data.length; i++) {
			result.data[i % stride] += data[i];
		}

		return result;
	}

	// sum over ALL tensor axes
	// squares each value too
	// used in loss stat tracking
	public float MSEsum() {
		float sum = 0.0f;
		for (float v : data) {
			sum += v * v;
		}

		return sum;
	}

	// tensor multiplies
	public static Tensor multiply(Tensor a, Tensor b, boolean aTranspose, boolean bTranspose) {
		int ar = a.rank;
		int br = b.rank;

		if (ar == 1 && br == 1) {
			// vector vector
			// outer product or inner product

			// check if sizes same for outer product
			if (a.data.length != b.data.length) {
				// outer product
				return outerProduct(a, b);
			}
		}

		if ((ar == 1 && br == 2) || (br == 1 && ar == 2)) {
			// matrix vector
			if (ar == 1)
				return multiplyMV(b, a, bTranspose);
			return multiplyMV(a, b, aTranspose);
		}
		if (ar == 2 && br == 2) {
			// matrix matrix
			// maintain order it was passed in
			return multiplyMM(a, b, aTranspose, bTranspose);
		}
		if ((ar == 3 && br == 2) || (br == 3 && ar == 2)) {
			// convolution
			return null;
		}
		System.err.println("Rank A: " + ar + ", and Rank B: " + br + " are incompatible tensor multiplies!");
		return null;
	}

	// m and v are matrix and vector respectively
	private static Tensor multiplyMV(Tensor m, Tensor v, boolean matrixTd) {
		if (!matrixTd) {
			// compat check
			if (m.shape.getDim(1) != v.data.length) {
				System.err.println("Incompatible matrix vector multiply! Matrix size: " + m.shape.toString()
						+ ", Vector size: " + v.shape.toString());
			}
			int outputDim = m.shape.getDim(0);
			Tensor value = new Tensor(new Shape(outputDim));
			value.init();

			for (int i = 0; i < outputDim; i++) {
				float sum = 0.0f;
				for (int j = 0; j < v.data.length; j++) {
					sum += m.at(new int[] { i, j }) * v.data[j];
				}
				value.data[i] = sum;
			}

			return value;
		} else {
			// compat check
			if (m.shape.getDim(0) != v.data.length) {
				System.err.println("Incompatible matrix vector transpose multiply! Matrix size: " + m.shape.toString()
						+ ", Vector size: " + v.shape.toString());
			}
			int outputDim = m.shape.getDim(1);
			Tensor value = new Tensor(new Shape(outputDim));
			value.init();

			for (int i = 0; i < outputDim; i++) {
				float sum = 0.0f;
				for (int j = 0; j < v.data.length; j++) {
					sum += v.data[j] * m.at(new int[] { j, i });
				}
				value.data[i] = sum;
			}

			return value;
		}
	}

	// a and b are matrices
	private static Tensor multiplyMM(Tensor a, Tensor b, boolean aTransposed, boolean bTransposed) {
		int aRows = !aTransposed ? a.shape.getDim(1) : a.shape.getDim(0);
		int aCols = !aTransposed ? a.shape.getDim(0) : a.shape.getDim(1);
		int bRows = !bTransposed ? b.shape.getDim(1) : b.shape.getDim(0);
		int bCols = !bTransposed ? b.shape.getDim(0) : b.shape.getDim(1);
		// compat check
		if (aCols != bRows) {
			System.err.println("Incompatible matrix multiply! Matrix A size: " + a.shape.toString()
					+ ", Matrix B size: " + b.shape.toString());
			return null;
		}

		Shape outputShape = new Shape(bCols, aRows);
		Tensor result = new Tensor(outputShape);
		result.init();

		for (int i = 0; i < aRows; i++) {
			for (int v = 0; v < bCols; v++) {
				float sum = 0.0f;
				for (int j = 0; j < aCols; j++) {
					float aVal = !aTransposed ? a.at(a.calcIndex(i, j)) : a.at(a.calcIndex(j, i));
					float bVal = !bTransposed ? b.at(b.calcIndex(j, v)) : b.at(b.calcIndex(v, j));
					//float aVal = !aTransposed ? a.at(new int[] { i, j }) : a.at(new int[] { j, i });
					//float bVal = !bTransposed ? b.at(new int[] { j, v }) : b.at(new int[] { v, j });
					sum += aVal * bVal;
					//sum += aVal * bVal;
				}
				int index = result.calcIndex(i, v);
				result.data[index] = sum;
			}
		}

		return result;
	}

	// a and b are vectors
	private static Tensor outerProduct(Tensor a, Tensor b) {
		Tensor n = new Tensor(new Shape(a.shape.getDim(0), b.shape.getDim(0)));
		n.init();

		// populate
		for (int i = 0; i < a.data.length; i++) {
			for (int j = 0; j < b.data.length; j++) {
				// we can use direct indexing since tensors a and b are rank 1
				n.set(new int[] { i, j }, a.data[i] * b.data[j]);
			}
		}

		return n;
	}

	@Override
	// nice rep of tensor
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(shape.toString()).append("\n");

		switch (rank) {
		case 1:
			// vector representation
			sb.append("[");
			for (int i = 0; i < data.length; i++) {
				sb.append(data[i]);
				if (i != data.length - 1) {
					sb.append(", ");
				}
			}
			sb.append("]");
			break;

		case 2:
			// mat representation
			sb.append("[\n");
			int rows = shape.getDim(0);
			int cols = shape.getDim(1);
			for (int i = 0; i < rows; i++) {
				sb.append("  [");
				for (int j = 0; j < cols; j++) {
					sb.append(at(new int[] { i, j }));
					if (j != cols - 1) {
						sb.append(", ");
					}
				}
				sb.append("]");
				if (i != rows - 1) {
					sb.append(",\n");
				}
			}
			sb.append("\n]");
			break;

		default:
			// arrays of matrices
			int numMatrices = data.length / (shape.getDim(0) * shape.getDim(1));
			for (int matrix = 0; matrix < numMatrices; matrix++) {
				sb.append("Matrix ").append(matrix).append(":\n[\n");
				for (int i = 0; i < shape.getDim(0); i++) {
					sb.append("  [");
					for (int j = 0; j < shape.getDim(1); j++) {
						sb.append(at(new int[] { matrix, i, j }));
						if (j != shape.getDim(1) - 1) {
							sb.append(", ");
						}
					}
					sb.append("]");
					if (i != shape.getDim(0) - 1) {
						sb.append(",\n");
					}
				}
				sb.append("\n]");
				if (matrix != numMatrices - 1) {
					sb.append("\n\n");
				}
			}
			break;
		}

		return sb.toString();
	}

	public String dataView() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < data.length; i++) {
			sb.append(data[i]);
			if (i != data.length - 1)
				sb.append(", ");
		}

		return sb.toString();
	}

}
