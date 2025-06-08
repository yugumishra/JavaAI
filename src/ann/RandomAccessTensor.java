package ann;

import java.util.List;

//a subclass of tensor that access a parent tensor randomly
//only across the batch dimension
//read only
public class RandomAccessTensor extends Tensor {
	Tensor parent;
	List<Integer> samples;

	public RandomAccessTensor(Tensor parent, List<Integer> samples, boolean parentContainsBatchAsWell) {
		super(new Shape(samples.size(), parent.shape, parentContainsBatchAsWell));
		calcStrides();
		this.parent = parent;
		this.samples = samples;
	}

	@Override
	public float at(int[] ind) {
		//return a value from parent that maps the batch index appropriately
		return parent.data[calcIndex(ind)];
	}
	
	public float at(int i) {
		return parent.data[i];
	}
	
	@Override
	//map index to follow the batch
	int calcIndex(int[] ind) {
		int batchIndex = samples.get(ind[ind.length-1]);
		ind[ind.length-1] = batchIndex;

		int flatIndex = parent.calcIndex(ind);
		return flatIndex;
	}
	
	@Override
	//speed up for rank 2 tensor
	int calcIndex(int i, int j) {
		return super.calcIndex(i, samples.get(j));
	}
}
