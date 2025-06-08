package ann;

public class Dropout extends Layer{
    float dropoutp;
    Tensor mask;

    public Dropout(Layer prev, float rate) {
        super(prev, prev.getShape());
        this.dropoutp = rate;
    }

    public Dropout(Layer prev) {
        this(prev, 0.5f);
    }

    @Override
    public Tensor forward(Tensor in) {
        if(super.ann.training) {
            mask = Tensor.randomMask(in.shape, 1.0f - dropoutp);
            mask.mul(1.0f / (1.0f - dropoutp));
            in.elementWiseMultiply(mask);
            
        }
        return in;
    }

    @Override
    public Tensor backprop(Tensor in, Optimizer optimizer) {
        if(ann.training) {
            in.elementWiseMultiply(mask);
        }
        return in;
    }

    @Override
    public int numTrainableParams() {
        return 0;
    }
}
