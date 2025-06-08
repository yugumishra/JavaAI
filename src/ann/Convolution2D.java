package ann;

public class Convolution2D extends Layer {

    Padding padding;
    Stride stride;
    int inC;
    int outC;
    int kernel;

    public Convolution2D(Layer prev, Shape s, int kernel, int outC, Padding padding, Stride stride) {
        super(prev, s);

        this.kernel = kernel;
        this.outC = outC;
        this.padding = padding;
        this.stride = stride;
    }

    // @Override
    // public Tensor[] weightInit() {

    // }

    @Override
    public Tensor forward(Tensor in) {
        throw new UnsupportedOperationException("Unimplemented method 'forward'");
    }

    @Override
    public Tensor backprop(Tensor in, Optimizer optimizer) {
        throw new UnsupportedOperationException("Unimplemented method 'backprop'");
    }

    @Override
    public int numTrainableParams() {
        throw new UnsupportedOperationException("Unimplemented method 'numTrainableParams'");
    }
    
}

class Padding {
    int paddingX;
    int paddingY;

    public Padding(int paddingX, int paddingY) {
        this.paddingX = paddingX;
        this.paddingY = paddingY;
    }
}

class Stride {
    int strideX;
    int strideY;

    public Stride(int strideX, int strideY) {
        this.strideX = strideX;
        this.strideY = strideY;
    }
}


