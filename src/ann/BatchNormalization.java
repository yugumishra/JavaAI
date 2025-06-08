package ann;

public class BatchNormalization extends Layer {
    Tensor runningMean;
    Tensor runningVar;

    float momentum;
    float eps = 1e-6f;

    Tensor x;
    Tensor invVar;
    Tensor xHat;

    public BatchNormalization(Layer prev) {
        super(prev, prev.getShape());
    }

    @Override
    public Tensor[] weightInit() {
        Shape featureShape = new Shape(super.getPrev().getShape().dims);

        weights = new Tensor[2];
        weights[0] = new Tensor(featureShape);
        weights[0].type = ParameterType.BN_GAMMA;
        weights[1] = new Tensor(featureShape);
        weights[1].type = ParameterType.BN_BETA;

        weights[0].init();
        weights[0].ones();
        //init sets to zeros
        weights[1].init();

        return weights;
    }

    public void initTrackers(Shape s) {
        runningMean = new Tensor(s);
        runningVar = new Tensor(s);
        runningMean.init();
        runningVar.init();
        runningMean.type = ParameterType.BN_MU;
        runningVar.type = ParameterType.BN_SIGSQ;
    }

    @Override
    public void weightSet(Tensor[] tensors) {
        super.weights = new Tensor[]{tensors[0], tensors[1]};
        runningMean = tensors[2];
        runningVar = tensors[3];
    }

    @Override
    public Tensor forward(Tensor in) {
        if(super.ann.training) {
            x = new Tensor(in);
            Tensor[] distributionData = in.distribution(0);
            
            invVar = new Tensor(distributionData[1]);
            invVar.invsqrt(eps);
            
            //get shape ready for broadcast
            for(int i = 0; i< 2; i++) {
                distributionData[i].shape.batch();
            }
            //normalize distribution
            in.sub(distributionData[0]);
            in.normalize(distributionData[1], eps);

            xHat = new Tensor(in);

            //debatch
            for(int i =0; i< 2; i++) {
                distributionData[i].shape.debatch();
            }

            if(runningMean == null) {
                initTrackers(distributionData[0].shape);
            }

            //update running mean and var
            distributionData[0].mul(1 - momentum);
            distributionData[1].mul(1 - momentum);
            runningMean.mul(momentum);
            runningMean.add(distributionData[0]);
            runningVar.mul(momentum);
            runningVar.add(distributionData[1]);

            //return
            weights[0].shape.batch();
            in.elementWiseMultiply(weights[0]);
            weights[0].shape.debatch();
            weights[1].shape.batch();
            in.add(weights[1]);
            weights[1].shape.debatch();
            return in;
        }else {
            //normalize distribution
            runningMean.shape.batch();
            in.sub(runningMean);
            runningMean.shape.debatch();
            runningVar.shape.batch();
            in.normalize(runningVar, eps);
            runningVar.shape.debatch();

            //return
            weights[0].shape.batch();
            in.elementWiseMultiply(weights[0]);
            weights[0].shape.debatch();
            weights[1].shape.batch();
            in.add(weights[1]);
            weights[1].shape.debatch();
            return in;
        }
    }

    @Override
    public Tensor backprop(Tensor in, Optimizer optimizer) {
        int N = in.shape.dims[0];

        Tensor[] weightGradients = new Tensor[2];
        weightGradients[0] = new Tensor(in);
        weightGradients[0].elementWiseMultiply(xHat);
        weightGradients[0] = weightGradients[0].sum(0);

        weightGradients[1] = new Tensor(in).sum(0);

        Tensor dXHat = new Tensor(in);
        weights[0].shape.batch();
        dXHat.elementWiseMultiply(weights[0]);
        weights[0].shape.debatch();

        Tensor dx = new Tensor(dXHat);
        dx.mul(N);
        

        Tensor dXX = new Tensor(xHat);
        dXX.elementWiseMultiply(dXHat);
        dXX = dXX.sum(0);
        dXX.shape.batch();
        Tensor term3 = new Tensor(xHat);
        term3.elementWiseMultiply(dXX);
        dXX.shape.debatch();

        Tensor term2 = new Tensor(dXHat).sum(0);
        term2.shape.batch();
        dx.sub(term2);
        term2.shape.debatch();

        dx.sub(term3);

        weights[0].shape.batch();
        dx.elementWiseMultiply(weights[0]);
        weights[0].shape.debatch();

        invVar.shape.batch();
        dx.elementWiseMultiply(invVar);
        invVar.shape.debatch();

        dx.mul(1.0f/((float) N));        

        // Update gamma and beta
        optimizer.update(weights, gradients, weightGradients, N);

        return dx;
    }



    @Override
    public int numTrainableParams() {
        return weights[0].data.length * 2;
    }
}
