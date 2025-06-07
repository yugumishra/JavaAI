package ann;

//details the optimizer parameters for optimization
//implements basic minibatch SGD with exponential lr decay (maybe create a scheduler class for that later?)
public class Optimizer {
    float lr;
    float decay;
    float decayedLR;
    float l2Gamma;
    boolean l2;

    public Optimizer(float lr, float decay, float l2gamma) {
        this.lr = lr;
        this.decay = decay;
        this.decayedLR = lr;
        this.l2Gamma = l2gamma;
        if(l2Gamma < 0.0f) l2 = false;
    }

    public float getLR() {
        return decayedLR;
    }

    public void reset() {
        this.decayedLR = lr;
    }

    public void decay() {
        decayedLR *= decay;
    }

    //for basic minibatch SGD, prevGrads can be null reference
    //which it is
    public Tensor[] update(Tensor[] weights, Tensor[] prevGrads, Tensor[] grads, int batchSize) {
        float scale = decayedLR / ((float) batchSize);
        for(int i = 0; i< weights.length; i++) {
            grads[i].mul(scale);
            if(l2 && ParameterType.l2Applicable(weights[i].type)) {
                weights[i].mul((1.0f - decayedLR * l2Gamma));
            }
            weights[i].sub(grads[i]);
        }

        return grads;
    }
}
