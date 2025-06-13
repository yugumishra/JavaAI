package ann;

//details the optimizer parameters for optimization
//implements basic minibatch SGD with exponential lr decay (maybe create a scheduler class for that later?)
public class Optimizer {
    float lr;
    float decay;
    float decayedLR;
    float l2Gamma;
    boolean l2;
    
    //switches between gradient descent/ascent
    //based on whether you want to minimize log probs (increase likelihood of expected being seen again)
    //or       whether you want to maximize log probs (decrease likelihood of expected being seen again)
    boolean invertDirection;
    //false means gradient descent
    //true means invert, so gradient ascent

    public Optimizer(float lr, float decay, float l2gamma, boolean invertDirection) {
        this.lr = lr;
        this.decay = decay;
        this.decayedLR = lr;
        this.l2Gamma = l2gamma;
        if(l2Gamma < 0.0f) l2 = false;
        this.invertDirection = invertDirection;
    }

    public Optimizer(float lr, float decay, float l2gamma) {
        this(lr, decay, l2gamma, false);
    }

    public Optimizer(float lr, float decay) {
        this(lr, decay, 0.0f, false);
    }

    public Optimizer(float lr) {
        this(lr, 1.0f, 0.0f, false);
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
            if(!invertDirection) {
                weights[i].sub(grads[i]);
            }else {
                weights[i].add(grads[i]);
            }
        }

        return grads;
    }
}
