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

    //special scale tensor
    //needed for very niche per example scales on gradients (really just a hack for gamma discounting)
    //multiplied to the difference f(x) - e(x) (to scale the gradient)
    Tensor customScale;

    public Optimizer(float lr, float decay, float l2gamma, boolean invertDirection) {
        this.lr = lr;
        this.decay = decay;
        this.decayedLR = lr;
        this.l2Gamma = l2gamma;
        if(l2Gamma < 0.0f) l2 = false;
        this.invertDirection = invertDirection;

        //set custom scale to null (if you want to scale custom, you must do so in a different method call)
        customScale = null;
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

    public Tensor customScale() {
        return customScale;
    }

    //by reference copy of custom scale (which is multiplied to f(x) - e(x), before backward pass)
    public void customScale(Tensor t) {
        customScale = t;
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
