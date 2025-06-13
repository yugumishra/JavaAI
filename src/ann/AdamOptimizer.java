package ann;

//adam optim (child of optim)
public class AdamOptimizer extends Optimizer{
    float b1;
    float b2;
    int count = 1;

    public AdamOptimizer(float lr, float decay, float b1, float b2, float l2gamma, boolean invertDirection) {
        super(lr, decay, l2gamma, invertDirection);
        this.b1 = b1;
        this.b2 = b2;
    }

    public AdamOptimizer(float lr, float decay, float b1, float b2, float l2gamma) {
        this(lr, decay, b1, b2, l2gamma, false);
    }

    public AdamOptimizer(float lr, float decay, float l2gamma) {
        this(lr, decay, 0.9f, 0.999f, l2gamma, false);
    }

    public AdamOptimizer(float lr, boolean invertDirection) {
        this(lr, 1.0f, 0.9f, 0.999f, 0.0f, invertDirection);
    }

    public AdamOptimizer(float lr) {
        this(lr, 1.0f, 0.9f, 0.999f, 0.0f, false);
    }

    public AdamOptimizer(float lr, float b1, float b2, float l2gamma) {
        this(lr, 1.0f, b1, b2, l2gamma);
    }

    public AdamOptimizer(float lr, float decay) {
        this(lr, decay, 0.9f, 0.999f, 0.0f, false);
    }



    //adam optimizer
    //uses momentum and rmsprop for individualized learing rates
    //needs previous grad

    
    @Override
public Tensor[] update(Tensor[] weights, Tensor[] prevGrads, Tensor[] grads, int batchSize) {
    //mean estimate update equation: m_t = b_1 * m_t_prev + (1 - b_1) * loss w.r.t this param
    //variance update estimate update equation: v_t = b_2 * v_t_prev + (1 - b_2) * (loss w.r.t this param) ^ 2
    //updated 0 bias with division by 1/(1-b_n^2)
    //final param update:  param = param - m_t / (sqrt(v_t) + eps) * lr

    //grads holds the loss w.r.t each param
    //prev grads holds m_t_prev and v_t_prev for each weight (so x2 the length)
    boolean firstTimeStep = prevGrads == null;
    if (firstTimeStep) {
        prevGrads = new Tensor[2 * weights.length];
    }
    
    for (int i = 0; i < weights.length; i++) {
        Tensor m, v;
        if (firstTimeStep) {
            m = new Tensor(grads[i]);
            m.mul(1 - b1);
            v = new Tensor(grads[i]);
            v.square();
            v.mul(1 - b2);
        } else {
            // m = b1 * m_prev + (1 - b1) * grad
            m = new Tensor(grads[i]);
            m.mul(1 - b1);
            prevGrads[2*i + 0].mul(b1);
            m.add(prevGrads[2*i + 0]);

            // v = b2 * v_prev + (1 - b2) * grad^2
            Tensor g2 = new Tensor(grads[i]);
            g2.square();
            g2.mul(1 - b2);
            prevGrads[2*i + 1].mul(b2);
            g2.add(prevGrads[2*i + 1]);

            v = g2;
        }

        //bias correct into mhat and vhat
        Tensor mHat = new Tensor(m);
        mHat.mul(1.0f / (1 - (float)Math.pow(b1, count)));
        Tensor vHat = new Tensor(v);
        vHat.mul(1.0f / (1 - (float)Math.pow(b2, count)));

        //apply adam update
        Tensor update = new Tensor(mHat);
        update.normalize(vHat, 1e-6f);
        update.mul(super.getLR() / batchSize);

        //l2 regularization
        if(l2 && ParameterType.l2Applicable(weights[i].type)) {
            weights[i].mul((1.0f - decayedLR * l2Gamma));
        }
        //gradient descent/ascent
        if(!invertDirection) {
            weights[i].sub(update);
        }else {
            weights[i].add(update);
        }
        
        //store uncorrected
        prevGrads[2*i + 0] = m;
        prevGrads[2*i + 1] = v;
    }

    return prevGrads;
}

}

