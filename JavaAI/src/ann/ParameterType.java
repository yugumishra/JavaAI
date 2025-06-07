package ann;

public enum ParameterType {
    INVALID(-1), DENSE_WEIGHT(1), DENSE_BIAS(2), BN_GAMMA(3), BN_BETA(4), BN_MU(5), BN_SIGSQ(6);

    int id;

    private ParameterType(int id) {
        this.id = id;
    }

    public int get() {
        return id;
    }

    public static ParameterType get(int id) {
        switch(id) {
        case -1:
        return INVALID;
        case 1:
        return DENSE_WEIGHT;
        case 2:
        return DENSE_BIAS;
        case 3:
        return BN_GAMMA;
        case 4:
        return BN_BETA;
        case 5:
        return BN_MU;
        case 6:
        return BN_SIGSQ;
        default:
        return INVALID;
        }
    }

    public static boolean l2Applicable(ParameterType t) {
        return t == DENSE_WEIGHT;
    }
}
