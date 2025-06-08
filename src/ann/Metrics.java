package ann;

//object that encapsulates the ann training metrics
//loss
//accuracy
public class Metrics {
    //epoch averages (really sums, until the end at which point they are divided)
    float loss;
    float accuracy;

    int batches;

    boolean doLoss;
    boolean doAccuracy;

    public Metrics(boolean doLoss, boolean doAccuracy) {
        this.doLoss = doLoss;
        this.doAccuracy = doAccuracy;

        loss = 0.0f;
        accuracy = 0.0f;
    }

    public boolean doLoss() {
        return doLoss;
    }

    public boolean doAccuracy() {
        return doAccuracy;
    }

    public void addLoss(float l) {
        this.loss += l;
    }

    public void setAccuracy(float acc) {
        this.accuracy = acc;
    }

    public void batchDone() {
        //signifies the batch is over (increments batch so average is kept track of properly)
        batches++;
    }

    public float getLoss() {
        return loss;
    }

    public float getAccuracy() {
        return accuracy;
    }

    //prints the averaged metrics in a nice way
    public void printMetrics() {
        //average the metrics
        loss /= batches;

        System.out.println("The Loss Average was " + loss + ".");
        System.out.println("Validation Accuracy was " + accuracy + "%.");
    }

}
