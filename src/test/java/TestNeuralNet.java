import com.github.adermont.neuralnetwork.base.NeuralNet;
import com.github.adermont.neuralnetwork.base.NeuralNetListener;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.dataset.mnist.MnistDataset;
import com.github.adermont.neuralnetwork.layer.CollectorLayer;
import com.github.adermont.neuralnetwork.layer.DenseLayer;
import com.github.adermont.neuralnetwork.math.Matrix;
import com.github.adermont.neuralnetwork.math.Value;
import com.github.adermont.neuralnetwork.util.NNUtil;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

public class TestNeuralNet
{
    @Test
    public void dense1()
    {
        double[][] x = NNUtil.range2dim(0, 2, 3);
        double[][] weights = {{1}, {-0.5}};
        double[] bias = {-1, 1};

        NeuralNet nn = new NeuralNet(1, 0.01);
        CollectorLayer collectorLayer = nn.setCollectorLayer(1, NeuronFunctions.RELU);
        collectorLayer.setWeights(weights);
        collectorLayer.setBias(bias);

        Value[][] y = new Value[x.length][];
        for (int i = 0; i < x.length; i++)
        {
            y[i] = nn.predict(x[i]);
        }

        double[] a = Arrays.stream(x).mapToDouble(array -> array[0]).toArray();
        double[] b = Arrays.stream(y).mapToDouble(array -> array[0].doubleValue()).toArray();
        NNUtil.plot("1 NEURONE", a, b);
    }

    @Test
    public void dense2()
    {
        NeuralNet nn = new NeuralNet(1, 0.01);

        // Layer 1
        double[][] weights = {{1}, {-0.5}};
        double[] bias = {-1, 1};

        DenseLayer denseLayer = nn.addDenseLayer(2, NeuronFunctions.RELU);
        denseLayer.setWeights(weights);
        denseLayer.setBias(bias);

        // Layer 2
        double[][] weights2 = {{1, 1}};
        double[] bias2 = {0};

        CollectorLayer collectorLayer = nn.setCollectorLayer(1, NeuronFunctions.RELU);
        collectorLayer.setWeights(weights2);
        collectorLayer.setBias(bias2);

        double[][] x = NNUtil.range2dim(-2, 3, 30);
        Value[][] y = new Value[x.length][];
        for (int i = 0; i < x.length; i++)
        {
            y[i] = nn.predict(x[i]);
        }

        double[] a = Arrays.stream(x).mapToDouble(array -> array[0]).toArray();
        double[] b = Arrays.stream(y).mapToDouble(array -> array[0].doubleValue()).toArray();
        NNUtil.plot("3 NEURONES", a, b);
    }

    public void dense3()
    {
        // Training inputs
        double[][] x_train = {
                {2, 3, -1}, //1
                {2, 2, 0}, //2
                {3, -1, 0.5}, //3
                {3., -2, 0.6}, //4
                {0.5, 0.5, -0.5}, //5
                {0.5, 1, 1}, //6
                {1, 1, -1},//7
                {-1, 1.5, -1},//8
                {-1, 2, 3},//9
                {-3, 1.5, 0.5},//10
        };
        // Training outputs
        double[][] y_train = {{1}, {-1}, {-1}, {1}, {1}, {1}, {-1}, {-1}, {1}, {-1}};

        NeuralNet nn = new NeuralNet(3, 0.1);

        // Layer 1 : 4 neurons
        DenseLayer layer1 = nn.addDenseLayer(4, NeuronFunctions.TANH);
        // Layer 2 : 4 neurones
        DenseLayer layer2 = nn.addDenseLayer(4, NeuronFunctions.TANH);
        // Collector layer : 1 neurone with TANH ==> regression between 0..1
        CollectorLayer collectorLayer = nn.setCollectorLayer(1, NeuronFunctions.TANH);

        // For statistics
        NeuralNetListener listener = new NeuralNetListener(nn);
        nn.addListener(listener);

        // Display graphs at end of training
        listener.displayParam().displayAccuracy();

        nn.summary();

        final int NB_EPOCH = 5000;
        nn.fit(x_train, y_train, NB_EPOCH, 100);

        System.out.println("----------------------------------------------");
        System.out.println("Training terminated:");
        System.out.println("----------------------------------------------");
        for (int xi = 0; xi < x_train.length; xi++)
        {
            Value[] p = nn.predict(x_train[xi]);
            double[] e = y_train[xi];
            double l = Math.abs(e[0] - p[0].doubleValue()) * 100 / 2;
            System.out.printf("Prediction %d[%s] : %.4f (expected=%.0f, ecart=%.1f%%)%n", xi,
                              Arrays.toString(x_train[xi]), p[0].doubleValue(), e[0], l);
        }
    }

    public void mnist() throws IOException
    {
        MnistDataset mnistDataset = new MnistDataset();
        Matrix[] matrices = mnistDataset.loadFromLocal(Path.of("D:", "Datasets", "mnist"));
        double[][] x_train = mnistDataset.xtrain(5000);
        double[][] y_train = mnistDataset.ytrain(5000);

        NeuralNet nn = new NeuralNet(x_train[0].length, 0.1);

        // Layer 1 : 100 neurons
        nn.addDenseLayer(20, NeuronFunctions.RELU);
        nn.addDenseLayer(20, NeuronFunctions.RELU);
        CollectorLayer collectorLayer = nn.setCollectorLayer(10, NeuronFunctions.SOFTMAX);

        // For statistics
        NeuralNetListener listener = new NeuralNetListener(nn);
        listener.logDebug();
        nn.addListener(listener);

        // Display graphs at end of training
        listener.displayAccuracy();

        nn.summary();

        final int NB_EPOCH = 2;
        nn.fit(x_train, y_train, NB_EPOCH, 50);

        System.out.println("----------------------------------------------");
        System.out.println("Training terminated:");
        System.out.println("----------------------------------------------");
        for (int xi = 0; xi < x_train.length; xi++)
        {
            Value[] p = nn.predict(x_train[xi]);
            int classification = collectorLayer.getOutputClass();

            double[] e = y_train[xi];
            int classPred = collectorLayer.getClass(e);

            int l = Math.abs(classification - classPred) * 10;
            System.out.printf("Prediction %d : %d (expected=%d, ecart=%d%%)%n", xi, classPred,
                              classification, l);
        }
    }

    public static void main(String[] args) throws IOException
    {
        new TestNeuralNet().mnist();
    }
}
