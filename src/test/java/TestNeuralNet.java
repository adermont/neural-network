import com.github.adermont.neuralnetwork.base.NeuralNet;
import com.github.adermont.neuralnetwork.base.NeuralNetListener;
import com.github.adermont.neuralnetwork.layer.CollectorLayer;
import com.github.adermont.neuralnetwork.layer.DenseLayer;
import com.github.adermont.neuralnetwork.math.DerivableFunction;
import com.github.adermont.neuralnetwork.math.Value;
import com.github.adermont.neuralnetwork.util.NNUtil;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

public class TestNeuralNet
{
    @Test
    public void dense1()
    {
        double[] x = NNUtil.rangeX(0, 2, 3);
        double[][] weights = {{1}, {-0.5}};
        double[] bias = {-1, 1};

        NeuralNet nn = new NeuralNet(1, 0.01);
        CollectorLayer collectorLayer = nn.setCollectorLayer(1, DerivableFunction.RELU);
        collectorLayer.setWeights(weights);
        collectorLayer.setBias(bias);
        nn.predict(x);

        NNUtil.plot("1 NEURONE", x,
                    Arrays.stream(nn.getOutput()).mapToDouble(Value::doubleValue).toArray());
    }

    @Test
    public void dense2()
    {

        NeuralNet nn = new NeuralNet(1, 0.01);

        // Layer 1
        double[][] weights = {{1}, {-0.5}};
        double[] bias = {-1, 1};

        DenseLayer denseLayer = nn.addDenseLayer(2, DerivableFunction.RELU);
        denseLayer.setWeights(weights);
        denseLayer.setBias(bias);

        // Layer 2
        double[][] weights2 = {{1, 1}};
        double[] bias2 = {0};

        CollectorLayer collectorLayer = nn.setCollectorLayer(1, DerivableFunction.RELU);
        collectorLayer.setWeights(weights2);
        collectorLayer.setBias(bias2);

        double[] x = NNUtil.rangeX(-2, 3, 30);
        nn.predict(x);

        NNUtil.plot("3 NEURONES", x,
                    Arrays.stream(nn.getOutput()).mapToDouble(Value::doubleValue).toArray());
    }

    public void dense3()
    {
        // Training Data
        double[][] x_train = {
                {2, 3, -1}, //
                {3, -1, 0.5}, //
                {0.5, 1, 1}, //
                {1, 1, -1},//
        };
        // Training output
        double[][] y_train = {{1}, {-1}, {-1}, {1}};

        NeuralNet nn = new NeuralNet(3, 0.1);
        nn.addListener(new NeuralNetListener());

        // Layer 1 : 4 neurons
        DenseLayer layer1 = nn.addDenseLayer(4, DerivableFunction.TANH);
        // Layer 2 : 4 neurones
        DenseLayer layer2 = nn.addDenseLayer(4, DerivableFunction.TANH);
        // Collector layer : 1 neurone with TANH ==> regression between 0..1
        CollectorLayer collectorLayer = nn.setCollectorLayer(1, DerivableFunction.TANH);

        nn.summary();

        final int NB_EPOCH = 1000;
        nn.fit(x_train, y_train, NB_EPOCH, 1.0);

        System.out.println("---------------------");
        System.out.println("Apprentissage terminé");
        System.out.println("---------------------");
        for (int xi = 0; xi < x_train.length; xi++)
        {
            Value[] p = nn.predict(x_train[xi]);
            double[] e = y_train[xi];
            double l = Math.abs(e[0] - p[0].doubleValue()) * 100 / 2;
            System.out.printf("Prediction %d : %.4f (expected=%.0f, ecart=%.1f%%)%n", xi,
                              p[0].doubleValue(), e[0], l);
        }
    }

    public static void main(String[] args)
    {
        new TestNeuralNet().dense3();
    }
}
