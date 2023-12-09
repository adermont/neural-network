import com.github.adermont.neuralnetwork.base.ActivationFunction;
import com.github.adermont.neuralnetwork.base.NeuralNetwork;
import com.github.adermont.neuralnetwork.layer.CollectorLayer;
import com.github.adermont.neuralnetwork.layer.DenseLayer;
import com.github.adermont.neuralnetwork.util.NNUtil;
import org.junit.jupiter.api.Test;

public class TestNeuralNetwork
{
        @Test
        public void dense1(){
            double[] x = NNUtil.rangeX(0, 2, 3);
            double[][] weights = {{1}, {-0.5}};
            double[] bias = {-1, 1};

            NeuralNetwork nn = new NeuralNetwork(1);
            CollectorLayer collectorLayer = nn.setCollectorLayer(1, ActivationFunction.RELU);
            collectorLayer.setWeights(weights);
            collectorLayer.setBias(bias);
            nn.predict(x);

            NNUtil.plot("1 NEURONE", x, nn.getFlattenOutput());
        }

    @Test
    public void dense2()
    {

        NeuralNetwork nn = new NeuralNetwork(1);

        // Layer 1
        double[][] weights = {{1}, {-0.5}};
        double[] bias = {-1, 1};

        DenseLayer denseLayer = nn.addDenseLayer(2, ActivationFunction.RELU);
        denseLayer.setWeights(weights);
        denseLayer.setBias(bias);

        // Layer 2
        double[][] weights2 = {{1, 1}};
        double[] bias2 = {0};

        CollectorLayer collectorLayer = nn.setCollectorLayer(1, ActivationFunction.RELU);
        collectorLayer.setWeights(weights2);
        collectorLayer.setBias(bias2);

        double[] x = NNUtil.rangeX(-2, 3, 30);
        nn.predict(x);

        NNUtil.plot("3 NEURONES", x, nn.getFlattenOutput());
    }

    public static void main(String[] args)
    {
        new TestNeuralNetwork().dense2();
    }
}
