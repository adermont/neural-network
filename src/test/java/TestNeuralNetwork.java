import com.github.adermont.neuralnetwork.base.NeuralNetwork;
import com.github.adermont.neuralnetwork.layer.CollectorLayer;
import com.github.adermont.neuralnetwork.layer.DenseLayer;
import com.github.adermont.neuralnetwork.math.DerivableFunction;
import com.github.adermont.neuralnetwork.math.Value;
import com.github.adermont.neuralnetwork.util.NNUtil;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestNeuralNetwork
{
    @Test
    public void dense1()
    {
        double[] x = NNUtil.rangeX(0, 2, 3);
        double[][] weights = {{1}, {-0.5}};
        double[] bias = {-1, 1};

        NeuralNetwork nn = new NeuralNetwork(1);
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

        NeuralNetwork nn = new NeuralNetwork(1);

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
        NeuralNetwork nn = new NeuralNetwork(3);

        // Layer 1 : 4 neurons
        DenseLayer layer1 = nn.addDenseLayer(4, DerivableFunction.TANH);
        // Layer 2 : 4 neurones
        DenseLayer layer2 = nn.addDenseLayer(4, DerivableFunction.TANH);
        // Layer 3 : 1 neurone
        CollectorLayer collectorLayer = nn.setCollectorLayer(1, DerivableFunction.TANH);

        double[][] x = {
                {2, 3, -1}, //
                {3, -1, 0.5}, //
                {0.5, 1, 1}, //
                {1, 1, -1},//
        };
        double[][] y = {{1}, {-1}, {-1}, {1}};

        List<Double> lossPlot = new ArrayList<>();

        final int NB_BATCH = 10000;
        final int NB_EPOCH = 10;

        // Lancement des epoch
        for (int iEpoch = 0; iEpoch < NB_EPOCH; iEpoch++)
        {
            NNUtil.shuffle(x, y);

            // Lancement des batchs
            for (int iBatch = 0; iBatch < NB_BATCH; iBatch++)
            {
                nn.nextBatch(NB_BATCH);
                Value loss = null;
                for (int xi = 0; xi < x.length; xi++)
                {
                    loss = nn.learn(x[xi], y[xi]);
                }

                // Cumul d'erreur à la fin du batch et rétropropagation
                //                System.out.printf("Total loss after batch %d: %.4f%n", iBatch, loss.doubleValue());
                lossPlot.add(loss.doubleValue());

                nn.updateWeights(0.01, false);
            }

            double erreurMoyenneEpoch = lossPlot.stream().reduce(0.0, (v, w) -> v + w)
                                                .doubleValue() / lossPlot.size();
            System.out.printf("########## EPOCH %d, ", iEpoch);
            System.out.println("Erreur moyenne par batch sur l'epoch: " + erreurMoyenneEpoch);
            lossPlot.clear();
        }

        System.out.println("---------------------");
        System.out.println("Apprentissage terminé");
        System.out.println("---------------------");
        for (int xi = 0; xi < x.length; xi++)
        {
            Value[] p = nn.predict(x[xi]);
            double[] e = y[xi];
            double l = Math.abs(e[0] - p[0].doubleValue()) * 100 / 2;
            System.out.printf("Prediction %d : %.5f (expected=%f, ecart=%.1f%%)%n", xi,
                              p[0].doubleValue(), e[0], l);
        }
    }

    public static void main(String[] args)
    {
        new TestNeuralNetwork().dense3();
    }
}
