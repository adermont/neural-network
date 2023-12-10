package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.layer.CollectorLayer;
import com.github.adermont.neuralnetwork.layer.DataLayer;
import com.github.adermont.neuralnetwork.layer.DenseLayer;
import com.github.adermont.neuralnetwork.layer.NeuralLayer;
import com.github.adermont.neuralnetwork.math.DerivableFunction;
import com.github.adermont.neuralnetwork.math.Value;

import java.util.*;

public class NeuralNetwork
{
    private int                inputDim;
    private Value[]            output;
    private DataLayer          inputLayer;
    private Deque<NeuralLayer> hiddenLayers;
    private CollectorLayer     outputLayer;

    private Value lossOfCurrentBatch;
    private int   currentBatch; // numero de batch courant
    private int   maxBatch; // Nombre max de batches

    public NeuralNetwork(int inputDim)
    {
        this.inputDim = inputDim;
        this.hiddenLayers = new ArrayDeque<>();
        this.lossOfCurrentBatch = new Value("loss", 0);

        double[] firstLayerValues = new double[inputDim];
        this.inputLayer = new DataLayer(firstLayerValues);
        Optional.ofNullable(hiddenLayers.peekFirst()).ifPresent(dense -> dense.connect(inputLayer));
    }

    private void initLayer(NeuralLayer dense)
    {
        NeuralLayer previous = null;
        if (hiddenLayers.peekLast() == null)
        {
            previous = inputLayer;
        }
        else
        {
            previous = hiddenLayers.peekLast();
        }
        dense.connect(previous);

        double[][] weights = new double[dense.getNeuronCount()][];
        for (int i = 0; i < weights.length; i++)
        {
            weights[i] = new Random().doubles(previous.getNeuronCount(), -1, 1).toArray();
        }
        dense.setWeights(weights);
        double[] bias = new double[dense.getNeuronCount()];
        for (int i = 0; i < bias.length; i++)
        {
            bias[i] = new Random().nextDouble(-1, 1);
        }
        dense.setBias(bias);
    }

    public DenseLayer addDenseLayer(int nbNeurons, DerivableFunction pFunction)
    {
        DenseLayer dense = new DenseLayer(nbNeurons, pFunction);
        initLayer(dense);
        hiddenLayers.add(dense);
        return dense;
    }

    public CollectorLayer setCollectorLayer(int nbNeurons, DerivableFunction pFunction)
    {
        CollectorLayer collector = new CollectorLayer(nbNeurons, pFunction);
        initLayer(collector);
        return outputLayer = collector;
    }

    public Value[] predict(double[] pX)
    {
        for (Neuron neuron : inputLayer.getNeurons())
        {
            ((DataNeuron) neuron).setData(pX);
        }
        inputLayer.propagate();
        return this.output = outputLayer.getOutput();
    }

    public void nextBatch(int pNB_BATCH)
    {
        this.currentBatch++;
        this.maxBatch = pNB_BATCH;
        this.lossOfCurrentBatch = new Value("loss@" + currentBatch, 0);
        this.outputLayer.resetGradients();
    }

    public Value learn(double[] pX, double[] pY)
    {
        Value[] prediction = predict(pX);
        for (int i = 0; i < pY.length; i++)
        {
            String sBatch = "batch" + currentBatch;
            Value expected = new Value(sBatch + ".xpct" + i, pY[i]);
            Value diff = expected.minus(prediction[i]).label(sBatch + ".diff" + i);
            lossOfCurrentBatch = lossOfCurrentBatch.plus(diff.pow(2)).label(sBatch + ".loss" + i);
        }
        return lossOfCurrentBatch;
    }

    public Value[] getOutput()
    {
        return output;
    }

    public List<Value> parameters()
    {
        List<Value> result = new ArrayList<>();
        hiddenLayers.forEach(layer -> result.addAll(layer.parameters()));
        result.addAll(outputLayer.parameters());
        return result;
    }

    public void updateWeights()
    {
        updateWeights(0.01, true);
    }

    public void updateWeights(double pStep, boolean optimize)
    {
        if (optimize)
        {
            pStep = 1.0 - 0.9 * currentBatch / maxBatch;
            //System.out.println("- Updating weights, learning rate=" + pStep);
        }
        lossOfCurrentBatch.retroPropagate();
        List<Value> parameters = parameters();
        for (Value parameter : parameters)
        {
            parameter.update(pStep);
        }
    }
}
