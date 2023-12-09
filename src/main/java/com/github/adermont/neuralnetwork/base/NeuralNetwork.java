package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.layer.CollectorLayer;
import com.github.adermont.neuralnetwork.layer.DataLayer;
import com.github.adermont.neuralnetwork.layer.DenseLayer;
import com.github.adermont.neuralnetwork.layer.NeuralLayer;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.Optional;
import java.util.function.DoubleUnaryOperator;

public class NeuralNetwork
{
    private double[]           inputValues;
    private double[][]         outputValues;
    private DataLayer          inputLayer;
    private Deque<NeuralLayer> neuralLayers;
    private CollectorLayer     outputLayer;

    public NeuralNetwork(int inputDim)
    {
        neuralLayers = new ArrayDeque<>();

        double[] firstLayerValues = new double[inputDim];
        inputLayer = new DataLayer(firstLayerValues);
        Optional.ofNullable(neuralLayers.peekFirst()).ifPresent(dense -> dense.connect(inputLayer));
    }

    protected void setInput(double[] input)
    {
        inputValues = input;
    }

    public DenseLayer addDenseLayer(int nbNeurons, ActivationFunction pFunction)
    {
        DenseLayer layer = new DenseLayer(nbNeurons, pFunction);
        Optional.ofNullable(neuralLayers.peekLast())
                .ifPresentOrElse(dense -> layer.connect(dense), () -> layer.connect(inputLayer));

        neuralLayers.add(layer);
        return layer;
    }

    public CollectorLayer setCollectorLayer(int nbNeurons, ActivationFunction pFunction)
    {
        CollectorLayer collector = new CollectorLayer(nbNeurons, pFunction);
        Optional.ofNullable(neuralLayers.peekLast())
                .ifPresentOrElse(dense -> collector.connect(dense),
                                 () -> collector.connect(inputLayer));
        outputLayer = collector;
        return collector;
    }

    public void predict(double[] pX)
    {
        setInput(pX);

        double[][] output = new double[inputValues.length][];
        int iOut = 0;
        for (double inputValue : inputValues)
        {
            for (Neuron neuron : inputLayer.getNeurons())
            {
                ((DataNeuron)neuron).setData(new double[]{inputValue});
            }
            inputLayer.propagate();
            output[iOut++] = outputLayer.getOutput().get();
        }

        this.outputValues = output;

    }

    public double[] getFlattenOutput()
    {
        if (outputLayer.getNeurons().length > 1)
        {
            throw new UnsupportedOperationException("Flatten only works with collector layer having only 1 neuron");
        }
        return Arrays.stream(outputValues)
              .flatMapToDouble(Arrays::stream)
              .toArray();
    }

    public double[][] getOutput()
    {
        return outputValues;
    }
}
