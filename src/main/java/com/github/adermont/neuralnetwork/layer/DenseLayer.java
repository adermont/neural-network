package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.ActivationFunction;
import com.github.adermont.neuralnetwork.base.Neuron;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

public class DenseLayer extends NeuralLayer
{

    public DenseLayer(int nbNeurons, ActivationFunction pFunction)
    {
        super(nbNeurons, pFunction);
    }

    @Override
    public void connect(NeuralLayer pPreviousLayer)
    {
        super.connect(pPreviousLayer);

        // Pour une couche dense, chaque neurone est connecté à tous les neurones de la couche précédente.
        Neuron[] inputs = pPreviousLayer.getNeurons();
        double[] weights = new double[inputs.length];
        Arrays.fill(weights, 1);

        Arrays.stream(getNeurons()).forEach(n -> {
            n.setInputs(inputs);
            n.setWeights(weights);
        });
    }

}
