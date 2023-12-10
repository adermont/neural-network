package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.Neuron;
import com.github.adermont.neuralnetwork.math.DerivableFunction;

import java.util.Arrays;

public class DenseLayer extends NeuralLayer
{

    public DenseLayer(int nbNeurons, DerivableFunction pFunction)
    {
        super(nbNeurons, pFunction);
    }

    @Override
    public void connect(NeuralLayer pPreviousLayer)
    {
        super.connect(pPreviousLayer);

        // Pour une couche dense, chaque neurone est connecté à tous les neurones de la couche précédente.
        Neuron[] inputs = pPreviousLayer.getNeurons();

        // Les poids sont initialisés à une valeur aléatoire dans le constructeur du neurone
        double[] weights = new double[inputs.length];

        Arrays.stream(getNeurons()).forEach(n -> {
            n.setInputs(inputs);
            n.setWeights(weights);
        });
    }

}
