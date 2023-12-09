package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.ActivationFunction;
import com.github.adermont.neuralnetwork.base.Neuron;

import java.util.Arrays;
import java.util.Optional;
import java.util.function.DoubleUnaryOperator;

public class CollectorLayer extends DenseLayer
{
    double[] output;

    public CollectorLayer(int nbNeurons, ActivationFunction pFunction)
    {
        super(nbNeurons, pFunction);
    }

    @Override
    protected Neuron createNeuron(int id, ActivationFunction pFunction)
    {
        Neuron neuron = super.createNeuron(id, pFunction);
        return neuron;
    }

    @Override
    public void propagate()
    {
        output = Arrays.stream(getNeurons()).mapToDouble(Neuron::getOutputAsDouble).toArray();
    }

    public Optional<double[]> getOutput(){
        return Optional.of(output);
    }
}
