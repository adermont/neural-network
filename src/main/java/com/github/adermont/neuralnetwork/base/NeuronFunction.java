package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.math.Value;

@FunctionalInterface
public interface NeuronFunction
{
    Value apply(Neuron n);
}
