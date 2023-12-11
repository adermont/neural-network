package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public interface DerivableFunction extends DoubleUnaryOperator
{
    DerivableFunction IDENTITY = new Identity();

    DerivableFunction HEAVISIDE = new Heaviside();

    DerivableFunction RELU = new Relu();

    DerivableFunction SIGMA = new Sigmoid();

    DerivableFunction TANH = new Tanh();

    DoubleUnaryOperator derivative();

    default String name()
    {
        return getClass().getSimpleName().toLowerCase();
    }
}
