package com.github.adermont.neuralnetwork.base;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

@FunctionalInterface
public interface ActivationFunction extends DoubleUnaryOperator
{
    public static final ActivationFunction IDENTITY  = x -> x;

    public static final ActivationFunction HEAVISIDE = x -> x >= 0 ? 1.0 : 0.0;

    public static final ActivationFunction RELU      = x -> x >= 0 ? x : 0.0;

    public static final ActivationFunction SIGMA     = x -> {
        return 1.0 / (1 + StrictMath.exp(-x));
    };

    public static final ActivationFunction TANH      = x -> {
        return (StrictMath.exp(x) - StrictMath.exp(-x)) / (StrictMath.exp(x) + StrictMath.exp(-x));
    };

}
