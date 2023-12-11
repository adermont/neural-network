package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public class Tanh implements DerivableFunction
{
    @Override
    public double applyAsDouble(double x)
    {
        return StrictMath.tanh(x);
    }

    @Override
    public DoubleUnaryOperator derivative()
    {
        return x -> {
            return 1.0 - x * x;
        };
    }
}