package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public class Tanh implements DerivableFunction
{
    @Override
    public double applyAsDouble(double x)
    {
        return StrictMath.tanh(x);
//        double exp2x = StrictMath.exp(2 * x);
//        return (exp2x - 1) / (exp2x + 1);
    }

    @Override
    public DoubleUnaryOperator derivative()
    {
        return x -> {
            return 1.0 - x * x;
        };
    }
}