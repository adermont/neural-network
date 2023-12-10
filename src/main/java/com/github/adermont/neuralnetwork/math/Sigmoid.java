package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public class Sigmoid implements DerivableFunction
{
    @Override
    public double applyAsDouble(double x)
    {
         return 1.0 / (1 + StrictMath.exp(-x));
    }

    @Override
    public DoubleUnaryOperator derivative()
    {
        return x -> {
            double sigma = 1.0 / (1 + StrictMath.exp(-x));
            return sigma * (1 - sigma);
        };
    }
}
