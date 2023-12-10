package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public class Relu implements DerivableFunction
{
    @Override
    public double applyAsDouble(double operand)
    {
        return Math.max(0, operand);
    }

    @Override
    public DoubleUnaryOperator derivative()
    {
        return x -> x < 0.0 ? 0.0 : 1.0;
    }
}
