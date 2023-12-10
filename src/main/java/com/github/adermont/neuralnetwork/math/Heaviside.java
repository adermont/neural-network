package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public class Heaviside implements DerivableFunction
{
    @Override
    public double applyAsDouble(double operand)
    {
        return operand < 0.0 ? 0.0 : 1.0;
    }

    @Override
    public DoubleUnaryOperator derivative()
    {
        return x -> 0.0;
    }
}