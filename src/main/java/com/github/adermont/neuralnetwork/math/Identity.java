package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public class Identity implements DerivableFunction
{
    @Override
    public double applyAsDouble(double operand)
    {
        return operand;
    }

    @Override
    public DoubleUnaryOperator derivative()
    {
        return x -> 1.0;
    }
}