package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public class Relu extends UnaryFunction implements DoubleUnaryOperator
{
    public Relu(Value operand)
    {
        super("relu", operand);
        this.data = applyAsDouble(operand.doubleValue());
    }

    @Override
    public double applyAsDouble(double operand)
    {
        return Math.max(0, operand);
    }

    @Override
    protected double derivative()
    {
        return this.data.doubleValue() < 0 ? 0.0 : 1.0;
    }

}
