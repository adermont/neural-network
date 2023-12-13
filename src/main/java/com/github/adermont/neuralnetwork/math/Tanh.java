package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public class Tanh extends UnaryFunction implements DoubleUnaryOperator
{
    public Tanh(Value operand)
    {
        super("tanh", operand);
        this.data = applyAsDouble(operand.doubleValue());
    }

    @Override
    public double applyAsDouble(double operand)
    {
        return StrictMath.tanh(operand);
    }

    @Override
    protected double derivative()
    {
        return 1 - (this.data.doubleValue() * this.data.doubleValue());
    }

}
