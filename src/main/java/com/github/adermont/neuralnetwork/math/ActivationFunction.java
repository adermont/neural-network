package com.github.adermont.neuralnetwork.math;

public class ActivationFunction extends UnaryFunction
{
    public ActivationFunction(Value operand, DerivableFunction delegate)
    {
        super(delegate.name(), operand, delegate);
    }

    @Override
    protected void backward()
    {
        operand.addGradient(function.derivative().applyAsDouble(data.doubleValue()) * this.grad);
    }
}
