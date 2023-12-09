package com.github.adermont.neuralnetwork.math;

public class ReLU extends UnaryFunction
{
    public ReLU(Value operand)
    {
        super("relu", operand);
    }

    @Override
    public double applyAsDouble(double operand)
    {
        return Math.max(0, operand);
    }

    @Override
    public void backward()
    {
        self.grad += (this.data.doubleValue() > 0 ? 1.0 : 0.0) * this.grad;
    }
}
