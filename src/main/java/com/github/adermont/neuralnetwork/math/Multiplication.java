package com.github.adermont.neuralnetwork.math;

public class Multiplication extends BinaryFunction
{
    public Multiplication(Value pLeftOperand, Value pRightOperand)
    {
        super("*", pLeftOperand, pRightOperand);
    }

    @Override
    public double applyAsDouble(double left, double right)
    {
        return left * right;
    }

    @Override
    protected void backward()
    {
        self.grad += other.data.doubleValue() * this.grad;
        other.grad += self.data.doubleValue() * this.grad;
    }

}
