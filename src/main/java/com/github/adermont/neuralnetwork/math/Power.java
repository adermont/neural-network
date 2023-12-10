package com.github.adermont.neuralnetwork.math;

public class Power extends BinaryFunction
{
    public Power(Value value, Value exponent)
    {
        super("^%d".formatted((int) exponent.data.intValue()), value, exponent);
    }

    @Override
    public double applyAsDouble(double left, double right)
    {
        return Math.pow(left, right);
    }

    public void backward()
    {
        self.grad += derivative().doubleValue() * this.grad;
    }

    /**
     * Dérivée de X^n = n.X^(n-1)
     *
     * @return
     */
    public Number derivative()
    {
        return (other.data.doubleValue() * StrictMath.pow(self.data.doubleValue(),
                                                          other.data.doubleValue() - 1));
    }

}
