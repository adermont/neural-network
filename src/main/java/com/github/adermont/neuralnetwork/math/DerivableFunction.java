package com.github.adermont.neuralnetwork.math;


import java.util.function.DoubleUnaryOperator;

public interface DerivableFunction extends DoubleUnaryOperator
{
    public static final DerivableFunction IDENTITY = new Identity();

    public static final DerivableFunction HEAVISIDE = new Heaviside();

    public static final DerivableFunction RELU = new Relu();

    public static final DerivableFunction SIGMA = new Sigmoid();

    public static final DerivableFunction TANH = new Tanh();

    DoubleUnaryOperator derivative();

    default String name(){
        return getClass().getSimpleName().toLowerCase();
    }
}
