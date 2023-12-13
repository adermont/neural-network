package com.github.adermont.neuralnetwork.math;

import java.util.function.BinaryOperator;

public interface LossFunction extends BinaryOperator<Value>
{
    /**
     * Mean Squared Error : (y - score).pow(2)
     */
    LossFunction MSE = (expect, score) -> expect.minus(score).pow(2).label("diff_loss()");

    /**
     * SVM max-margin : (1 - y * score).relu()
     * <p>
     * This function is commonly used with a regularization loss function that is :
     * <pre>
     *     reg = sum(0.0001 * p*p) for p in model.parameters()
     * </pre>
     */
    LossFunction MAX_MARGIN = (expect, score) -> new Value(1).minus(expect.mul(score)).relu()
                                                             .label("maxmargin_loss()");
}
