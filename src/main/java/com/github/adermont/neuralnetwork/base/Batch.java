package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.math.Value;

public record Batch(int batchSize, Value loss, double accuracy)
{
}
