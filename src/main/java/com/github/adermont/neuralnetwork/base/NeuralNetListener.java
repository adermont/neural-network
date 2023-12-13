package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.math.Value;
import com.github.adermont.neuralnetwork.util.NNUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NeuralNetListener
{
    protected static Logger logger = Logger.getLogger(NeuralNetListener.class.getName());

    static
    {
        Handler consoleHandler = new ConsoleHandler();
        consoleHandler.setLevel(Level.FINE);
        consoleHandler.setFormatter(new Formatter()
        {
            @Override
            public String format(LogRecord record)
            {
                return "%s%n".formatted(record.getMessage());
            }
        });
        logger.setUseParentHandlers(false);
        logger.addHandler(consoleHandler);
        logger.setLevel(Level.ALL);
    }

    private NeuralNet neuralNet;

    private int epochCount;
    private int currentEpoch;
    private int batchCount;
    private int currentBatch;
    private int batchSize;
    private int batchStartIndex;

    private List<Double> errorsByEpoch;
    private List<Double> accuracyByBatch;
    private List<Double> accuracyByEpoch;
    private List<Double> learningRateEvolution;

    // For parameters comparison
    private List<Double> parametersAtStart;
    private List<Double> parametersAtEnd;

    private boolean isDisplayParameters;
    private boolean isDisplayAccuracy;
    private boolean isDisplayLearningRate;

    public NeuralNetListener(NeuralNet nn)
    {
        neuralNet = nn;
        errorsByEpoch = new ArrayList<>();
        accuracyByBatch = new ArrayList<>();
        accuracyByEpoch = new ArrayList<>();
        learningRateEvolution = new ArrayList<>();
        parametersAtStart = new ArrayList<>();
        parametersAtEnd = new ArrayList<>();
    }

    /**
     * Performs reduction by making an average every 'data.size()/size' samples. As a consequence,
     * the result list will contain only 'size' samples.
     *
     * @param size The size of the desired output.
     * @param data The list to reduce.
     * @return A reduced list containing only 'size' samples, whose are average values of a batch of
     * samples.
     */
    public static List<Double> averageReduce(int size, List<Double> data)
    {
        if ( data.size() > size)
        {
            final int sampleLength = Math.max(data.size(), data.size() / size);

            return IntStream.rangeClosed(0, Math.min(data.size() / size, size)).mapToObj(
                                    i -> data.subList(i * sampleLength, i * sampleLength + sampleLength - 1).stream().mapToDouble(Double::doubleValue).average().orElse(0.0))
                            .collect(Collectors.toList());
        } else {
            return data;
        }
    }

    public void logInfo()
    {
        logger.setLevel(Level.INFO);
    }

    public void logDebug()
    {
        logger.setLevel(Level.ALL);
    }

    public NeuralNetListener displayParam()
    {
        isDisplayParameters = true;
        return this;
    }

    public NeuralNetListener displayAccuracy()
    {
        isDisplayAccuracy = true;
        return this;
    }

    public NeuralNetListener displayLearningRate()
    {
        isDisplayLearningRate = true;
        return this;
    }

    public void epochStarted(int pCurrentEpoch, int pEpochCount)
    {
        epochCount = pEpochCount;
        currentEpoch = pCurrentEpoch;
        batchCount = 0;
        batchSize = 0;
        batchStartIndex = 0;
    }

    public void epochTerminated(int pCurrentEpoch)
    {
        // Mean Squared Error
        double mse = errorsByEpoch.stream().reduce(0.0, (v, w) -> v + w)
                                  .doubleValue() / errorsByEpoch.size();

        // Root Mean Squared Error
        double rmse = StrictMath.sqrt(mse);

        logger.log(Level.INFO,
                   "EPOCH %d/%d : mse=%.5f rmse=%.5f".formatted(pCurrentEpoch, epochCount, mse,
                                                                rmse));

        accuracyByEpoch.add(accuracyByBatch.stream().reduce(0.0, (a, b) -> a + b)
                                           .doubleValue() / accuracyByBatch.size());
        accuracyByBatch.clear();
    }

    public void batchStarted(int pCurrentBatch, int pBatchCount, int pBatchSize,
                             int pBatchStartIndex)
    {
        currentBatch = pCurrentBatch;
        batchCount = pBatchCount;
        batchSize = pBatchSize;
        batchStartIndex = pBatchStartIndex;
        //logger.log(Level.FINE, "Starting batch "+pCurrentBatch+"/"+batchCount);
    }

    public void batchTerminated(int pCurrentBatch, Batch pBatch)
    {
        logger.log(Level.FINE,
                   "BATCH %d/%d : mse=%.5f accuracy=%.6f%%".formatted(pCurrentBatch, batchCount,
                                                                 pBatch.loss().doubleValue(),
                                                                 pBatch.accuracy() * 100));
        accuracyByBatch.add(pBatch.accuracy());
        errorsByEpoch.add(pBatch.loss().doubleValue());
    }

    public void trainingStarted()
    {
        logger.log(Level.INFO, "Starting training...");

        errorsByEpoch.clear();
        accuracyByBatch.clear();
        accuracyByEpoch.clear();
        learningRateEvolution.clear();
        parametersAtStart.clear();
        parametersAtEnd.clear();

        if (isDisplayParameters)
        {
            List<Value> parameters = neuralNet.parameters();
            parameters.stream().mapToDouble(Value::doubleValue)
                      .forEachOrdered(v -> parametersAtStart.add(v));
        }
    }

    public void trainingTerminated()
    {
        if (isDisplayParameters)
        {
            List<Value> parameters = neuralNet.parameters();
            parameters.stream().mapToDouble(Value::doubleValue)
                      .forEachOrdered(v -> parametersAtEnd.add(v));

            NNUtil.plot("parameters", parametersAtStart, parametersAtEnd);
        }
        if (isDisplayAccuracy)
        {
            NNUtil.plot("accuracy", averageReduce(500, accuracyByEpoch));
        }
        if (isDisplayLearningRate)
        {
            NNUtil.plot("learningRate", averageReduce(100, learningRateEvolution));
        }
    }

    public void learningRateChanged(double pLearningRate)
    {
        logger.log(Level.FINER, "Learning rate modified to %.5f".formatted(pLearningRate));
        learningRateEvolution.add(pLearningRate);
    }
}
