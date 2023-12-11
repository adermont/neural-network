package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.math.Value;

import java.util.List;
import java.util.logging.*;

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
        logger.setLevel(Level.INFO);
    }

    private int epochCount;
    private int currentEpoch;
    private int batchCount;
    private int currentBatch;
    private int batchSize;
    private int batchStartIndex;

    public void epochStarted(int pCurrentEpoch, int pEpochCount)
    {
        epochCount = pEpochCount;
        currentEpoch = pCurrentEpoch;
    }

    public void batchStarted(int pCurrentBatch, int pBatchCount, int pBatchSize,
                             int pBatchStartIndex)
    {
        currentBatch = pCurrentBatch;
        batchCount = pBatchCount;
        batchSize = pBatchSize;
        batchStartIndex = pBatchStartIndex;
    }

    public void epochTerminated(int pCurrentEpoch, List<Double> pErrorsByEpoch)
    {
        // Mean Squared Error
        double mse = pErrorsByEpoch.stream().reduce(0.0, (v, w) -> v + w)
                                   .doubleValue() / pErrorsByEpoch.size();

        // Root Mean Squared Error
        double rmse = StrictMath.sqrt(mse);

        logger.log(Level.INFO, "EPOCH %d/%d : mse=%.5f rmse=%.5f".formatted(pCurrentEpoch, epochCount, mse, rmse));
    }

    public void batchTerminated(int pCurrentBatch, Value pMse)
    {
        logger.log(Level.FINE, "BATCH %d/%d : mse=%.5f".formatted(pCurrentBatch, batchCount,
                                                                  pMse.doubleValue()));
    }
}
