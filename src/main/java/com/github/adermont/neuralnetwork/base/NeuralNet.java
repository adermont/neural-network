package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.io.NeuralNetSerializer;
import com.github.adermont.neuralnetwork.layer.CollectorLayer;
import com.github.adermont.neuralnetwork.layer.DataLayer;
import com.github.adermont.neuralnetwork.layer.DenseLayer;
import com.github.adermont.neuralnetwork.layer.NeuralLayer;
import com.github.adermont.neuralnetwork.math.LossFunction;
import com.github.adermont.neuralnetwork.math.Value;
import com.github.adermont.neuralnetwork.util.NNUtil;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

/**
 * A standard neural network.
 */
public class NeuralNet
{
    /**
     * Default learning rate used in case of insufficient batches (less than 10) to have an
     * auto-adaptative learning rate.
     */
    public static final double DEFAULT_LEARNING_RATE = 0.01;

    private int                inputDim;
    private Value[]            output;
    private DataLayer          inputLayer;
    private Deque<NeuralLayer> hiddenLayers;
    private CollectorLayer     outputLayer;
    private List<Value>        parameters;
    private double             learningRate;
    private LossFunction       lossFunction;

    private int epochCount; // Number of Epochs
    private int currentEpoch; // Current Epoch number
    private int batchCount; // Number of batches by epoch
    private int currentBatch; // Current batch number

    private List<NeuralNetListener> listeners;

    // Options:
    private boolean useRegularization;

    /**
     * Creates a new NeuralNet.
     *
     * @param inputDim            Dimension of the input (e.g. for an image of 32x32pixels,
     *                            inputDim=32x32).
     * @param defaultLearningRate Default learning rate (used if the number of batches is not
     *                            sufficient to have an adaptative learning rate).
     */
    public NeuralNet(int inputDim, double defaultLearningRate)
    {
        this.inputDim = inputDim;
        this.hiddenLayers = new LinkedList<>();
        this.parameters = new ArrayList<>();
        this.inputLayer = new DataLayer(inputDim, NeuronFunctions.IDENTITY);
        this.learningRate = defaultLearningRate;
        this.listeners = new ArrayList<>();
        this.lossFunction = LossFunction.MSE;
    }

    /**
     * Creates a new NeuralNet with a default learning rate of 0.01.
     *
     * @param inputDim Dimension of the input (e.g. for an image of 32x32pixels, inputDim=32x32).
     */
    public NeuralNet(int inputDim)
    {
        this(inputDim, DEFAULT_LEARNING_RATE);
    }

    /**
     * Adds a listener for events during the learning step.
     *
     * @param pNeuralNetListener The listener.
     */
    public void addListener(NeuralNetListener pNeuralNetListener)
    {
        this.listeners.add(pNeuralNetListener);
    }

    /**
     * Sets the loss function. See {@link LossFunction} for more details.
     *
     * @param pLossFunction      The loss function.
     * @param pUseRegularization Flag for using L2 regularization.
     */
    public void setLossFunction(LossFunction pLossFunction, boolean pUseRegularization)
    {
        if (pLossFunction == null)
        {
            throw new IllegalArgumentException("Loss Function can't be null");
        }
        this.lossFunction = pLossFunction;
        this.useRegularization = pUseRegularization;
    }

    /**
     * Initialize the parameters of a layer and connects it to the previous layer of the NN.
     *
     * @param currentLayer The layer to be initialized.
     */
    private void initLayer(NeuralLayer currentLayer)
    {
        // Connection to the previous layer
        NeuralLayer previousLayer = hiddenLayers.peekLast() == null ? inputLayer : hiddenLayers.peekLast();

        // Connect the layer to the previous one
        currentLayer.connect(previousLayer);

        // Initialize layer's parameters
        initializeLayerWeights(previousLayer, currentLayer);
        initializeLayerBias(currentLayer);
    }

    /**
     * Initializes weights of all neurons of a layer.
     *
     * @param previousLayer Previous layer.
     * @param currentLayer  Layer to be initialized.
     */
    private void initializeLayerWeights(NeuralLayer previousLayer, NeuralLayer currentLayer)
    {
        // Initialize weights with random values (array dim = neurons count of previous layer)
        double[][] weights = new double[currentLayer.getNeuronCount()][];
        for (int i = 0; i < weights.length; i++)
        {
            weights[i] = new Random().doubles(previousLayer.getNeuronCount(), -1, 1).toArray();
        }
        currentLayer.setWeights(weights);
    }

    /**
     * Initializes bias of all neurons of a layer.
     *
     * @param currentLayer Layer to be initialized.
     */
    private void initializeLayerBias(NeuralLayer currentLayer)
    {
        // Initialize bias with random values (array dim = neurons count of CURRENT layer)
        double[] bias = new double[currentLayer.getNeuronCount()];
        for (int i = 0; i < bias.length; i++)
        {
            bias[i] = new Random().nextDouble(-1, 1);
        }
        currentLayer.setBias(bias);
    }

    /**
     * Creates a new dense layer and connects it to the previous layer.
     *
     * @param nbNeurons Neurons count of the dense layer to be added.
     * @param pFunction The activation function of the neurons of the dense layer.
     * @return The dense layer with all neurons initialized with random weights and zero bias.
     */
    public DenseLayer addDenseLayer(int nbNeurons, NeuronFunctions pFunction)
    {
        DenseLayer dense = new DenseLayer(nbNeurons, pFunction);
        initLayer(dense);
        hiddenLayers.add(dense);
        return dense;
    }

    /**
     * Creates a dense layer as output layer. The neurons count determines the dimension of your
     * output. The DerivableFunction is mandatory, so if you just want to get output of the last
     * layer, use {@link NeuronFunctions#IDENTITY} as the activation function for the collector
     * layer.
     *
     * @param nbNeurons Number of neurons.
     * @param pFunction Activation function.
     * @return The created layer, automatically connected to previous layer.
     */
    public CollectorLayer setCollectorLayer(int nbNeurons, NeuronFunctions pFunction)
    {
        CollectorLayer collector = new CollectorLayer(nbNeurons, pFunction);
        initLayer(collector);
        return outputLayer = collector;
    }

    /**
     * @return The current loss function.
     */
    public LossFunction lossFunction()
    {
        return lossFunction;
    }

    /**
     * @return The last predicted output or null if no prediction or no learning has been made.
     */
    public Value[] getOutput()
    {
        return output;
    }

    /**
     * Number of input vector's dimension.
     *
     * @return input vector dimension.
     */
    public int inputDim()
    {
        return inputDim;
    }

    /**
     * @return All the network parameters.
     */
    public List<Value> parameters()
    {
        if (parameters.isEmpty())
        {
            hiddenLayers.forEach(layer -> parameters.addAll(layer.parameters()));
            parameters.addAll(outputLayer.parameters());
        }
        return parameters;
    }

    public NeuralLayer inputLayer()
    {
        return inputLayer;
    }

    public NeuralLayer outputLayer()
    {
        return outputLayer;
    }

    /**
     * @return List of all net layers.
     */
    public List<NeuralLayer> hiddenLayers()
    {
        List<NeuralLayer> result = new ArrayList<>();
        result.addAll(hiddenLayers);
        return result;
    }

    /**
     * @return The default learning rate (used only if batchCount > 10).
     */
    public double learningRate()
    {
        return learningRate;
    }

    /**
     * Only tries to predict a value according to the current network's parameters. This method does
     * not learn (to switch to learning mode, see {@link #fit(double[][], double[][], int, int)}.
     * <p>
     * Note : see also the loss() function if you wan to estimate the loss level of this
     * prediction.
     * </p>
     *
     * @param pX The input you want to predict an output.
     * @return An array of predicted output values.
     */
    public Value[] predict(double[] pX)
    {
        for (Neuron neuron : this.inputLayer.getNeurons())
        {
            ((DataNeuron) neuron).setData(pX);
        }
        this.inputLayer.propagate();
        return this.output = this.outputLayer.getOutput();
    }

    /**
     * Triggers a "TrainingStarted" event.
     */
    private void fireTrainingStarted()
    {
        this.listeners.forEach(l -> l.trainingStarted());
    }

    /**
     * Triggers a "TrainingTerminated" event.
     */
    private void fireTrainingTerminated()
    {
        this.listeners.forEach(l -> l.trainingTerminated());
    }

    /**
     * Triggers a "EpochStarted" event.
     *
     * @param pEpochNumber The epoch number.
     * @param pEpochCount  The epoch count.
     */
    private void fireEpochStartedEvent(int pEpochNumber, int pEpochCount)
    {
        listeners.forEach(l -> l.epochStarted(pEpochNumber, pEpochCount));
    }

    /**
     * Triggers a "EpochTerminated" event.
     *
     * @param pEpochNumber The epoch number.
     */
    private void fireEpochTerminatedEvent(int pEpochNumber)
    {
        listeners.forEach(l -> l.epochTerminated(pEpochNumber));
    }

    /**
     * Triggers a "BatchStarted" event.
     *
     * @param pBatchNumber     The batch that started.
     * @param pBatchCount      The total number of batchs in each epoch.
     * @param pBatchSize       Size of a single batch.
     * @param pBatchStartIndex Start index of the current starting batch.
     */
    private void fireBatchStartedEvent(int pBatchNumber, int pBatchCount, int pBatchSize,
                                       int pBatchStartIndex)
    {
        listeners.forEach(
                l -> l.batchStarted(pBatchNumber, pBatchCount, pBatchSize, pBatchStartIndex));
    }

    /**
     * Triggers a "BatchTerminated" event.
     *
     * @param pBatchNumber The batch that started.
     * @param pBatch       The batch containing loss value and batch accuracy.
     */
    private void fireBatchTerminatedEvent(int pBatchNumber, Batch pBatch)
    {
        listeners.forEach(l -> l.batchTerminated(currentBatch + 1, pBatch));
    }

    /**
     * Triggers a "LearningRateChanged" event.
     *
     * @param pLearningRate The new learning rate.
     */
    private void fireLearningRateChangedEvent(double pLearningRate)
    {
        listeners.forEach(l -> l.learningRateChanged(pLearningRate));
    }

    /**
     * Starts deep learning.
     *
     * @param x         All training inputs.
     * @param y         Expected output for those training data.
     * @param nbEpoch   Number of epochs.
     * @param batchSize Number of inputs per batch.
     */
    public void fit(double[][] x, double[][] y, int nbEpoch, int batchSize)
    {
        fireTrainingStarted();

        this.epochCount = nbEpoch;
        this.batchCount = (int) Math.ceil((double) x.length / batchSize);

        for (this.currentEpoch = 0; this.currentEpoch < epochCount; this.currentEpoch++)
        {
            fitNextEpoch(x, y);
        }
        fireTrainingTerminated();
    }

    /**
     * Starts a new epoch.
     *
     * @param x Training inputs.
     * @param y Expected outputs.
     */
    private void fitNextEpoch(double[][] x, double[][] y)
    {
        fireEpochStartedEvent(this.currentEpoch + 1, this.epochCount);

        // Randomize input (and output accordingly)
        NNUtil.shuffle(x, y);

        // Number of inputs for each batch :
        int batchSize = x.length / this.batchCount;

        // Start batches
        int batchStartIndex = 0;
        for (this.currentBatch = 0; this.currentBatch < this.batchCount; this.currentBatch++)
        {
            fireBatchStartedEvent(currentBatch + 1, batchCount, batchSize, batchStartIndex);

            // ==========================================
            // Do not forget to reset all the gradients !
            this.outputLayer.resetGradients();
            // ==========================================

            // Compute the batch loss
            Batch batch = computeLoss(x, y, batchStartIndex, batchSize);

            // Back propagation and optimization of parameters
            optimize(batch.loss());

            fireBatchTerminatedEvent(currentBatch + 1, batch);
        }

        fireEpochTerminatedEvent(currentEpoch + 1);
    }

    /**
     * Compute the percentage of correct values. This is a generic algorithm that may not fit to
     * every models. You should consider using another version.
     *
     * @param y
     * @param scores
     * @param batchStartIndex
     * @param tolerance
     * @return
     */
    protected double computeAccuracy(double[][] y, double[][] scores, int batchStartIndex,
                                     double tolerance)
    {
        double accuracy = 0.0;

        for (int i = 0; i < scores.length; i++)
        {
            int score = 0;
            for (int j = 0; j < scores[i].length; j++)
            {
                double scorei = scores[i][j];
                double yi = y[i + batchStartIndex][j];
                score += Math.abs(scorei - yi) < tolerance ? 1 : 0;
            }
            accuracy += (score == scores[i].length ? 1 : 0);
        }
        return accuracy / scores.length;
    }

    /**
     * The complete loss computation in batch mode with backward prop. This method also computes the
     * accuracy.
     *
     * @param x               Training inputs.
     * @param y               Training outputs.
     * @param batchStartIndex Index of batch's start, relative to 'x'.
     * @param batchSize       Size of the batch.
     * @return The loss Value as a function of the model's output.
     */
    protected Batch computeLoss(double[][] x, double[][] y, int batchStartIndex, int batchSize)
    {
        int length = Math.min(batchSize, x.length);
        Value loss = null;
        double[][] scores = new double[length][];

        for (int i = batchStartIndex; i < length; i++)
        {
            // Predict output
            Value[] score = predict(x[i]);

            // Store prediction
            scores[i] = Arrays.stream(score).mapToDouble(Value::doubleValue).toArray();

            // Accumulate absolute loss
            loss = dataLoss(y[i], score).plus(loss);
        }

        // Apply weighting on the data loss
        int nbLosses = Math.min(batchSize, x.length) - batchStartIndex;
        final Value weightedLoss = loss.div(nbLosses);

        // L2 regularization (optional, regLoss may be null)
        Value regularizedLoss = regLoss();
        Value totalLoss = weightedLoss.plus(regularizedLoss);

        // Compute batch accuracy
        double tolerance = 0.03;
        double accuracy = computeAccuracy(y, scores, batchStartIndex, tolerance);

        return new Batch(batchSize, totalLoss, accuracy);
    }

    /**
     * Compares the last output with an expected output value.
     *
     * @param expected The expected output.
     */
    protected Value dataLoss(double[] expected, Value[] output)
    {
        Value loss = null;
        //String sBatch = "batch" + currentBatch;

        for (int i = 0; i < expected.length; i++)
        {
            Value expectedValue = new Value(expected[i]);//.label(sBatch + ".expect" + i);
            Value dataLoss = lossFunction().apply(expectedValue, output[i]);

            loss = dataLoss.plus(loss);//.label(sBatch + ".loss" + i);
        }
        return loss;
    }

    /**
     * Some loss functions may need a regularization. This function aims at performing this
     * regularization and the result will be added to the weighted loss 'WL' of the batch (WL =
     * sum(losses) / nbLosses).
     *
     * @return The regularization loss (RL = alpha * sum(p*p) for p in parameters()).
     */
    protected Value regLoss()
    {
        if (useRegularization)
        {
            // Regularization
            Value alpha = new Value(0.0001);
            Double sum = parameters().stream().map(p -> p.doubleValue())
                                     .reduce(0.0, (v1, v2) -> v1 + (v2 * v2));
            return alpha.mul(sum);
        }
        return null;
    }

    /**
     * Starts a standard gradient descent by backpropagation.
     */
    private void optimize(Value loss)
    {
        // Optimize the learning rate : higher values in the beginning accelerates gradient descent
        if (batchCount >= 10)
        {
            learningRate = 1.0 - (0.9 * ((double) (currentBatch + 1) / (double) batchCount));
            fireLearningRateChangedEvent(learningRate);
        }

        // Start back propagation
        loss.backPropagation();

        // Update parameters
        parameters().parallelStream().forEach(p -> p.update(learningRate));
    }

    public void summary()
    {
        System.out.println("Model: sequential");
        System.out.println("_____________________________________________________________________");
        System.out.println(" Layer (type)\t\t\tOutput shape\t\t\tParam #");
        System.out.println("=====================================================================");
        for (NeuralLayer hiddenLayer : hiddenLayers)
        {
            System.out.println(
                    " %s\t\t\t\t%s\t\t\t\t%d".formatted(hiddenLayer.getClass().getSimpleName(),
                                                        "(None, " + hiddenLayer.getNeuronCount() + ")",
                                                        hiddenLayer.parameters().size()));
            System.out.println();
        }
        System.out.println(" Output (Dense)\t\t\t%s\t\t\t\t%d".formatted(
                "(None, " + outputLayer.getNeuronCount() + ")", outputLayer.parameters().size()));
        System.out.println("=====================================================================");
        System.out.println("Total params        : %d".formatted(parameters().size()));
        System.out.println("Trainable params    : %d".formatted(parameters().size()));
        System.out.println("Non-trainable params: %d".formatted(parameters().size()));
        System.out.println();
    }

    @Override
    public boolean equals(Object pO)
    {
        if (this == pO)
        {
            return true;
        }

        if (!(pO instanceof NeuralNet neuralNet))
        {
            return false;
        }
        boolean isEquals = new EqualsBuilder().append(hiddenLayers, neuralNet.hiddenLayers)
                                              .isEquals();

        return isEquals && new EqualsBuilder().append(inputDim, neuralNet.inputDim)
                                              .append(learningRate, neuralNet.learningRate)
                                              .append(useRegularization,
                                                      neuralNet.useRegularization)
                                              .append(inputLayer, neuralNet.inputLayer)
                                              .append(outputLayer, neuralNet.outputLayer)
                                              .append(parameters, neuralNet.parameters)
                                              .append(lossFunction, neuralNet.lossFunction)
                                              .isEquals();
    }

    @Override
    public int hashCode()
    {
        return new HashCodeBuilder(17, 37).append(inputDim).append(inputLayer).append(hiddenLayers)
                                          .append(outputLayer).append(parameters)
                                          .append(learningRate).append(lossFunction)
                                          .append(useRegularization).toHashCode();
    }

    public void save(Path dest) throws IOException
    {
        new NeuralNetSerializer().serialize(this, dest);
    }

    public static NeuralNet load(Path dest) throws IOException
    {
        return new NeuralNetSerializer().deserialize(dest);
    }

}
