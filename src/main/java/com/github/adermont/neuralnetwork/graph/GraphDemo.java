package com.github.adermont.neuralnetwork.graph;

import com.github.adermont.neuralnetwork.util.NNUtil;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.stage.Stage;

import static com.github.adermont.neuralnetwork.math.DerivableFunction.*;

public class GraphDemo
{


    public static void main(String[] args)
    {
        int nbValues = 1000;
        double start = -10, end = 10;

        double[] x = NNUtil.rangeX(start, end, nbValues);

        Platform.startup(() -> {
            Graph graph = new Graph();
            graph.plot("TANH", x, TANH);
            graph.plot("SIGMA", x, SIGMA);
            graph.plot("RELU", x, RELU);
            graph.plot("IDENTITY", x, IDENTITY);
            graph.plot("HEAVISIDE", x, HEAVISIDE);
            graph.plot("RELU o TANH", x, TANH.compose(RELU));

            graph.getXAxis().setAutoRanging(false);
            graph.getYAxis().setAutoRanging(false);
            graph.getXAxis().setLowerBound(-5);
            graph.getXAxis().setUpperBound(5);
            graph.getYAxis().setLowerBound(-1.5);
            graph.getYAxis().setUpperBound(1.5);

            Stage stage = new Stage();
            Scene scene = new Scene(graph, 800, 600);
            scene.getStylesheets().add(GraphDemo.class.getResource("style.css").toExternalForm());

            stage.setTitle("Graphe");
            stage.setScene(scene);
            stage.show();
        });
    }
}
