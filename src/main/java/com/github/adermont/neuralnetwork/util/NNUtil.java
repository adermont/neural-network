package com.github.adermont.neuralnetwork.util;

import com.github.adermont.neuralnetwork.graph.Graph;
import com.github.adermont.neuralnetwork.graph.GraphDemo;
import com.github.adermont.neuralnetwork.math.Value;
import guru.nidi.graphviz.attribute.Font;
import guru.nidi.graphviz.attribute.Label;
import guru.nidi.graphviz.attribute.Shape;
import guru.nidi.graphviz.attribute.*;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Node;
import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.stage.Stage;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

import static guru.nidi.graphviz.model.Factory.graph;
import static guru.nidi.graphviz.model.Factory.node;

public class NNUtil
{
    // Manually starts the JavaFX platform, just in case we are running in headless mode
    static
    {
        try
        {
            Platform.startup(() -> {
            });
        }
        catch (Exception ignored)
        {
        }
    }

    public static double[] range(double start, double end, int nbValues)
    {
        double step = (end - start) / (nbValues - 1);
        return DoubleStream.iterate(start, n -> n + step).limit((long) ((end - start) / step) + 1)
                           .toArray();
    }

    public static double[][] range2dim(double start, double end, int nbValues)
    {
        double step = (end - start) / (nbValues - 1);
        double[] doubles = DoubleStream.iterate(start, n -> n + step)
                                       .limit((long) ((end - start) / step) + 1).toArray();
        double[][] result = new double[doubles.length][];
        for (int i = 0; i < doubles.length; i++)
        {
            result[i] = new double[]{doubles[i]};
        }
        return result;
    }

    public static void plot(String title, List<Double>... series)
    {
        Platform.runLater(() -> {
            Graph graph = new Graph();
            graph.getXAxis().setAutoRanging(true);
            graph.getYAxis().setAutoRanging(true);
            graph.setCreateSymbols(true);

            int iSerie = 1;
            for (List<Double> values : series)
            {
                double[] x = range(1, values.size(), values.size());
                double[] y = values.stream().mapToDouble(Double::doubleValue).toArray();
                graph.plot(title + iSerie, x, y);
                iSerie++;
            }

            showGraph(graph);
        });
    }

    private static void showGraph(Graph g)
    {
        Platform.runLater(() -> {
            Stage stage = new Stage();
            Scene scene = new Scene(g, 800, 600);
            scene.getStylesheets().add(GraphDemo.class.getResource("style.css").toExternalForm());

            stage.setTitle("Graphe");
            stage.setScene(scene);
            stage.show();
        });
    }

    public static void plot(String title, double[] y)
    {
        double[] x = range(1, y.length, y.length);
        plot(title, x, y);
    }

    public static void plot(String title, double[] x, double[]... ySeries)
    {
        Platform.runLater(() -> {
            Graph graph = new Graph();
            for (double[] y : ySeries)
            {
                graph.plot(title, x, y);
            }
            graph.getXAxis().setAutoRanging(true);
            graph.getYAxis().setAutoRanging(true);
            graph.setCreateSymbols(true);

            showGraph(graph);
        });
    }

    public static void buildNodes(Value v, ArrayList<Node> nodes, ArrayList<Node> edges)
    {
        String id = "" + System.identityHashCode(v);
        String label = "{%s | %.3f | grad=%.3f}".formatted(v.label(), v.data().doubleValue(),
                                                           v.gradient());
        Node node0 = node(id).with(Label.of(label), Shape.M_RECORD, Style.lineWidth(2));
        nodes.add(node0);

        if (v.operator() != null)
        {
            final Node node1 = node(id + v.operator()).with(Label.of(v.operator()),
                                                            Style.lineWidth(2));
            nodes.add(node1.link(node0));
            v.children().forEach(child -> {
                String idChild = "" + System.identityHashCode(child);
                edges.add(node(idChild).link(node1));
                buildNodes(child, nodes, edges);
            });
        }

    }

    private static guru.nidi.graphviz.model.Graph buildGraphModel(Value v)
    {
        ArrayList<Node> nodes = new ArrayList<>();
        ArrayList<Node> edges = new ArrayList<>();
        buildNodes(v, nodes, edges);
        return graph("Graphe").directed().graphAttr().with(Rank.dir(Rank.RankDir.LEFT_TO_RIGHT))
                              .nodeAttr().with(Font.name("Consolas")).with(nodes).with(edges);
    }

    public static void graphviz(Value v)
    {
        Graphviz.noHeadless();
        guru.nidi.graphviz.model.Graph g = buildGraphModel(v);
        BufferedImage image = Graphviz.fromGraph(g).render(Format.SVG).toImage();
        displayImage(image, v);
    }

    public static void displayImage(BufferedImage image, Value v)
    {
        SwingUtilities.invokeLater(() -> {

            // Crée une étiquette pour afficher l'image
            JFrame frame = new JFrame();
            JLabel label = new JLabel(new ImageIcon(image));

            // Crée un conteneur pour la fenêtre
            Container container = frame.getContentPane();
            container.setLayout(new BorderLayout());
            container.add(label, BorderLayout.CENTER);

            frame.addKeyListener(new KeyAdapter()
            {
                @Override
                public void keyTyped(KeyEvent e)
                {
                    v.backPropagation();
                    guru.nidi.graphviz.model.Graph g = buildGraphModel(v);
                    BufferedImage image = Graphviz.fromGraph(g).width(1024).render(Format.PNG)
                                                  .toImage();
                    label.setIcon(new ImageIcon(image));
                }
            });

            // Configure la fenêtre
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }

    public static void shuffle(Object[] array, Object[] expected)
    {
        Random rnd = new Random();
        int size = array.length;
        for (int i = size; i > 1; i--)
        {
            int rdi = rnd.nextInt(i);
            swap(array, i - 1, rdi);
            swap(expected, i - 1, rdi);
        }
    }

    private static void swap(Object[] arr, int i, int j)
    {
        Object tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
