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
import java.util.Random;
import java.util.stream.DoubleStream;

import static guru.nidi.graphviz.model.Factory.graph;
import static guru.nidi.graphviz.model.Factory.node;

public class NNUtil
{
    public static double[] rangeX(double start, double end, int nbValues)
    {
        double step = (end - start) / (nbValues - 1);
        return DoubleStream.iterate(start, n -> n + step).limit((long) ((end - start) / step) + 1)
                           .toArray();
    }

    public static void plot(String title, double[] x, double[] y)
    {
        Platform.startup(() -> {
            Graph graph = new Graph();
            graph.plot(title, x, y);

            graph.getXAxis().setAutoRanging(true);
            graph.getYAxis().setAutoRanging(true);
            graph.setCreateSymbols(true);

            Stage stage = new Stage();
            Scene scene = new Scene(graph, 800, 600);
            scene.getStylesheets().add(GraphDemo.class.getResource("style.css").toExternalForm());

            stage.setTitle("Graphe");
            stage.setScene(scene);
            stage.show();
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
                    v.retroPropagate();
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

    public static void main(String[] args)
    {
        Value x1 = new Value("x1", 2);
        Value x2 = new Value("x2", 0);
        Value w1 = new Value("w1", -3);
        Value w2 = new Value("w2", 1);
        Value b = new Value("b", 6.8813735870195432);
        //        Value b = new Value("b", 8);
        Value x1w1 = x1.mul(w1, "");
        Value x2w2 = x2.mul(w2, "");
        Value x1w1x2w1 = x1w1.plus(x2w2, "");
        Value n = x1w1x2w1.plus(b, "n");
        Value o = n.tanh().label("o");
        o.retroPropagate();
        NNUtil.graphviz(o);
    }
}
