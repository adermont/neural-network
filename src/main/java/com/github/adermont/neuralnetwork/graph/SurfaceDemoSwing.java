package com.github.adermont.neuralnetwork.graph;

import org.jzy3d.analysis.AbstractAnalysis;
import org.jzy3d.analysis.AnalysisLauncher;
import org.jzy3d.chart.factories.SwingChartFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRBG;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Mapper;
import org.jzy3d.plot3d.builder.SurfaceBuilder;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;

public class SurfaceDemoSwing extends AbstractAnalysis
{
    public SurfaceDemoSwing()
    {
        super(new SwingChartFactory());
    }

    public static void main(String[] args) throws Exception
    {
        SurfaceDemoSwing d = new SurfaceDemoSwing();
        AnalysisLauncher.open(d);
    }

    public static Shape surface(Range xRange, Range yRange, float alpha)
    {
        Mapper mapper = new Mapper()
        {
            @Override
            public double f(double x, double y)
            {
                return 1/(2+x*x+y*y);
            }
        };
        int steps = 100;

        Shape surface = new SurfaceBuilder().orthonormal(
                new OrthonormalGrid(xRange, steps, yRange, steps), mapper);
        ColorMapper colorMapper = new ColorMapper(new ColorMapRBG(),
                                                  surface.getBounds().getZmin(),
                                                  surface.getBounds().getZmax(),
                                                  new Color(1, 1, 1, alpha));
        surface.setColorMapper(colorMapper);

        surface.setFaceDisplayed(true);
        surface.setWireframeDisplayed(true);
        surface.setWireframeColor(Color.BLACK);
        surface.setWireframeWidth(1);

        return surface;
    }

    @Override
    public void init()
    {
        Range r = new Range(-5, 5);
        Shape surface = surface(r, r, 0.8f);

        chart = new SwingChartFactory().newChart(Quality.Advanced());
        chart.add(surface);
    }
}