package com.example.harel_000.opencvapp;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.*;

public class CompositeMode implements IoverlayOnImage {
    protected List<IoverlayOnImage> _features;

    public CompositeMode(List<IoverlayOnImage> features) {
        _features = new ArrayList<>(features);
    }
    protected CompositeMode()
    {
        _features = new ArrayList<>();
    }
    @Override
    public void overlayOnImage(Mat inputFrame, Rect face) {
        for (IoverlayOnImage feature : _features) {
            feature.overlayOnImage(inputFrame, face);
        }
    }
}
