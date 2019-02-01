package com.example.harel_000.opencvapp;

import org.opencv.core.Mat;

public class BeardMode extends CompositeMode {
    public BeardMode(Mat image) {
        _features.add(new Mode(image,(double)1/2,0,(double)2/3,1));
    }
}
