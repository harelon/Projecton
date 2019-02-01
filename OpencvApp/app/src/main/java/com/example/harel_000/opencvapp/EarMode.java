package com.example.harel_000.opencvapp;

import org.opencv.core.*;

public class EarMode extends CompositeMode{
    private static double _earHeightRatio = (double)1/3;
    private static double _earWidthRatio = (double)1/4;
    private static  double _earTopRatio = -_earHeightRatio;
    private static double _leftEarLeftRatio=0;
    private static double _rightEarLeftRatio=(double)(1-_earWidthRatio);
    public EarMode(Mat image) {
        _features.add(new Mode(image,_earTopRatio,_rightEarLeftRatio,_earHeightRatio,_earWidthRatio));
        Mat flippedImage=new Mat();
        Core.flip(image,flippedImage,1);
        _features.add(new Mode(flippedImage,_earTopRatio,_leftEarLeftRatio,_earHeightRatio,_earWidthRatio));
    }
}