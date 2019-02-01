package com.example.harel_000.opencvapp;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

public interface IoverlayOnImage {
    void overlayOnImage(Mat inputFrame, Rect face);
}
