package com.example.harel_000.opencvapp;

import android.util.Log;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;


public class Mode implements IoverlayOnImage {
    protected Mat _image;
    protected Mat _invertedMask;
    protected Mat _resizedImage;
    protected Mat _resizedInvertedMask;
    protected Size _changeableSize = new Size(0, 0);
    protected double _topRatio;
    protected double _leftRatio;
    protected double _heightRatio;
    protected double _widthRatio;

    public Mode(Mat image, double topRatio, double leftRatio, double heightRatio, double widthRatio) {
        _image = image;
        _invertedMask = new Mat();
        Imgproc.cvtColor(_image, _invertedMask, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.threshold(_invertedMask, _invertedMask, 1, 255, Imgproc.THRESH_BINARY_INV);
        Imgproc.cvtColor(_invertedMask, _invertedMask, Imgproc.COLOR_GRAY2RGBA);
        _topRatio = topRatio;
        _leftRatio = leftRatio;
        _heightRatio = heightRatio;
        _widthRatio = widthRatio;
    }

    protected void putFilterOnImage(Mat filter, Mat invertedMask, Mat roi) {

        Log.d("placingFilter", this.getClass().toString() + " was placed succesfully");
    }

    @Override
    public void overlayOnImage(Mat inputFrame, Rect face) {
        int h = face.height;
        int w = face.width;
        if (w < 10 || h < 10) {
            return;
        }
        int x = face.x;
        int y = face.y;
        _changeableSize.width = (int) (w * _widthRatio);
        _changeableSize.height = (int) (h * _heightRatio);
        _resizedImage = new Mat(_changeableSize, CvType.CV_8UC4);
        _resizedInvertedMask = new Mat(_changeableSize, CvType.CV_8UC4);
        Imgproc.resize(_image, _resizedImage, _changeableSize);
        Imgproc.resize(_invertedMask, _resizedInvertedMask, _changeableSize);
        int shiftDown = (int) (h * _topRatio);
        int shiftRight = (int) (w * _leftRatio);
        if (shiftDown + y > inputFrame.rows() || shiftDown + y < 0 || shiftRight + x > inputFrame.cols() || shiftRight + x < 0
                ||shiftDown + y + (int)_changeableSize.height > inputFrame.rows() || shiftDown + y +(int)_changeableSize.height  < 0
                || shiftRight + x + (int) _changeableSize.width > inputFrame.cols() || shiftRight + x + (int) _changeableSize.width < 0) {
            Log.d("placingExceptions","ROI not valid");
            return;
        }
        Mat roi = inputFrame.submat(y + shiftDown, y + shiftDown + (int) _changeableSize.height, x + shiftRight, x + shiftRight + (int) _changeableSize.width);
        Core.bitwise_and(roi, _resizedInvertedMask, roi);
        Core.bitwise_or(roi, _resizedImage, roi);
    }
}

