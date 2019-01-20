package com.example.harel_000.opencvapp;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;
import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import android.view.Window;
import android.view.WindowManager;

import android.view.SurfaceView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity
        implements CvCameraViewListener {

    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadeClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;
    double placeEarsInFaceHeight = 3 / 5;
    double placeEarsInFaceWidth = -1 / 6;
    int EarRatioHeight = 3;
    int EarRatioWidth = 4;
    Mat earImage;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    initializeOpenCVDependencies();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    private void initializeOpenCVDependencies() {

        try {
            // Copy the resource into a temp file so OpenCV can load it
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);


            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }

        // And we are ready to go
        openCvCameraView.enableView();
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        openCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        openCvCameraView.setVisibility(SurfaceView.VISIBLE);
        openCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);
        // The faces will be a 20% of the height of the screen
        absoluteFaceSize = (int) (height * 0.2);
        try {
            earImage = Utils.loadResource(this, R.drawable.ear3);
            Log.d("earPlacing", "cols =" + earImage.cols() + " rows=" + earImage.rows() + " type=" + earImage.type());
            Log.d("earPlacing", "ears loaded");
        } catch (Exception e) {
            Log.d("earPlacing", e.toString());
        }
        Log.d("earPlacing", "ears resized");
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(Mat aInputFrame) {
        // Create a grayscale image
//        Log.d("ear problems","returning ears");
        Imgproc.cvtColor(aInputFrame, grayscaleImage, Imgproc.COLOR_RGBA2RGB);

        MatOfRect faces = new MatOfRect();

        // Use the classifier to detect faces
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 2, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }

        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();
        for (Rect face : facesArray) {
            int h = face.height;
            int w = face.width;
            int x = face.x;
            int y = face.y;
            if (h < 100 || w < 100) {
                continue;
            }
            Mat leftEar = new Mat((int) h / EarRatioHeight, (int) w / EarRatioWidth, CvType.CV_8UC4);
            Size earSize = new Size(leftEar.width(), leftEar.height());
            Mat rightEar = new Mat();
            Log.d("earPlacing", "size =" + earSize.toString());
            Imgproc.resize(earImage, leftEar, earSize);
            Log.d("earPlacing", "ears resized");
            Core.flip(leftEar, rightEar, 1);
            Log.d("earPlacing ", "flipped left ear into right ear");
            Mat leftEarInvertedMask = new Mat(leftEar.width(), leftEar.height(), CvType.CV_8UC4);
            Mat rightEarInvertedMask = new Mat(leftEar.width(), leftEar.height(), CvType.CV_8UC4);
            Log.d("earPlacing ", "created both ears inverted mask");
            Imgproc.cvtColor(leftEar, leftEarInvertedMask, Imgproc.COLOR_BGRA2GRAY);
            Imgproc.threshold(leftEarInvertedMask, leftEarInvertedMask, 1, 255, Imgproc.THRESH_BINARY_INV);
            Imgproc.cvtColor(leftEarInvertedMask, leftEarInvertedMask, Imgproc.COLOR_GRAY2BGRA);
            Log.d("earPlacing ", "actually used the mask");
            Core.flip(leftEarInvertedMask, rightEarInvertedMask, 1);
            Log.d("earPlacing ", "flipped left mask into right mask");
            if (y - leftEar.rows() < 0 || x + w - rightEar.cols() < 0 || x + w - rightEar.cols() < 0 || x + rightEar.cols() > aInputFrame.cols()) {
                Log.d("earPlacing ", "roi not vaild");
                continue;
            }
            Mat leftEarROI = aInputFrame.submat(y - leftEar.rows(), y, x + w - rightEar.cols(), x + w);
            Mat rightEarROI = aInputFrame.submat(y - rightEar.rows(), y, x, x + rightEar.cols());
            Log.d("earPlacing ", "created right ear roi");
            Core.bitwise_and(rightEarROI, rightEarInvertedMask, rightEarROI);
            Core.bitwise_or(rightEarROI, rightEar, rightEarROI);
            Log.d("earPlacing ", "placed ear into rightEarROI");
            Core.bitwise_and(leftEarROI, leftEarInvertedMask, leftEarROI);
            Core.bitwise_or(leftEarROI, leftEar, leftEarROI);
            Log.d("earPlacing ", "placed ear into leftEarROI");
        }
        return aInputFrame;
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
}