package com.example.harel_000.opencvapp;

import android.app.Activity;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.res.Configuration;
import android.os.Bundle;
import android.util.Log;
import android.view.*;
import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import android.view.WindowManager;

import android.view.View.OnTouchListener;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity
        implements CvCameraViewListener, OnTouchListener {

    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadeFaceClassifier;
    private CascadeClassifier cascadeSmileClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;
    private IoverlayOnImage[] modes;
    IoverlayOnImage currentMode;
    int modeIndex = 0;
    long lastTouch = 0;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    initializeOpenCVDependencies();
                    openCvCameraView.setOnTouchListener(MainActivity.this);
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    private void initializeOpenCVDependencies() {
        // And we are ready to go
        cascadeFaceClassifier = loadCascade(R.raw.lbpcascade_frontalface, "lbpcascade_frontalface.xml");
        cascadeSmileClassifier = loadCascade(R.raw.haarcascade_smile, "haarcascade_smile.xml");
        modes = new IoverlayOnImage[3];
        modes[0] = new EarMode(loadImage(R.drawable.ear1));
        modes[1] = new EarMode(loadImage(R.drawable.ear2));
        modes[2] = new BeardMode(loadImage(R.drawable.beard2));
        currentMode = modes[modeIndex];
        openCvCameraView.enableView();
    }

    public CascadeClassifier loadCascade(int resource, String name) {
        try {
            // Copy the resource into a temp file so OpenCV can load it
            InputStream is = getResources().openRawResource(resource);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, name);
            FileOutputStream os = new FileOutputStream(mCascadeFile);


            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            return new CascadeClassifier(mCascadeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }
        return null;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);
        openCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        openCvCameraView.setVisibility(SurfaceView.VISIBLE);
        openCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);
        // The faces will be a 20% of the height of the screen
        absoluteFaceSize = (int) (height * 0.2);
    }

    public void switchImage() {
        modeIndex = (modeIndex + 1) % modes.length;
        currentMode = modes[modeIndex];
    }

    @Override
    public void onCameraViewStopped() {

    }

    public boolean onTouch(View v, MotionEvent event) {
        long currentTouchTime = event.getEventTime();
        if (!(currentTouchTime > lastTouch + 200)) {
            return false;
        }
        lastTouch = currentTouchTime;
        Log.d("featurePlacing", "sensed touch successfully");
        switchImage();
        Log.d("featurePlacing", "image updated successfully");
        return true;
    }

    @Override
    public Mat onCameraFrame(Mat aInputFrame) {
        // Create a grayscale image
//        Log.d("ear problems","returning ears");
        Imgproc.cvtColor(aInputFrame, grayscaleImage, Imgproc.COLOR_RGBA2GRAY);
        MatOfRect detectedMats = new MatOfRect();
        // Use the classifier to detect faces
        // If there are any faces found, draw a rectangle around it
        if (cascadeFaceClassifier != null) {
            cascadeFaceClassifier.detectMultiScale(grayscaleImage, detectedMats, 1.1, 2, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }
        Rect[] detectedArray = detectedMats.toArray();
        for (Rect face : detectedArray) {
            currentMode.overlayOnImage(aInputFrame, face);
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

    public Mat loadImage(int imagePath) {
        Mat ImageLoaded = new Mat();
        try {
            ImageLoaded = Utils.loadResource(this, imagePath);
            Imgproc.cvtColor(ImageLoaded, ImageLoaded, Imgproc.COLOR_BGR2RGBA);
        } catch (Exception e) {

        }
        return ImageLoaded;
    }
}
