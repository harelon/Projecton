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
import android.view.View.OnTouchListener;
import android.view.MotionEvent;
import android.view.View;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

enum Feature {
    EAR, BEARD
}

enum SpecificFeature {
    RABBIT_EARS, CAT_EARS, BROWN_BEARD
}

public class MainActivity extends Activity
        implements CvCameraViewListener, OnTouchListener {

    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadeFaceClassifier;
    private CascadeClassifier cascadeSmileClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;
    double placeEarsInFaceHeight = 3 / 5;
    double placeEarsInFaceWidth = -1 / 6;
    int EarRatioHeight = 3;
    int EarRatioWidth = 4;
    long lastTouch = 0;
    int beardPlacementWidth;
    int beardPlacementHeight;
    Mat filterImage;
    Feature currentFilter = Feature.EAR;
    Feature nextFilter = Feature.EAR;
    private SpecificFeature currentFeature = SpecificFeature.RABBIT_EARS;
    private SpecificFeature nextFeature = SpecificFeature.RABBIT_EARS;
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
        LoadImage(true);
        Log.d("featurePlacing", "image didn't change loading settings");
    }

    public void LoadImage(boolean OnStart) {
        if (!OnStart) {
            currentFeature = nextFeature;
            currentFilter = nextFilter;
        }
        Log.d("featurePlacing", currentFeature.name());
        int path = 0;
        try {
            switch (currentFilter) {
                case EAR:
                    switch (currentFeature) {
                        case RABBIT_EARS:
                            path = R.drawable.ear1;
                            Log.d("featurePlacing", "current path ear1");
                            nextFeature = SpecificFeature.CAT_EARS;
                            Log.d("featurePlacing", "next image cat ears");
                            break;
                        case CAT_EARS:
                            path = R.drawable.ear2;
                            Log.d("featurePlacing", "current path ear2");
                            nextFeature = SpecificFeature.BROWN_BEARD;
                            nextFilter = Feature.BEARD;
                            Log.d("featurePlacing", "next image brown beard");
                            break;
                    }
                    break;
                case BEARD:
                    path = R.drawable.beard2;
                    Log.d("featurePlacing", "current path beard");
                    nextFeature = SpecificFeature.RABBIT_EARS;
                    nextFilter = Feature.EAR;
                    Log.d("featurePlacing", "next image rabbit ears");
                    break;
            }
            filterImage = Utils.loadResource(this, path);
            Imgproc.cvtColor(filterImage, filterImage, Imgproc.COLOR_BGR2RGBA);
        } catch (Exception e) {
            Log.d("featurePlacing", e.toString());
        }
        Log.d("featurePlacing", "feature swapped successfully");
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
        LoadImage(false);
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
        if (currentFilter == Feature.EAR) {
            for (Rect face : detectedArray) {
                int h = face.height;
                int w = face.width;
                int x = face.x;
                int y = face.y;
                if (h < 10 || w < 10) {
                    continue;
                }
                Mat leftEar = new Mat((int) h / EarRatioHeight, (int) w / EarRatioWidth, CvType.CV_8UC4);
                Mat rightEar = new Mat();
                Log.d("earPlacing", "size =" + leftEar.size().toString());
                Log.d("earPlacing", filterImage.size().toString());
                Imgproc.resize(filterImage, leftEar, leftEar.size());
                Log.d("earPlacing", "ears resized");
                Core.flip(leftEar, rightEar, 1);
                Log.d("earPlacing ", "flipped left ear into right ear");
                Mat leftEarInvertedMask = new Mat(leftEar.width(), leftEar.height(), CvType.CV_8UC4);
                Mat rightEarInvertedMask = new Mat(leftEar.width(), leftEar.height(), CvType.CV_8UC4);
                Log.d("earPlacing ", "created both ears inverted mask");
                Imgproc.cvtColor(leftEar, leftEarInvertedMask, Imgproc.COLOR_RGBA2GRAY);
                Imgproc.threshold(leftEarInvertedMask, leftEarInvertedMask, 1, 255, Imgproc.THRESH_BINARY_INV);
                Imgproc.cvtColor(leftEarInvertedMask, leftEarInvertedMask, Imgproc.COLOR_GRAY2RGBA);
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
        } else if (currentFilter == Feature.BEARD) {
            for (Rect smile : detectedArray) {
                int h = smile.height;
                int w = smile.width;
                int x = smile.x;
                int y = smile.y;
                if (h < 200 || w < 200) {
                    continue;
                }
                Mat beard = new Mat((int) h, (int) w, CvType.CV_8UC4);
                Imgproc.resize(filterImage, beard, beard.size());
                Log.d("beardPlacing", "beard resized");
                Mat beardInvertedMask = new Mat(beard.width(), beard.height(), CvType.CV_8UC4);
                Log.d("beardPlacing ", "created beard inverted mask");
                Imgproc.cvtColor(beard, beardInvertedMask, Imgproc.COLOR_RGBA2GRAY);
                Imgproc.threshold(beardInvertedMask, beardInvertedMask, 1, 255, Imgproc.THRESH_BINARY_INV);
                Imgproc.cvtColor(beardInvertedMask, beardInvertedMask, Imgproc.COLOR_GRAY2RGBA);
                Log.d("beardPlacing ", "converted beard color");
                if (y + h - beard.rows()*2/3< 0 || y + h- beard.rows()*2/3 +beard.cols() > aInputFrame.rows() || x + beard.cols() > aInputFrame.cols()) {
                    Log.d("beardPlacing ", "roi not vaild");
                    continue;
                }
                Mat beardROI = aInputFrame.submat(y + h - beard.rows()*2/3, y + h- beard.rows()*2/3 +beard.cols(), x, x + beard.cols());
                Log.d("beardPlacing ", "got matching beardROI");
                Log.d("beardPlacing", beardROI.size().toString());
                Log.d("beardPlacing", aInputFrame.size().toString());
                Core.bitwise_and(beardROI, beardInvertedMask, beardROI);
                Log.d("beardPlacing", "used and on inverted mask and roi");
                Core.bitwise_or(beardROI, beard, beardROI);
                Log.d("beardPlacing", "placed beard into beardROI");
            }
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