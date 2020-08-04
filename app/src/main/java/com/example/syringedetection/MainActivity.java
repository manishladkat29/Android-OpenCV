package com.example.syringedetection;


import android.annotation.SuppressLint;
import android.content.Context;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;


import org.opencv.android.OpenCVLoader;

import butterknife.ButterKnife;
import butterknife.OnClick;

public class MainActivity extends AppCompatActivity implements PhotoFragment.OnFragmentInteractionListener {

    int PERMISSION_ALL = 1;
    boolean flagPermissions = false;

    String[] PERMISSIONS = {
            android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
            android.Manifest.permission.CAMERA
    };

    //android custom camera - crop image inside rectangle


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);
        checkPermissions();

        if(OpenCVLoader.initDebug()){
            Log.d("Start", "OpenCV Loaded");
        }
    }

    @OnClick(R.id.make_photo_button)
    void onClickScanButton() {
        // check permissions
        if (!flagPermissions) {
            checkPermissions();
            return;
        }
        //start photo fragment
        getSupportFragmentManager()
                .beginTransaction()
                .replace(R.id.res_photo_layout, new PhotoFragment())
                .addToBackStack(null)
                .commit();
    }

    @SuppressLint("NewApi")
    void checkPermissions() {
        if (!hasPermissions(this, PERMISSIONS)) {
            requestPermissions(PERMISSIONS,
                    PERMISSION_ALL);
            flagPermissions = false;
        }
        flagPermissions = true;

    }

    public static boolean hasPermissions(Context context, String... permissions) {
        if (context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }


    @Override
    public void onFragmentInteraction(Bitmap bitmap) {
        if (bitmap != null) {
            ImageFragment imageFragment = new ImageFragment();
            imageFragment.imageSetupFragment(bitmap);

            getSupportFragmentManager()
                    .beginTransaction()
                    .replace(R.id.res_photo_layout, imageFragment)
                    .addToBackStack(null)
                    .commit();
        }
    }
}