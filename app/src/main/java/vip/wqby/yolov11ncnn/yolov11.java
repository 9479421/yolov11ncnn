package vip.wqby.yolov11ncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class yolov11 {


    public native boolean init(AssetManager manager, boolean use_gpu);
    public native OutResult[] detect(Bitmap bitmap);

    static {
        System.loadLibrary("yolov11ncnn");
    }
}
