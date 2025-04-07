package vip.wqby.yolov11ncnn;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.ImageView;

import java.io.IOException;
import java.io.InputStream;

import vip.wqby.yolov11ncnn.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("yolov11ncnn");
    }

    private ActivityMainBinding binding;

    yolov11 yolov11 = new yolov11();
    private static final int PICK_IMAGE_REQUEST = 1;

    String classes[] = {"Cholesterol Test Strip",
            "Uric Acid Test Strip",
            "Glucose Strip",
            "Various Analyzers",
            "Urine Analyzer",
            "Infrared Thermomete",
            "Pulse Wave Monito",
            "Sterile Needle",
            "Limb Clamp Green",
            "Limb Clamp Black",
            "Limb Clamp Red",
            "Limb Clamp Yellow",
            "Suction Bulb Red",
            "Suction Bulb  Brown",
            "Suction Bulb Green",
            "Suction Bulb Purple",
            "Suction Bulb Yellow",
            "Suction Bulb Black",
            "ECG Lead Wire"};


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            Uri uri = data.getData();

            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);

                Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                Canvas canvas = new Canvas(mutableBitmap);

                Paint paint = new Paint();
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(5);
                paint.setColor(0xFFFF0000); // 红色


                OutResult[] detect = yolov11.detect(mutableBitmap);
                for (OutResult obj : detect) {
                    System.out.println(obj.classId + " " + obj.confidence + " " + obj.x1 + " " + obj.y1 + " " + obj.x2 + " " + obj.y2);

                    canvas.drawRect(obj.x1, obj.y1, obj.x2, obj.y2, paint);

                    String name = classes[obj.classId];
                    paint.setTextSize(50);

                    canvas.drawText(name, obj.x1, obj.y1 - 10, paint);
                }

                // 现在你可以使用这个Bitmap对象，例如显示在ImageView中
                ImageView imageView = binding.imageView;
                imageView.setImageBitmap(mutableBitmap);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());


        yolov11.init(getAssets(), false);


        binding.buttonSelect.setOnClickListener(v -> {

            //从相册中选择图片
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);

        });


    }

}