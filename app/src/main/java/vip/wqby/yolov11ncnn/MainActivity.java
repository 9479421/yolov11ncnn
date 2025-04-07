package vip.wqby.yolov11ncnn;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;
import android.Manifest;

import vip.wqby.yolov11ncnn.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {



    static {
        System.loadLibrary("yolov11ncnn");
    }

    public int generateRandomColor() {
        Random random = new Random();
        // 生成随机的 RGB 值
        int red = random.nextInt(256);
        int green = random.nextInt(256);
        int blue = random.nextInt(256);

        // 返回组合的 RGB 颜色
        return Color.rgb(red, green, blue);
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


    public void drawBitmap(Bitmap bitmap){

        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5);
        paint.setTextSize(50);


        OutResult[] detect = yolov11.detect(mutableBitmap);
        for (OutResult obj : detect) {
            paint.setColor(generateRandomColor()); // 红色


            System.out.println(obj.classId + " " + obj.confidence + " " + obj.x1 + " " + obj.y1 + " " + obj.x2 + " " + obj.y2);

            canvas.drawRect(obj.x1, obj.y1, obj.x2, obj.y2, paint);

            String name = classes[obj.classId];

            canvas.drawText(name, obj.x1, obj.y1 - 10, paint);
        }

        // 现在你可以使用这个Bitmap对象，例如显示在ImageView中
        ImageView imageView = binding.imageView;
        imageView.setImageBitmap(mutableBitmap);

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            Uri uri = data.getData();

            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                drawBitmap(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        if (requestCode == REQUEST_TAKE_PHOTO  && resultCode == RESULT_OK) {
            try {
                Bitmap bitmap = handleSamplingAndRotationBitmap(currentPhotoPath);
                drawBitmap(bitmap);
            } catch (Exception e) {
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


        //申请摄像头权限
        checkCameraPermission();

        binding.buttonSelect.setOnClickListener(v -> {

            //从相册中选择图片
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);

        });

        binding.buttonTakephoto.setOnClickListener(v -> {
            //拍照
            dispatchTakePictureIntent();
        });

    }


    private static final int REQUEST_TAKE_PHOTO = 1;
    private String currentPhotoPath;

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) == null) return;

        File photoFile = createImageFile(); // 创建临时文件
        if (photoFile == null) return;

        Uri photoURI = FileProvider.getUriForFile(this,
                getPackageName() + ".fileprovider",
                photoFile);

        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
        startActivityForResult(takePictureIntent, REQUEST_TAKE_PHOTO);
    }

    private File createImageFile() {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        try {
            File image = File.createTempFile(imageFileName, ".jpg", storageDir);
            currentPhotoPath = image.getAbsolutePath();
            return image;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }


    private Bitmap handleSamplingAndRotationBitmap(String photoPath) {
        // 获取图片尺寸
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(photoPath, options);

        // 计算缩放比例（这里缩小到原图的1/4）
        options.inSampleSize = calculateInSampleSize(options, 800, 800);
        options.inJustDecodeBounds = false;
        Bitmap bitmap = BitmapFactory.decodeFile(photoPath, options);

        // 处理旋转角度
        try {
            ExifInterface exif = new ExifInterface(photoPath);
            int orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_UNDEFINED
            );
            bitmap = rotateBitmap(bitmap, orientation);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bitmap;
    }

    private int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;
            while ((halfHeight / inSampleSize) >= reqHeight
                    && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }
        return inSampleSize;
    }

    private Bitmap rotateBitmap(Bitmap bitmap, int orientation) {
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.postRotate(90);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.postRotate(180);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.postRotate(270);
                break;
            default:
                return bitmap;
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    // 在 Activity/Fragment 中定义权限请求回调
    private ActivityResultLauncher<String> requestCameraLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    // 权限已授予，执行拍照操作
                    dispatchTakePictureIntent();
                } else {
                    // 权限被拒绝，提示用户
                    Toast.makeText(this, "需要摄像头权限才能拍照", Toast.LENGTH_SHORT).show();
                }
            });

    // 触发权限请求的方法
    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            // 已有权限，直接拍照
//            dispatchTakePictureIntent();
        } else {
            // 请求权限
            requestCameraLauncher.launch(Manifest.permission.CAMERA);
        }
    }
}