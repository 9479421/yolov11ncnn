#include <android/bitmap.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <jni.h>
#include <string>
#include <vector>

// ncnn
#include "layer.h"
#include "net.h"
#include "benchmark.h"
#include "cpu.h"


struct Object {
//    cv::Rect_<float> rect;
    int x1;
    int y1;
    int x2;
    int y2;

    int classId;
    float confidence;
};

struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};


static float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

static float intersection_area(const Object &a, const Object &b) {
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1) {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].confidence;

    while (i <= j) {
        while (faceobjects[i].confidence > p)
            i++;

        while (faceobjects[j].confidence < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] =
                (faceobjects[i].x2 - faceobjects[i].x1) * (faceobjects[i].y2 - faceobjects[i].y1);
    }

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void
generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides,
                          std::vector<GridAndStride> &grid_strides) {
    for (int i = 0; i < (int) strides.size(); i++) {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static ncnn::Mat transpose_ncnn_mat(const ncnn::Mat &mat) {
    int w = mat.w;
    int h = mat.h;
    int c = mat.c;
    int elemsize = mat.elemsize; // 每个元素的字节数，通常为4（float）

    // 创建一个新的 Mat，交换 w 和 h
    ncnn::Mat transposed(h, w, c);

    for (int z = 0; z < c; z++) {
        for (int y = 0; y < h; y++) {
            const float *src_ptr = mat.channel(z).row(y);
            float *dst_ptr = transposed.channel(z).row(y);
            for (int x = 0; x < w; x++) {
                transposed.channel(z).row(x)[y] = src_ptr[x];
            }
        }
    }

    return transposed;
}

static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat &pred,
                               float prob_threshold, std::vector<Object> &objects) {
    const int num_points = grid_strides.size();
    const int num_class = 19;

    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++) {
        const float *scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++) {
            float confidence = scores[k];
            if (confidence > score) {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold) {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void *) pred.row(i));
            {
                ncnn::Layer *softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++) {
                float dis = 0.f;
                const float *dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++) {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.x1 = x0;
            obj.y1 = y0;
            obj.x2 = x1;
            obj.y2 = y1;
            obj.classId = label;
            obj.confidence = box_prob;

            objects.push_back(obj);
        }
    }
}

static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID x1Id;
static jfieldID y1Id;
static jfieldID x2Id;
static jfieldID y2Id;
static jfieldID classIdId;
static jfieldID confidenceId;


static ncnn::UnlockedPoolAllocator blob_pool_allocator;
static ncnn::PoolAllocator workspace_pool_allocator;
static ncnn::Net yolo;

int target_size = 640;
float mean_vals[3] = {103.53f, 116.28f, 123.675f};
float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};


extern "C"
JNIEXPORT jboolean JNICALL
Java_vip_wqby_yolov11ncnn_yolov11_init(JNIEnv *env, jobject thiz, jobject manager,
                                       jboolean use_gpu) {

    jclass localObjCls = env->FindClass("vip/wqby/yolov11ncnn/OutResult");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));
    constructortorId = env->GetMethodID(objCls, "<init>", "()V");
    x1Id = env->GetFieldID(objCls, "x1", "I");
    y1Id = env->GetFieldID(objCls, "y1", "I");
    x2Id = env->GetFieldID(objCls, "x2", "I");
    y2Id = env->GetFieldID(objCls, "y2", "I");
    classIdId = env->GetFieldID(objCls, "classId", "I");
    confidenceId = env->GetFieldID(objCls, "confidence", "F");


    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    AAssetManager *mgr = AAssetManager_fromJava(env, manager);

    {
        int ret = yolo.load_param(mgr, "best.ncnn.param");
        if (ret != 0) {
            return JNI_FALSE;
        }
    }

    {
        int ret = yolo.load_model(mgr, "best.ncnn.bin");
        if (ret != 0) {
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}


extern "C"
JNIEXPORT jobjectArray JNICALL
Java_vip_wqby_yolov11ncnn_yolov11_detect(JNIEnv *env, jobject thiz, jobject bitmap) {

    float prob_threshold = 0.4;   //最低置信度
    float nms_threshold = 0.5;


    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w, h);

    // pad to target_size rectangle
    int wpad = (target_size + 31) / 32 * 32 - w;
    int hpad = (target_size + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 0.f);
    in_pad.substract_mean_normalize(0, norm_vals);


    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("in0", in_pad);
    std::vector<Object> proposals;
    ncnn::Mat out;
    ex.extract("out0", out);

/*    //输出in和out
    __android_log_print(ANDROID_LOG_ERROR, "yolov8ncnn", "%d %d", out.w,out.h);*/

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    ncnn::Mat permute = transpose_ncnn_mat(out);
    generate_proposals(grid_strides, permute, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    __android_log_print(ANDROID_LOG_ERROR, "yolov11ncnn", "count: %d", count);

    std::vector<Object> objects;
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].x1 - (wpad / 2)) / scale;
        float y0 = (objects[i].y1 - (hpad / 2)) / scale;
        float x1 = (objects[i].x2 - (wpad / 2)) / scale;
        float y1 = (objects[i].y2 - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

        objects[i].x1 = floor(x0);
        objects[i].y1 = floor(y0);
        objects[i].x2 = floor(x1);
        objects[i].y2 = floor(y1);

        __android_log_print(ANDROID_LOG_ERROR, "yolov11ncnn", "detect: %d %d %d %d %d",
                            objects[i].classId, (int) x0, (int) y0, (int) x1, (int) y1);
    }

    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
    for (size_t i = 0; i < objects.size(); i++) {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);

        env->SetIntField(jObj, x1Id, objects[i].x1);
        env->SetIntField(jObj, y1Id, objects[i].y1);
        env->SetIntField(jObj, x2Id, objects[i].x2);
        env->SetIntField(jObj, y2Id, objects[i].y2);

        env->SetIntField(jObj, classIdId, objects[i].classId);
        env->SetFloatField(jObj, confidenceId, objects[i].confidence);


        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    return jObjArray;
}