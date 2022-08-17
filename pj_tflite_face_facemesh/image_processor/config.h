#ifndef SELFIE_STICKER_CONFIG_H
#define SELFIE_STICKER_CONFIG_H

#include <string> 
#include <vector>
#include <array>
#include "defines.h"

namespace SelfieSticker{
    
    struct RectTransformationOptions {
        int rotation_vector_start_keypoint_index = 0;
        int rotation_vector_end_keypoint_index = 1;
        float rotation_vector_target_angle_degrees = 0;
        float scale_x = 1.5;
        float scale_y = 1.5;
        bool square_long = true;
        bool square_short = false; 
        float rotation = 0;
        float shift_x = 0;
        float shift_y = 0;
    };

    struct NonMaxSuppressionOptions {
        //bool verbose = false; 
        float min_suppression_threshold = 0.3;
        int max_num_detections = 1;
        std::string overlap_type = "INTERSECTION_OVER_UNION";
        std::string algorithm = "WEIGHTED";
        int keypoint_coord_offset = 4;
        int num_keypoints = 6;
        int num_values_per_keypoint = 2;
    };

    struct SsdAnchorsCalculatorOptions {
        std::vector<float> aspect_ratios = {1.0f};
        std::vector<int> strides = {8,16,16,16};
        int num_layers = 4;
        int strides_size = 4;
        int aspect_ratios_size = 1;
        float min_scale = 0.1484375;
        float max_scale = 0.75;
        float interpolated_scale_aspect_ratio = 1.0;
        int input_size_height = 128;
        int input_size_width = 128;

        float x_scale = 128.0;
        float y_scale = 128.0;
        float w_scale = 128.0;
        float h_scale = 128.0;

        float anchor_offset_x = 0.5;
        float anchor_offset_y = 0.5;
        bool fixed_anchor_size = true;
        int  num_coords = 16;
        int keypoint_coord_offset = 4;
        int num_keypoints = 6;
        int num_values_per_keypoint = 2;
    };

    struct Settings {
        bool verbose = false; 
        
        NormalType detect_norm_type   =  FLOAT_MINUS_PLUS_ONE;
        NormalType landmark_norm_type   =  FLOAT_ZERO_ONE;
        
        std::string image_path = "../data/lena.jpg";
        
        int detect_regression_index = 0;
        int detect_classifier_index = 1;
        float score_clipping_thresh = 100.0;
        bool keep_aspect_ratio = true;
        bool landmark_keep_aspect_ratio = false;
        BorderMode border_mode = BORDER_ZERO;
        BorderMode landmark_border_mode = BORDER_ZERO;

        // landmark result 
        int landmark_all_index = 0;
        int landmark_dims = 3;
        int landmark_all_predict_tensor_size = 468;
        
        int n_detect_threads = 2;
        int n_landmark_threads = 2;
        
        float detect_confidence = 0.5;
        int detect_result_width=16;
        int face_detect_predict_tensor_size = 896;

        int detect_wanted_width = 128;
        int detect_wanted_height = 128;
        int detect_wanted_channels = 1;

        int landmark_wanted_width = 192;
        int landmark_wanted_height = 192;
        int landmark_wanted_channels = 1;
        
        float input_mean = 127.5f;
        float input_std = 127.5f;
        
        SsdAnchorsCalculatorOptions ssd_options;
        NonMaxSuppressionOptions nms_options;
        RectTransformationOptions rect_transform_options;

        const int keypoint_nums = 6;
        const int bbox_dim = 4;
        const int point_dim = 2;

        //std::string face_detect_model_path = "../weights/face_detection_short_range.tflite";
        std::string face_detect_model_path = "";
        char* detect_model_buffer; 
        long  detect_model_buffer_size;

        //std::string face_landmark_model_path = "../weights/face_landmark.tflite";
        std::string face_landmark_model_path = "";
        char* landmark_model_buffer; 
        long  landmark_model_buffer_size;
        Stickers stickers;
    };

}

#endif //SELFIE_STICKER_CONFIG_H
