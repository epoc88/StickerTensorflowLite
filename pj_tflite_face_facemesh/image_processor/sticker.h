#ifndef SIICKER_H
#define STICKER_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "defines.h"
#include "config.h"
#include "utils.h"

namespace SelfieSticker
{
    class HatSticker{
    public:
        HatSticker();
        ~HatSticker(){};
        void init();
        cv::Mat landmarks_to_hat_matrix(std::vector<Point3_t<float>>& landmarks, 
                                        int width, 
                                        int height);

        cv::Mat apply(cv::Mat& src, std::vector<Point3_t<float>>& landmarks);
        StickerData sticker_data;
        Point2_t<float> kp_left;
        Point2_t<float> kp_right;
        cv::Mat HAT_POSE_MATRIX;
        cv::Mat sticker_image;
        bool verbose = false;
    };

    class EyeSticker{
    public:
        EyeSticker();
        EyeSticker(StickerData& data);
        ~EyeSticker(){};
        void init();
        cv::Mat landmarks_to_sticker_matrix(std::vector<Point3_t<float>>& landmarks, 
                                        int width, 
                                        int height);

        cv::Mat apply(cv::Mat& src, std::vector<Point3_t<float>>& landmarks);
        // mean coordinates for eye
        Point3_t<float> get_mean_point(std::vector<Point3_t<float>>& landmarks, std::vector<int>& idxs);
        
        cv::Mat xf_tri_to_tri(std::vector<Point2_t<float>> tri_src, 
                              std::vector<Point3_t<float>> tri_dst);
        
        cv::Mat xf_cat_pair(cv::Mat a,  cv::Mat b);
        
        cv::Mat xf_inv(cv::Mat x);
        
        void xf_from_base_vectors(Point2_t<float> i,  
                             Point2_t<float> j,
                             Point2_t<float> xlat,
                             cv::Mat& src_mat );

        StickerData sticker_data;
        Point2_t<float> kp_left_eye;
        Point2_t<float> kp_right_eye;
        cv::Mat sticker_image;
        std::vector<int> MEDIAPIPE_EYE_LEFT_CW_INDICES;
        std::vector<int> MEDIAPIPE_EYE_RIGHT_CW_INDICES;
        bool verbose = false;
    };
}

#endif //
