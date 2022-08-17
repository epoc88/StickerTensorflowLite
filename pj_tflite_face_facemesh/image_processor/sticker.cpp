#include "sticker.h"

namespace SelfieSticker
{

    void EyeSticker::init(){
        
        kp_left_eye.x = 180.0;
        kp_left_eye.y = 255.0;
        kp_right_eye.x = 350.0;
        kp_right_eye.y = 255.0;

        sticker_image = cv::imread(sticker_data.image_path,cv::IMREAD_UNCHANGED);

        MEDIAPIPE_EYE_LEFT_CW_INDICES = {243, 112, 26, 22, 23, 24, 110, 25, 130, 247, 30, 29, 27, 28, 56, 190};
        MEDIAPIPE_EYE_RIGHT_CW_INDICES = {463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414};

    }
    
    EyeSticker::EyeSticker(){
        init(); 
    }
    
    EyeSticker::EyeSticker(StickerData& data){
        sticker_data = sticker_data;
        init(); 
    }

    
    Point3_t<float> EyeSticker::get_mean_point(std::vector<Point3_t<float>>& landmarks, 
                                               std::vector<int>& idxs){
        Point3_t<float> mean_point(0,0,0);
        Point3_t<float> kp; 
        for(int i = 0; i < idxs.size(); i++){
            kp = landmarks[idxs[i]];
            mean_point.x += kp.x;
            mean_point.y += kp.y;
            mean_point.z += kp.z;
        }
        int number = idxs.size();
        mean_point.x /= number;
        mean_point.y /= number;
        mean_point.z /= number;
        return mean_point;
    }

    cv::Mat EyeSticker::landmarks_to_sticker_matrix(std::vector<Point3_t<float>>& landmarks,
                                                int width, int height){
        for (auto& kp: landmarks){
            kp.x *= width;
            kp.y *= height;
            kp.z *= width;
        }

        std::vector<Point3_t<float>> part_kps = {landmarks[sticker_data.kp_idxs[0]],
                                                 landmarks[sticker_data.kp_idxs[1]],
                                                 landmarks[sticker_data.kp_idxs[2]]};
        
        Point3_t<float> mesh_left = part_kps[0]; 
        Point3_t<float> mesh_right = part_kps[1]; 
        Point3_t<float> mesh_front = part_kps[2]; 

        
        Point3_t<float> mesh_left_eye = get_mean_point(landmarks,MEDIAPIPE_EYE_LEFT_CW_INDICES);
        Point3_t<float> mesh_right_eye = get_mean_point(landmarks,MEDIAPIPE_EYE_RIGHT_CW_INDICES);
        
        // Test
        /*
        mesh_left.x = 311.11; mesh_left.y = 717.44; mesh_left.z = 148.42;
        mesh_right.x = 711.93; mesh_right.y = 712.03; mesh_right.z = 158.99;
        mesh_front.x = 515.47; mesh_front.y = 730.70; mesh_front.z = -59.70;
        mesh_left_eye.x = 416.33; mesh_left_eye.y=733.15; mesh_left_eye.z = -12.72;
        mesh_right_eye.x = 613.59; mesh_right_eye.y=730.78; mesh_right_eye.z = -7.70;
        */
        /*
        std::cout<<"mesh left: "<< mesh_left.x << " " 
                                << mesh_left.y << " "
                                << mesh_left.z << std::endl;
        std::cout<<"mesh right: "<< mesh_right.x << " " 
                                 << mesh_right.y << " "
                                 << mesh_right.z << std::endl;
        std::cout<<"mesh front: "<< mesh_front.x << " " 
                                << mesh_front.y << " "
                                << mesh_front.z << std::endl;
        std::cout<<"left eye: "<< mesh_left_eye.x << " " << mesh_left_eye.y << std::endl;
        std::cout<<"right eye: "<< mesh_right_eye.x << " " << mesh_right_eye.y << std::endl;
        */
        
        Point3_t<float> mesh_center;
        mesh_center.x = (mesh_left.x + mesh_right.x)/2.0;
        mesh_center.y = (mesh_left.y + mesh_right.y)/2.0;
        mesh_center.z = (mesh_left.z + mesh_right.z)/2.0;
 
        Point3_t<float> mesh_origin;
        mesh_origin.x = (mesh_left_eye.x + mesh_right_eye.x)/2.0;
        mesh_origin.y = (mesh_left_eye.y + mesh_right_eye.y)/2.0;
        mesh_origin.z = (mesh_left_eye.z + mesh_right_eye.z)/2.0;
        
        cv::Vec3f mesh_dir_right;
        mesh_dir_right[0] = mesh_right.x - mesh_left.x;
        mesh_dir_right[1] = mesh_right.y - mesh_left.y;
        mesh_dir_right[2] = mesh_right.z - mesh_left.z;
        
        cv::Vec3f mesh_dir_right_eye;
        mesh_dir_right_eye[0] = mesh_right_eye.x - mesh_left_eye.x;
        mesh_dir_right_eye[1] = mesh_right_eye.y - mesh_left_eye.y;
        mesh_dir_right_eye[2] = mesh_right_eye.z - mesh_left_eye.z;

        cv::Vec3f mesh_dir_fwd;
        mesh_dir_fwd[0] = mesh_front.x - mesh_center.x;
        mesh_dir_fwd[1] = mesh_front.y - mesh_center.y;
        mesh_dir_fwd[2] = mesh_front.z - mesh_center.z;
        
        
        cv::Vec3f mesh_dir_up = mesh_dir_fwd.cross(mesh_dir_right);
        /*
        std::cout << "mesh_dir_right " 
                  << mesh_dir_right[0] << " "
                  << mesh_dir_right[1] << " "
                  << mesh_dir_right[2] << std::endl;
        std::cout << "mesh_dir_fwd " 
                  << mesh_dir_fwd[0]<< " "
                  << mesh_dir_fwd[1]<< " "
                  << mesh_dir_fwd[2]<< std::endl;
        std::cout << "mesh_dir_up " 
                  << mesh_dir_up[0]<< " "
                  << mesh_dir_up[1]<< " "
                  << mesh_dir_up[2]<< std::endl;
        */
        float nrm = cv::norm(mesh_dir_up, cv::NORM_L2);
        float nrm2 = cv::norm(mesh_dir_right_eye, cv::NORM_L2);
        
        cv::Vec3f mesh_dir_up_nrm(mesh_dir_up[0]/nrm, mesh_dir_up[1]/nrm,mesh_dir_up[2]/nrm);
        
        Point3_t<float> mesh_peak;
        mesh_peak.x = mesh_origin.x + mesh_dir_up_nrm[0] * nrm2 *0.5;
        mesh_peak.y = mesh_origin.y + mesh_dir_up_nrm[1] * nrm2 *0.5;
        mesh_peak.z = mesh_origin.z + mesh_dir_up_nrm[2] * nrm2 *0.5;
        /*
        std::cout << "mesh_peak " 
                  << mesh_peak.x<< " "
                  << mesh_peak.y<< " "
                  << mesh_peak.z<< std::endl;
        */
        Point2_t<float> img_origin((kp_left_eye.x + kp_right_eye.x)/2, 
                                   (kp_left_eye.y + kp_right_eye.y)/2);
        /*
        std::cout << "img_origin " 
                  << img_origin.x<< " "
                  << img_origin.y<< std::endl;
        */
        cv::Vec2f img_dir_right((kp_right_eye.x - kp_left_eye.x), 
                                (kp_right_eye.y - kp_left_eye.y));
        /*
        std::cout << "img_dir_right " 
                  << img_dir_right[0]<< " "
                  << img_dir_right[1]<< std::endl;
        */
        float img_right_nrm = cv::norm(img_dir_right,cv::NORM_L2);
        
        Point2_t<float> img_dir_right_nrm(img_dir_right[0] /img_right_nrm, 
                                          img_dir_right[1] /img_right_nrm);
        /*
        std::cout << "img_dir_right_nrm " 
                  << img_dir_right_nrm.x<< " "
                  << img_dir_right_nrm.y<< std::endl;
        */
        Point2_t<float> img_dir_up_nrm(img_dir_right_nrm.y, -1*img_dir_right_nrm.x);
        /*
        std::cout << "img_dir_up_nrm " 
                  << img_dir_up_nrm.x<< " "
                  << img_dir_up_nrm.y<< std::endl;
        */
        Point2_t<float> img_peak;
        img_peak.x = img_origin.x + img_dir_up_nrm.x * img_right_nrm*0.5; 
        img_peak.y = img_origin.y + img_dir_up_nrm.y * img_right_nrm*0.5; 
        /*
        std::cout << "img_peak " 
                  << img_peak.x<< " "
                  << img_peak.y<< std::endl;
        */
        std::vector<Point2_t<float>> src = {kp_left_eye, kp_right_eye, img_peak};
        std::vector<Point3_t<float>> dst = {mesh_left_eye, mesh_right_eye, mesh_peak};

        cv::Mat m = xf_tri_to_tri(src, dst);
        return m;
    }
  
    cv::Mat EyeSticker::xf_inv(cv::Mat x){
        cv::Mat inv_x;
        
        cv::Mat x_square = x(cv::Rect(0,0,2,2));
        
        cv::invert(x_square, inv_x);
        cv::Mat ix = inv_x * -x(cv::Rect(2,0,1,2));
        
        cv::Mat result(2,3,CV_32F);
        cv::hconcat(inv_x, ix, result);
        //std::cout << "result "<<std::endl;
        //std::cout << cv::format(result,cv::Formatter::FMT_PYTHON) << std::endl;
        return result;
    }

    void EyeSticker::xf_from_base_vectors(Point2_t<float> i,  
                                          Point2_t<float> j,
                                          Point2_t<float> xlat,
                                          cv::Mat& result){
        if (xlat.x== 0 and xlat.y == 0){
            float data1[2][3] = {i.x, j.x, 0.0,
                                 i.y, j.y, 0.0}; 
            result= cv::Mat(2,3,CV_32F,data1).clone();
        }else{
            float data2[2][3] = {i.x, j.x, xlat.x,
                                 i.y, j.y, xlat.y}; 
            result = cv::Mat(2,3, CV_32F,data2).clone();
            //std::cout << "result "<< &result << " " << result.type() << std::endl;
            //std::cout << cv::format(result,cv::Formatter::FMT_PYTHON) << std::endl;
        }
    }

    cv::Mat EyeSticker::xf_cat_pair(cv::Mat a,  cv::Mat b){
        cv::Mat ab2 = a(cv::Rect(0,0,2,2)) * b(cv::Rect(0,0,2,2));      

        cv::Mat cola = a(cv::Rect(2,0,1,2));
        
        cv::Mat colb = b(cv::Rect(2,0,1,2));

        cv::Mat abx = a(cv::Rect(0,0,2,2)) * colb + cola;

        cv::Mat result(2,3,CV_32F);
        cv::hconcat(ab2,abx,result);
        //std::cout << "result "<<std::endl;
        //std::cout << cv::format(result,cv::Formatter::FMT_PYTHON) << std::endl;
        return result;
    }
   
    cv::Mat EyeSticker::xf_tri_to_tri(std::vector<Point2_t<float>> tri_src, 
                                      std::vector<Point3_t<float>> tri_dst){
        
        
        Point2_t<float> tri_src_diff1(tri_src[1].x-tri_src[0].x, tri_src[1].y-tri_src[0].y);
        Point2_t<float> tri_src_diff2(tri_src[2].x-tri_src[0].x, tri_src[2].y-tri_src[0].y);
        
        //std::cout << "src diff1 "<< tri_src_diff1.x << " " << tri_src_diff1.y << std::endl;
        //std::cout << "src diff2 "<< tri_src_diff2.x << " " << tri_src_diff2.y << std::endl;
        
        cv::Mat src;
        xf_from_base_vectors(tri_src_diff1, tri_src_diff2, tri_src[0], src);
        //std::cout << "src "<< &src <<std::endl;
        //std::cout << cv::format(src,cv::Formatter::FMT_PYTHON) << std::endl;
        
        Point2_t<float> tri_dst_diff1(tri_dst[1].x-tri_dst[0].x, tri_dst[1].y-tri_dst[0].y);
        Point2_t<float> tri_dst_diff2(tri_dst[2].x-tri_dst[0].x, tri_dst[2].y-tri_dst[0].y);
        Point2_t<float> tri_dst_0(tri_dst[0].x, tri_dst[0].y);
        
        cv::Mat dst;
        xf_from_base_vectors(tri_dst_diff1, tri_dst_diff2, tri_dst_0, dst);
        //std::cout << "dst "<<std::endl;
        //std::cout << cv::format(dst,cv::Formatter::FMT_PYTHON) << std::endl;
        
        cv::Mat inv_src = xf_inv(src);
        return xf_cat_pair(dst, inv_src);    
    }

    cv::Mat EyeSticker::apply(cv::Mat& src, 
                              std::vector<Point3_t<float>>& landmarks){
        int w = src.cols;
        int h = src.rows;

        long long start_time;
        start_time = get_timestamp();
        cv::Mat m_affine = landmarks_to_sticker_matrix(landmarks, w, h);
        LOGD << "landmark to sticker matrix time: " << get_timestamp() - start_time << "\n";
        //std::cout << "m "<<std::endl;
        //std::cout << cv::format(m_affine,cv::Formatter::FMT_PYTHON) << std::endl;
        
        start_time = get_timestamp();
        cv::Mat aligned_sticker_image; 
        cv::warpAffine(sticker_image, 
                       aligned_sticker_image, 
                       m_affine, 
                       cv::Size(w,h), 
                       cv::INTER_LINEAR);
        LOGD << "warpAffine time: " << get_timestamp() - start_time << "\n";
        
        //cv::imwrite("aligned_eye.png", aligned_sticker_image);
        
        start_time = get_timestamp();
        cv::Mat channels[4];
        cv::split(aligned_sticker_image, channels);
        cv::Mat alpha = channels[3];

        cv::cvtColor(alpha, alpha, cv::COLOR_GRAY2RGB);
        alpha.convertTo(alpha, CV_32FC3);

        alpha = alpha / 255.0;

        cv::Mat beta = cv::Scalar(1,1,1)-alpha;
        LOGD << "compose time 0: " << get_timestamp() - start_time << "\n";
        
        cv::Mat sticker_rgb;
        std::vector<cv::Mat> rgb_channels = {channels[2],channels[1],channels[0]};
        cv::merge(rgb_channels,sticker_rgb);
        
        sticker_rgb.convertTo(sticker_rgb, CV_32FC3);
        src.convertTo(src, CV_32FC3);
       
        LOGD << "compose time 1: " << get_timestamp() - start_time << "\n";
        src = src.mul(beta);
        sticker_rgb = sticker_rgb.mul(alpha);
        cv::add(src,sticker_rgb,src);
        LOGD << "compose time 2: " << get_timestamp() - start_time << "\n";
        
        src.convertTo(src, CV_8UC3);
        cv::cvtColor(src, src, cv::COLOR_RGB2BGR);

        return src;
    } 
//--------------------------------------------------------


    void HatSticker::init(){
        float k[3][3] = {1.0, 0.0, 0.0, 
                         0.0, 1.0, 1.4,
                         0.0, 0.0, 1.0};

        HAT_POSE_MATRIX = cv::Mat(3,3,CV_32F,k);
        //std::cout << "init hat pose "<< &HAT_POSE_MATRIX <<std::endl;
        //std::cout << cv::format(HAT_POSE_MATRIX,cv::Formatter::FMT_PYTHON) << std::endl;
        
        kp_left.x = 190.0;
        kp_left.y = 60.0;
        kp_right.x = 820.0;
        kp_right.y = 60.0;

        sticker_image = cv::imread(sticker_data.image_path,cv::IMREAD_UNCHANGED);
        //cv::cvtColor(sticker_image, sticker_image, cv::COLOR_BGR2RGBA); 
    }

    HatSticker::HatSticker(){
        init(); 
    }

    cv::Mat HatSticker::landmarks_to_hat_matrix(std::vector<Point3_t<float>>& landmarks,
                                                int width, int height){
        
        Point3_t<float> mesh_left = landmarks[sticker_data.kp_idxs[0]]; 
        Point3_t<float> mesh_right = landmarks[sticker_data.kp_idxs[1]]; 
        Point3_t<float> mesh_front = landmarks[sticker_data.kp_idxs[2]]; 
        
        mesh_left.x *= width;
        mesh_left.y *= height;
        mesh_left.z *= width;

        mesh_right.x *= width;
        mesh_right.y *= height;
        mesh_right.z *= width;

        mesh_front.x *= width;
        mesh_front.y *= height;
        mesh_front.z *= width;

        if(verbose){
            std::cout << "mesh_left " << mesh_left.x<< std::endl;
            std::cout << "mesh_left " << mesh_left.y<< std::endl;
            std::cout << "mesh_left " << mesh_left.z<< std::endl;
            
            std::cout << "mesh_right " << mesh_right.x<< std::endl;
            std::cout << "mesh_right " << mesh_right.y<< std::endl;
            std::cout << "mesh_right " << mesh_right.z<< std::endl;
        } 
        // test new init

        Point3_t<float> mesh_center;
        mesh_center.x = (mesh_left.x + mesh_right.x)/2.0;
        mesh_center.y = (mesh_left.y + mesh_right.y)/2.0;
        mesh_center.z = (mesh_left.z + mesh_right.z)/2.0;
        
        cv::Vec3f mesh_dir_right;
        mesh_dir_right[0] = mesh_right.x - mesh_left.x;
        mesh_dir_right[1] = mesh_right.y - mesh_left.y;
        mesh_dir_right[2] = mesh_right.z - mesh_left.z;

        cv::Vec3f mesh_dir_fwd;
        mesh_dir_fwd[0] = mesh_front.x - mesh_center.x;
        mesh_dir_fwd[1] = mesh_front.y - mesh_center.y;
        mesh_dir_fwd[2] = mesh_front.z - mesh_center.z;

        cv::Vec3f mesh_origin;
        mesh_origin[0] = mesh_center.x;
        mesh_origin[1] = mesh_center.y;
        mesh_origin[2] = mesh_center.z;
        
        if(verbose){
            std::cout << "mesh_dir_fwd " << mesh_dir_fwd<< std::endl;
            std::cout << "mesh_dir_right " << mesh_dir_right<< std::endl;
        } 
        cv::Vec3f mesh_dir_up = mesh_dir_fwd.cross(mesh_dir_right);
        
        float nrm = cv::norm(mesh_dir_up, cv::NORM_L2);
        float nrm2 = cv::norm(mesh_dir_right, cv::NORM_L2);
        
        cv::Vec3f mesh_dir_up_nrm(mesh_dir_up[0]/nrm, mesh_dir_up[1]/nrm,mesh_dir_up[2]/nrm);
        
        mesh_dir_up = mesh_dir_up_nrm * nrm2;

        float mesh_m_data[3][3] = {mesh_dir_right[0]*0.5, mesh_dir_up[0]*0.5,mesh_origin[0],
                                   mesh_dir_right[1]*0.5, mesh_dir_up[1]*0.5,mesh_origin[1],
                                   0,0,1.0};

        cv::Mat mesh_m(3,3,CV_32F,mesh_m_data);
        
        Point2_t<float> img_center((kp_left.x+kp_right.x)/2, (kp_left.y+kp_right.y)/2);
        Point2_t<float> img_h((kp_right.x-kp_left.x)/2, (kp_right.y-kp_left.y)/2);
        Point2_t<float> img_v(img_h.y, -1*img_h.x);

        float img_data[3][3] = {img_h.x, img_v.x, img_center.x,
                                img_h.y, img_v.y, img_center.y,
                                0,0,1};
        cv::Mat img_m(3,3,CV_32F,img_data);
        cv::Mat inv_img_m;
        cv::invert(img_m, inv_img_m);
            
        // TODO 必须重新赋值 比较奇怪   
        float k[3][3] = {1.0, 0.0, 0.0, 
                         0.0, 1.0, 1.4,
                         0.0, 0.0, 1.0};
        HAT_POSE_MATRIX = cv::Mat(3,3,CV_32F,k);

        
        cv::Mat m = mesh_m * HAT_POSE_MATRIX * inv_img_m;  
        cv::Mat m2 = m(cv::Rect(0,0,3,2));
        //std::cout << "m2 "<<std::endl;
        //std::cout << cv::format(m2,cv::Formatter::FMT_PYTHON) << std::endl;
        return m2;
    }
    
    cv::Mat HatSticker::apply(cv::Mat& src, 
                              std::vector<Point3_t<float>>& landmarks){
        int w = src.cols;
        int h = src.rows;
        cv::Mat m_affine = landmarks_to_hat_matrix(landmarks, w, h);
        
        //cv::imwrite("origin_hat.png", sticker_image);
        cv::Mat aligned_sticker_image; 
        cv::warpAffine(sticker_image, 
                       aligned_sticker_image, 
                       m_affine, 
                       cv::Size(w,h), 
                       cv::INTER_LINEAR);
        
        //cv::imwrite("aligned_hat.png", aligned_sticker_image);
        cv::Mat channels[4];
        cv::split(aligned_sticker_image, channels);
        cv::Mat alpha = channels[3];

        cv::cvtColor(alpha, alpha, cv::COLOR_GRAY2RGB);
        alpha.convertTo(alpha, CV_32FC3);

        alpha = alpha / 255.0;

        cv::Mat beta = cv::Scalar(1,1,1)-alpha;
        
        cv::Mat hat;
        std::vector<cv::Mat> rgb_channels = {channels[2],channels[1],channels[0]};
        cv::merge(rgb_channels,hat);
        
        hat.convertTo(hat, CV_32FC3);
        src.convertTo(src, CV_32FC3);
       
        src = src.mul(beta);
        hat = hat.mul(alpha);
        cv::add(src,hat,src);
        
        src.convertTo(src, CV_8UC3);
        cv::cvtColor(src, src, cv::COLOR_RGB2BGR);
        
        //cv::imwrite("result1.jpg", src);
        return src;
    }

}

/*
int main(){
    SelfieSticker::HatSticker hat_sticker;
    
    SelfieSticker::StickerData sticker_data;
    sticker_data.kp_idxs = {0,1,2};
    hat_sticker.sticker_data = sticker_data;

    std::vector<SelfieSticker::Point3_t<float>> landmarks; 
    SelfieSticker::Point3_t<float> mesh_left;
    mesh_left.x = 371.25;
    mesh_left.y = 713.85;
    mesh_left.z = 152.172;
    
    SelfieSticker::Point3_t<float> mesh_right = {766.040,719.4368,141.04};
    SelfieSticker::Point3_t<float> mesh_front(564.25987244, 730.42693734, -59.13696);

    landmarks.push_back(mesh_left);
    landmarks.push_back(mesh_right);
    landmarks.push_back(mesh_front);
    
    //hat_sticker.landmarks_to_hat_matrix(landmarks);
    cv::Mat src = cv::imread("../data/lena.jpg");
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
    cv::flip(src, src, 1);
    hat_sticker.apply(src, landmarks);
    std::cout << "test success" << std::endl;
    return 0;
} 
*/
