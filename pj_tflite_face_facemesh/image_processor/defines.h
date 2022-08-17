#ifndef SELFIE_STICKER_DEFINES_H
#define SELFIE_STICKER_DEFINES_H

#include <string>
#include <vector>
#include <map>

#define LOG(x) std::cerr
#define LOGE std::cerr
#define LOGD std::cout


    template<class T>
    struct Point2_t {
        T x;
        T y;
        Point2_t(){};
        Point2_t(T _x,T _y){
            x = _x;
            y = _y;
        };
        void set_x(T _x){ x = _x;}
        void set_y(T _y){ y = _y;}
    };
    
    template<class T>
    struct Point3_t {
        T x;
        T y;
        T z;
        Point3_t(){};
        Point3_t(T _x, T _y, T _z){
            x = _x;
            y = _y;
            z = _z;
        };
        void set_x(T _x){ x = _x;}
        void set_y(T _y){ y = _y;}
        void set_z(T _z){ y = _z;}
    };


namespace SelfieSticker
{

    enum NormalType {
        RAW_PIXEL             = 0,
        FLOAT_ZERO_ONE        = 1,
        FLOAT_MINUS_PLUS_ONE  = 2
    };
    
    enum BorderMode {
        BORDER_UNSPECIFIED = 0,
        BORDER_ZERO = 1,
        BORDER_REPLICATE = 2
    };
    
    struct StickerData{
        std::string type = "hat";
        std::string image_path = "../data/emoji_celebration.png";
        std::vector<int> kp_idxs = {127,356,168};
        float depth_offset = 0;
        std::string transformation_model = "3d";
    };

    struct Stickers{
        std::vector<StickerData> sticker_items;
        std::vector<float> lip_color = {};
    };

    struct DetectionSpec {
        std::pair<int, int> image_size;
    };
    

    struct RotatedRect {
        float center_x = 0;
        float center_y = 0;
        float width = 0;
        float height = 0;
        float rotation = 0;
    };
    
    struct ValueTransformation {
        float scale;
        float offset;
    };
    
    struct NormalizedRect {
        float x_center = 0;
        float y_center = 0;
        float height = 0;
        float width = 0;
        float rotation = 0;
        NormalizedRect(){};
        NormalizedRect(float _x_center, float _y_center, float _height, float _width, float _rotation){
            x_center = _x_center;
            y_center = _y_center;
            height = _height;
            width = _width;
            rotation = _rotation;
        };
    };

    struct Anchor {
        float x_center;
        float y_center;
        float h;
        float w;
        void set_x_center(float _x_center) {x_center = _x_center;}
        void set_y_center(float _y_center) {y_center = _y_center;}
        void set_w(float _w) {w = _w;}
        void set_h(float _h) {h = _h;}
    };

    // TODO replace NormalizedDetection 
    struct NormDetection {
        float x1;
        float y1;
        float x2;
        float y2;
        std::array<Point2_t<float>,6> keypoints;
        NormDetection(){};
        NormDetection(float _x1, float _y1, float _x2, float _y2){
            x1 = _x1;
            y1 = _y1;
            x2 = _x2;
            y2 = _y2;
        }; 
        int Area(){
            return (x2-x1)*(y2-y1);
        }
    };  

    struct Detection {
        int x1;
        int y1;
        int x2;
        int y2;
        std::array<Point2_t<int>,6> keypoints;
        Detection(){};
        Detection(int _x1,int _y1,int _x2,int _y2){
            x1 = _x1;
            y1 = _y1;
            x2 = _x2;
            y2 = _y2;
        };
        Detection(float _x1, float _y1, float _x2, float _y2){
            x1 = _x1;
            y1 = _y1;
            x2 = _x2;
            y2 = _y2;
        }; 
        int Area(){
            return (x2-x1)*(y2-y1);
        }
    };


    typedef std::vector<Detection> Detections;
    typedef std::vector<NormDetection> NormDetections;
    typedef std::vector<std::pair<int, float>> IndexedScores;

    typedef std::vector<float> NormalizedDetection;
    typedef std::vector<NormalizedDetection> NormalizedDetections;
}

#endif //SELFIE_STICKER_DEFINES_H
