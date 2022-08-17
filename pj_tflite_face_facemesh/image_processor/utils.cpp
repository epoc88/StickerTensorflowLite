/*
#include <sys/uio.h> 
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <iomanip>

#include <ctime>
#include <sys/time.h>
#include <chrono>


namespace SelfieSticker
{

long long get_timestamp(){

    std::chrono::milliseconds timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    return timestamp.count();
}

}
*/

#include "utils.h"

long long get_timestamp(){

        std::chrono::milliseconds timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            return timestamp.count();
}

