//
//  main.cpp
//  SpellingBee
//
//  Created by Daniel Oberg on 2015-11-27.
//  Copyright © 2015 United Lambdas. All rights reserved.
//

#include <algorithm>
#include <cstddef>
#include <map>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <iterator>

#include <glob.h>
#include <vector>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/locale.hpp>
#include <boost/filesystem/fstream.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <SFML/Audio.hpp>

#include "aquila/global.h"
#include "aquila/source/FramesCollection.h"
#include "aquila/transform/Mfcc.h"
#include "aquila/tools/TextPlot.h"
#include "aquila/source/window/BarlettWindow.h"
#include "aquila/source/WaveFile.h"

#include "DownloadFilesFromAzure.hh"


/*
 OPENCV CHEAT SHEET
 ==================
 
 Parameter Order
 ---------------
 Rows, Cols
 Height, Width
 at(row, cols)
 
 */


using namespace std;
using namespace cv;
using namespace cv::ml;

using std::ostream;

template<typename T>
ostream& operator<< (ostream& out, const vector<T>& v) {
    out << "[";
    for(const auto i : v) {
        out << i << ", ";
    }
    out << "]";
    return out;
}

template<typename A, typename B>
pair<B,A> flip_pair(const pair<A,B> &p)
{
    return pair<B,A>(p.second, p.first);
}


template<typename A, typename B>
map<B,A> flip_map(const map<const A, const B> &src)
{
    map<B,A> dst;
    transform(src.begin(), src.end(), inserter(dst, dst.begin()), flip_pair<A,B>);
    return dst;
}

sf::SoundBuffer recordToBuffer() {
    sf::SoundBufferRecorder recorder;
    std::cout << "Click when ready" << std::endl << std::flush;
    
    getchar();
    std::cout << "Start Talking" << std::endl << std::flush;
    recorder.start();
    getchar();
    recorder.stop();
    std::cout << "Analyzing" << std::endl << std::flush;
    
    const sf::SoundBuffer& buffer = recorder.getBuffer();
    
    return buffer;
}

void saveAudio(std::string filename_prefix, const sf::SoundBuffer& buffer) {
    for (size_t i=0; i<SIZE_T_MAX; i++) {
        std::string filename = filename_prefix + std::to_string(i) + ".wav";
        if (boost::filesystem::exists(filename)) {
            continue;
        } else {
            buffer.saveToFile(filename);
            break;
        }
    }
}

inline std::vector<std::string> glob(const std::string& pat){
    using namespace std;
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        ret.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return ret;
}

void recordAllHiraganaSounds(std::string foldername) {
    for (const auto &w : romajiToHiragana) {
        // Record some audio data
        sf::SoundBufferRecorder recorder;
        std::cout << "Say " << w.first << " " << w.second << std::endl << std::flush;
        
        getchar();
        std::cout << "Start Talking" << std::endl << std::flush;
        recorder.start(44100);
        getchar();
        recorder.stop();
        std::cout << "Analyzing" << std::endl << std::flush;
        
        // Get the buffer containing the captured audio data
        const sf::SoundBuffer& buffer = recorder.getBuffer();
        // Save it to a file (for example...)
        saveAudio(std::string("./") + foldername + std::string("/") + w.first, buffer);
    }
}

std::vector<float> pcmToJpg(std::string filename) {
     std::vector<double> vec_double;
    {
        boost::filesystem::path path(std::string("./raw/") + filename);
        if (!boost::filesystem::exists(path)) {
            std::cout << "fdsjafl";
        }

        std::ifstream file(std::string("./raw/") + filename, std::ios::in | std::ios::binary);
        
        std::vector<int16_t> vec;
        int16_t inf ;
        while( file.read( reinterpret_cast<char*>( &inf ), sizeof(int16_t) ) )
            vec.push_back(inf) ;
        
        file.close();
        
        for (const auto i : vec)
            vec_double.push_back(((double)i)/INT16_MAX);
        
    }
    
    Aquila::SignalSource buffer(vec_double, 44100);
    uint16_t FRAME_SIZE = 512; // 44100 / 100; // 44100 samples per second
    uint16_t MFCCS = 12;
    
    Aquila::FramesCollection frames(buffer, FRAME_SIZE);
    Aquila::Mfcc mfcc(FRAME_SIZE);
    
    cv::Mat1f result(0, MFCCS);
    
    for (const Aquila::Frame& frame : frames) {
        auto mfccValues = mfcc.calculate(frame, MFCCS);
        
        auto firstMfccs = std::vector<double>(mfccValues.begin(), mfccValues.begin()+MFCCS);
        
        cv::Mat1f row = cv::Mat1f::zeros(1, MFCCS);
        
        for (int i = 0; i < MFCCS; i++) {
            row.col(i) = firstMfccs.at(i) / 1;
        }
        result.push_back(row);
    }
    
    int ymin = result.rows;
    int ymax = 0;
    
    for (int x = 0; x < result.cols; x++) {
        for (int y = 0; y < result.rows; y++) {
            auto pixel = result(y, x);
            
            if (pixel > 0.5) {
                if (y < ymin) {
                    ymin = y;
                }
                
                if (y > ymax) {
                    ymax = y;
                }
            }
        }
    }
    cv::Mat out = cv::Mat::zeros(result.size(), result.type());
    result(cv::Rect(0,ymin, result.cols,result.rows-ymin)).copyTo(out(cv::Rect(0,0,result.cols,result.rows-ymin)));

    cv::Mat1f correctSize;
    cv::resize(out, correctSize, cv::Size(12, 60), 0, 0, cv::INTER_CUBIC);
    
    cv::Mat transposed = correctSize.t();
    
    cv::Mat grayImage;
    transposed.convertTo(grayImage, CV_8U, 255.0);
    
    imwrite( "./jpg/" + filename + ".jpg", grayImage );
    
    std::vector<float> array;
    if (correctSize.isContinuous()) {
        array.assign((float*)correctSize.datastart, (float*)correctSize.dataend);
    } else {
        for (int i = 0; i < correctSize.rows; ++i) {
            array.insert(array.end(), (float*)correctSize.ptr<uchar>(i), (float*)correctSize.ptr<uchar>(i)+correctSize.cols);
        }
    }
    
    return array;
}

std::map<size_t, std::string> romajiFromClassIndex() {
    std::map<size_t, std::string> result;
    
    size_t i = 0;
    for (const auto &w : romajiToHiragana) {
        result[i] = w.first;
        i++;
    }
    return result;
}

std::map<std::string, size_t> classIndexFromRomaji() {
    std::map<std::string, size_t> result;
    
    size_t i = 0;
    for (const auto &w : romajiToHiragana) {
        result[w.first] = i;
        i++;
    }
    return result;
}

int main(int argc, char *argv[])
{
    using namespace boost::locale::boundary;
    boost::locale::generator gen;
    std::locale loc = gen("ja_JP.UTF-8");
    
    downloadAll();
    
    auto filenames = glob("./raw/*.pcm");
    
    for (const auto filename : filenames) {
        boost::filesystem::path path(filename);
        pcmToJpg(path.filename().string());
    }
    
    return 0;
}
