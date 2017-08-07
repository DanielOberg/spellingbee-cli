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
#include <cmath>


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
    uint16_t FRAME_SIZE = 256; // 44100 / 100; // 44100 samples per second
    uint16_t MFCCS = 12;
    
    Aquila::FramesCollection frames(buffer, FRAME_SIZE);
    Aquila::Mfcc mfcc(FRAME_SIZE);
    
    cv::Mat1f mfccMat(0, MFCCS);
    
    for (const Aquila::Frame& frame : frames) {
        auto mfccValues = mfcc.calculate(frame, MFCCS);
        
        auto firstMfccs = std::vector<double>(mfccValues.begin(), mfccValues.begin()+MFCCS);
        
        cv::Mat1f row = cv::Mat1f::zeros(1, MFCCS);
        
        for (int i = 0; i < MFCCS; i++) {
            row.col(i) = firstMfccs.at(i) / 0.25;
        }
        mfccMat.push_back(row);
    }
    mfccMat = mfccMat.t();
    
    cv::Mat1f normalized;
    cv::normalize(mfccMat, normalized, 1, -0.2, cv::NORM_MINMAX);
    
    cv::Mat1f deltaMfccs(normalized.rows, normalized.cols);
    {
        for (int r = 0; r < normalized.rows; r++) {
            for (int c = 0; c < normalized.cols; c++) {
                int h = (c - 1 == -1)? c : c - 1;
                int j = (c + 1 == normalized.cols)? c : c + 1;
                
                double result = (normalized[r][j] - normalized[r][h]) / 2.0;
                deltaMfccs[r][c] = result;
            }
        }
    }
    cv::normalize(deltaMfccs, deltaMfccs, 1, -0.2, cv::NORM_MINMAX);

    cv::Mat1f deltaDeltaMfccs(deltaMfccs.rows, deltaMfccs.cols);
    {
        for (int r = 0; r < deltaMfccs.rows; r++) {
            for (int c = 0; c < deltaMfccs.cols; c++) {
                int h = (c - 1 == -1)? c : c - 1;
                int j = (c + 1 == deltaMfccs.cols)? c : c + 1;
                
                double result = (deltaMfccs[r][j] - deltaMfccs[r][h]) / 2.0;
                deltaDeltaMfccs[r][c] = result;
            }
        }
    }
    cv::normalize(deltaDeltaMfccs, deltaDeltaMfccs, 1, -0.2, cv::NORM_MINMAX);

    cv::Mat concatenated;
    
    cv::vconcat(mfccMat, deltaMfccs, concatenated);
    cv::vconcat(concatenated, deltaDeltaMfccs, concatenated);
    
    int xmin = mfccMat.cols - 1;
    int xmax = 0;
    for (int x = 0; x < mfccMat.cols; x++) {
        for (int y = 0; y < mfccMat.rows; y++) {
            auto pixel = mfccMat[y][x];
            
            if (pixel > 0.3) {
                if (x < xmin) {
                    xmin = x;
                }
                if (x > xmax) {
                    xmax = x;
                }
            }
        }
    }
    if (xmin + 3 >= xmax) {
        xmin = 0;
        xmax = mfccMat.cols - 1;
    }
    
    const int MAX_WIDTH = 96;
    
    if (xmax-xmin > MAX_WIDTH)
        xmax = xmin + MAX_WIDTH;
    
    cv::Mat out = cv::Mat::zeros(concatenated.rows, MAX_WIDTH, concatenated.type());
    concatenated(cv::Rect(xmin,0, xmax - xmin,concatenated.rows)).copyTo(out(cv::Rect(0,0, xmax - xmin,concatenated.rows)));
    
    cv::Mat grayImage;
    out.convertTo(grayImage, CV_8U, 255.0);
    
    imwrite( "./jpg/" + filename + ".jpg", grayImage );
    
    std::vector<float> array;
    if (grayImage.isContinuous()) {
        array.assign((float*)grayImage.datastart, (float*)grayImage.dataend);
    } else {
        for (int i = 0; i < grayImage.rows; ++i) {
            array.insert(array.end(), (float*)grayImage.ptr<uchar>(i), (float*)grayImage.ptr<uchar>(i)+grayImage.cols);
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
