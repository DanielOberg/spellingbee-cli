//
//  SoundRecorder.m
//  SpellingBee
//
//  Created by Daniel Oberg on 2017-06-24.
//  Copyright © 2017 Daniel Oberg. All rights reserved.
//

#include "SoundRecorder.hh"

#include <algorithm>
#include <cstddef>
#include <map>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <string>
#include <thread>
#include <iostream>

#include <OpenAL/al.h>
#include <OpenAL/alc.h>
#include <glob.h>

#include <boost/lockfree/spsc_queue.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

#include "aquila/aquila.h"
#include "aquila/source/FramesCollection.h"
#include "aquila/transform/Mfcc.h"
#include "aquila/source/WaveFile.h"


const int SRATE = 44100;
const int SSIZE = 44100*0.50; //1024;

const int SENSITIVITY = 1;


boost::lockfree::spsc_queue<cv::Mat> images{100};
boost::lockfree::spsc_queue<std::vector<int16_t>> raw{100};


bool lastImage(cv::Mat &result) {
    return images.pop(result);
}

bool lastRaw(std::vector<int16_t> &result) {
    return raw.pop(result);
}

bool isSilence(cv::Mat &img) {
    return cv::mean(img)(0) < 1.5;
}

bool soundToImage(Aquila::SignalSource buffer, cv::Mat &resultingImage) {
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
            row.col(i) = firstMfccs.at(i) / SENSITIVITY;
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
    
    if (ymax+1 >= result.rows)
        return false;
    
    cv::Mat1f correctSize;
    cv::resize(out(cv::Rect(0, 0, MFCCS, 44)), correctSize, cv::Size(192, 192));
    
    cv::Mat grayImage;
    correctSize.convertTo(grayImage, CV_8U, 255.0);
    
    resultingImage = grayImage;
    return true;
}

int recordSound(bool *shouldStop) {
    alGetError();
    ALCdevice *device = alcCaptureOpenDevice(nullptr, SRATE, AL_FORMAT_MONO16, SSIZE);
    if (alGetError() != AL_NO_ERROR || !device) {
        return 0;
    }
    alcCaptureStart(device);
    if (alGetError() != AL_NO_ERROR || !device) {
        return 0;
    }
    std::vector<int16_t> big_buffer;

    
    while (true) {
        ALint samples_available;
        alcGetIntegerv(device, ALC_CAPTURE_SAMPLES, (ALCsizei)sizeof(ALint), &samples_available);
        if (samples_available > 0)
        {
        
            std::vector<int16_t> buffer_int16;

            buffer_int16.resize(samples_available);
            
            alcCaptureSamples(device, &buffer_int16[0], samples_available);
            
            big_buffer.insert(big_buffer.end(), buffer_int16.begin(), buffer_int16.end());
            
            if (big_buffer.size() >= SSIZE) {
                std::vector<double> buffer_double;

                buffer_double.resize(big_buffer.size());
                for (int i=0; i < big_buffer.size(); i++)
                    buffer_double[i] = ((double)big_buffer[i]) / INT16_MAX;
                
                Aquila::SignalSource signalSource(buffer_double, SRATE);
                cv::Mat img;
                auto isOk = soundToImage(signalSource, img);

                if (isOk) {
                    if (!isSilence(img)) {
                        images.push(img);
                        raw.push(big_buffer);
                    }
                    big_buffer.clear();
                }
            }
        }
        
        if (*shouldStop)
            break;
    }
    
    alcCaptureStop(device);
    alcCaptureCloseDevice(device);
    
    return 0;
}
