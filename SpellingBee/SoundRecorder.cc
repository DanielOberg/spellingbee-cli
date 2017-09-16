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
    uint16_t FRAME_SIZE = 128; // 44100 / 100; // 44100 samples per second
    uint16_t MFCCS = 12;
    
    Aquila::FramesCollection frames(buffer, FRAME_SIZE);
    Aquila::Mfcc mfcc(FRAME_SIZE);
    
    cv::Mat1f mfccMat(0, MFCCS);
    
    for (const Aquila::Frame& frame : frames) {
        auto mfccValues = mfcc.calculate(frame, MFCCS);
        
        auto firstMfccs = std::vector<double>(mfccValues.begin(), mfccValues.begin()+MFCCS);
        
        cv::Mat1f row = cv::Mat1f::zeros(1, MFCCS);
        
        for (int i = 0; i < MFCCS; i++) {
            row.col(i) = firstMfccs.at(i) / SENSITIVITY;
        }
        mfccMat.push_back(row);
    }
    mfccMat = mfccMat.t();
    
    cv::Mat1f normalized;
    cv::normalize(mfccMat, normalized, 0, 1, cv::NORM_MINMAX);
    
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
    cv::normalize(deltaMfccs, deltaMfccs, 0, 1, cv::NORM_MINMAX);
    
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
    cv::normalize(deltaDeltaMfccs, deltaDeltaMfccs, 0, 1, cv::NORM_MINMAX);
    
    cv::Mat concatenated;
    
    cv::vconcat(normalized, deltaMfccs, concatenated);
    cv::vconcat(concatenated, deltaDeltaMfccs, concatenated);
    
    const int MAX_WIDTH = 100;
    
    double min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(mfccMat, &min, &max, &min_loc, &max_loc);
    
    int xavg = ((max_loc.x - min_loc.x) / 2) + min_loc.x;
    int xmax = xavg + (MAX_WIDTH / 2);
    int xmin = xavg - (MAX_WIDTH / 2);
    
    if (xmax >= mfccMat.cols - 1) {
        xmax = mfccMat.cols - 1;
    }
    
    if (xmin < 0) {
        xmin = 0;
        xmax = MAX_WIDTH;
    }
    
    cv::Mat out = cv::Mat::zeros(concatenated.rows, MAX_WIDTH, concatenated.type());
    concatenated(cv::Rect(xmin,0, xmax - xmin,concatenated.rows)).copyTo(out(cv::Rect(0,0, xmax - xmin,concatenated.rows)));
    
    cv::Mat grayImage;
    out.convertTo(grayImage, CV_8U, 255.0);
    
//    imwrite( "./jpg/" + filename + ".jpg", grayImage );
    
    std::vector<float> array;
    if (grayImage.isContinuous()) {
        array.assign((float*)grayImage.datastart, (float*)grayImage.dataend);
    } else {
        for (int i = 0; i < grayImage.rows; ++i) {
            array.insert(array.end(), (float*)grayImage.ptr<uchar>(i), (float*)grayImage.ptr<uchar>(i)+grayImage.cols);
        }
    }
    
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
