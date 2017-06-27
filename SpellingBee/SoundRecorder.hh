//
//  SoundRecorder.h
//  SpellingBee
//
//  Created by Daniel Oberg on 2017-06-24.
//  Copyright © 2017 Daniel Oberg. All rights reserved.
//

// To listen use /Applications/VLC.app/Contents/MacOS/VLC --demux=rawaud --rawaud-channels 1 --rawaud-samplerate 44100

#ifndef SoundRecorder_h
#define SoundRecorder_h

#include <vector>

#include <opencv2/core.hpp>
    
int recordSound(bool *shouldStop);
bool lastImage(cv::Mat &result);
bool lastRaw(std::vector<int16_t> &result);

#endif /* SoundRecorder_h */
