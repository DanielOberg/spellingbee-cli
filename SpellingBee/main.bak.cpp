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

#include <boost/filesystem.hpp>
#include <boost/locale.hpp>
#include <boost/filesystem/fstream.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <SFML/Audio.hpp>

#include "aquila/global.h"
#include "aquila/source/FramesCollection.h"
#include "aquila/transform/Mfcc.h"
#include "aquila/tools/TextPlot.h"
#include "aquila/source/window/BarlettWindow.h"
#include "aquila/source/WaveFile.h"
#include "aquila/transform.h"
#include "aquila/functions.h"

#include "japanese.h"


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

std::vector<float> featuresFromAudio(Aquila::WaveFile buffer, int maxFrames, std::string debugName) {
    uint16_t FRAME_SIZE = 1024; // (uint16_t)((44100/1000.0)*20.0); // 44100 / 100; // 44100 samples per second
    
    Aquila::FramesCollection frames(buffer, FRAME_SIZE);
    
    Aquila::Spectrogram::Spectrogram spectrogram(frames);
    cv::Mat1f cvMat(spectrogram.getSpectrumSize()/2, spectrogram.getFrameCount());
    
    for (std::size_t x = 0; x < spectrogram.getFrameCount(); ++x)
    {
        for (std::size_t y = 0; y < spectrogram.getSpectrumSize()/2; ++y)
        {
            const Aquila::ComplexType point = spectrogram.getPoint(x, y);
            cvMat.at<float>(y, x) = (float)Aquila::dB(point);
        }
    }
    cv::Mat1f correctSize;
    cv::resize(cvMat, correctSize, cv::Size(100, spectrogram.getSpectrumSize()/2));
    
    cv::Mat simplify;
    correctSize.convertTo(simplify, CV_8UC1, 1.5);
    cv::Mat withColor;
    cv::applyColorMap(simplify, withColor, COLORMAP_JET);

    
    imwrite( std::string("./") + debugName + std::string(".jpg"), withColor );
    
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

std::pair<std::vector<std::vector<float> >, std::vector<int> > featuresFromAllAudio(const std::string folderName) {
    std::vector<std::vector<float> > features;
    std::vector<int> labels;
    
    int i = 0;
    for (const auto &w : romajiToHiragana) {
        for (int j=0; j<SIZE_T_MAX; j++) {
            std::string filename = std::string("./") + folderName + "/" + w.first + std::to_string(j) + ".wav";
            if (boost::filesystem::exists(filename)) {
                //                sf::SoundBuffer buffer;
                //                buffer.loadFromFile(filename);
                
                Aquila::WaveFile buffer(filename);
                const auto featureRow = featuresFromAudio(buffer, 100, w.first + std::to_string(j));
                if (featureRow.size() > 0) {
                    features.push_back(featureRow);
                    labels.push_back(i);
                }
            } else {
                break;
            }
        }
        i++;
    }
    return std::make_pair(features, labels);
}

cv::Mat matFromVectors(const std::vector<std::vector<float> > &features, size_t nrOfFeaturesToUse = 0) {
    const size_t nrOfSampleRows = features.size();
    
    if (nrOfFeaturesToUse == 0) {
        nrOfFeaturesToUse = 100000;
        for (const auto &fv : features) {
            nrOfFeaturesToUse = MIN(nrOfFeaturesToUse, fv.size());
        }
    }
    
    // Data for visual representation
    Mat sampleData = Mat::zeros((int)features.size(), (int)nrOfFeaturesToUse, CV_32FC1);
    for (int r = 0; r < MIN(sampleData.rows, features.size()); r++) {
        for (int c = 0; c < MIN(sampleData.cols, features.at(r).size()); c++) {
            sampleData.at<float>(r, c) = features.at(r).at(c);
        }
    }
    
    return sampleData;
}

cv::Mat matFromVector(const std::vector<int> &labels) {
    // Set up training data
    Mat labelData = Mat::zeros((int)labels.size(), 1, CV_32S);
    for (int r = 0; r < labels.size(); r++) {
        labelData.at<int>(r) = labels.at(r);
    }
    return labelData;
}

//void saveCSVFileWith(const std::string &filename,const cv::Mat &sampleData, const cv::Mat &labelData) {
//    Mat labelDataFloat(labelData);
//    labelDataFloat.convertTo(labelDataFloat, CV_32F);
//
//    Mat allData;
//    cv::hconcat(sampleData, labelDataFloat, allData);
//
//    boost::filesystem::path p{filename};
//    boost::filesystem::ofstream ofs{p};
//
//    ofs << cv::format(allData, cv::Formatter::FMT_CSV);
//
//}

void saveCSVFileWith(const std::string &filename,const cv::Mat &sampleData, const cv::Mat &labelData) {
    boost::filesystem::path p{filename};
    boost::filesystem::ofstream ofs{p};
    
    std::vector<std::pair<std::string, std::string>> romajiList(romajiToHiragana.begin(), romajiToHiragana.end());
    
    for (int r = 0; r < sampleData.rows; r++) {
        for (int c = 0; c < sampleData.cols; c++) {
            ofs << sampleData.at<float>(r, c) << ",";
        }
        
        ofs << "\"" << romajiList.at(labelData.at<int>(r)).first << "\"" << std::endl;
    }
    
}

Ptr<StatModel> mlTrainedFrom(const std::vector<std::vector<float> > &features, const std::vector<int> &labels) {
    // PRECONDITIONS
    {
        assert(features.size() == labels.size() && "Feature rows and label rows have to match");
        
        assert(features.size() > 0 && "No _rows_ of features exists");
        for (const auto &fv : features) {
            assert(fv.size() > 0 && "All rows should contain _features_");
        }
        
        
        int maxLabel = 0;
        for (const auto &l : labels) {
            maxLabel = MAX(maxLabel, l);
        }
        
        //assert(maxLabel == (romajiToHiragana.size()-1) && "All Hiragana labels are not represented");
        
//        std::multiset<int> uniqueLabelSet(labels.begin(), labels.end());
//        for (int i = 0; i < maxLabel; i++) {
//            assert(uniqueLabelSet.count(i) > 0 && "All labels should have at least one row with features");
//        }
    }
    
    Mat sampleData = matFromVectors(features);
    
    const size_t nrOfSampleRows = sampleData.rows;
    const size_t nrOfFeaturesToUse = sampleData.cols;
    
    // Set up training data
    Mat labelData = matFromVector(labels);
    
    saveCSVFileWith("trainingSet0.csv", sampleData, labelData);
    
    
    auto ml = SVM::create();
    ml->setType(SVM::C_SVC);
    ml->setKernel(SVM::RBF);
//    ml->setGamma(1000.033750);
//    ml->setC(1.20);
    ml->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6));
    
    auto trainingData = TrainData::create(sampleData, ROW_SAMPLE, labelData);
    
    trainingData->setTrainTestSplitRatio(0.8);
    
    ml->trainAuto(trainingData);
    
    //ml->train(trainingData);
    
    // POSTCONDITIONS
    {
        assert((nrOfFeaturesToUse > 0) && "Features trained on and variable count should match");
        assert((nrOfFeaturesToUse == ml->getVarCount()) && "Features trained on and variable count should match");
    }
    
    return ml;
}

int main(int argc, char *argv[])
{
    using namespace boost::locale::boundary;
    boost::locale::generator gen;
    std::locale loc = gen("ja_JP.UTF-8");
    
    const unsigned int USER_TEST_SIZE = 10;
    std::srand(0); //unsigned ( std::time(0) ) );
    
    const auto hiraganaToRomaji = flip_map(romajiToHiragana);
    
    if (sf::SoundBufferRecorder::isAvailable())
    {
        //        recordAllHiraganaSounds(std::string("TrainingSet0"));
        //                return 0;
        const size_t startIndex = ((rand() % japaneseWords.size()) / USER_TEST_SIZE) * USER_TEST_SIZE;
        const size_t endIndex = startIndex + USER_TEST_SIZE;
        
        //                cout << endl << endl << "Memorisation phase" << endl;
        //                for (size_t i = startIndex; i < endIndex; i++) {
        //                    const auto word = japaneseWords.at(i);
        //
        //                    cout << endl << endl << word.explanation << " : " << word.hiragna << " : " << word.romaji << endl << flush;
        //
        //                    ssegment_index map(boundary_type::character, word.hiragna.begin(), word.hiragna.end(), loc);
        //                    for(ssegment_index::iterator it=map.begin(), e=map.end(); it != e; ++it) {
        //                        const auto hiraganaCharacter = *it;
        //                        const auto romanjiCharacter = hiraganaToRomaji.at(hiraganaCharacter);
        //
        //                        std::cout << hiraganaCharacter << " " << romanjiCharacter << endl << flush;
        //
        //                        const auto sound = recordToBuffer();
        //
        //                        saveAudio(romanjiCharacter, sound);
        //                    }
        //                }
        
        
        
        const auto romajiFromClass = romajiFromClassIndex();
        const auto featuresAndLabels = featuresFromAllAudio("TrainingSet0");
        //        std::ofstream featureFile("features.bin", std::ios::out | std::ofstream::binary);
        //        std::copy(featuresAndLabels.first.begin(), featuresAndLabels.first.end(), std::ostreambuf_iterator<char>(featureFile));
        //        std::ofstream labelFile("labels.bin", std::ios::out | std::ofstream::binary);
        //        std::copy(featuresAndLabels.second.begin(), featuresAndLabels.second.end(), std::ostreambuf_iterator<char>(labelFile));
        //
        //        std::vector<std::vector<float> > features;
        //        {
        //            std::ifstream featureFile("features.bin", std::ios::in | std::ifstream::binary);
        //            std::istreambuf_iterator<char> iter(featureFile);
        //            std::copy(iter.begin(), iter.end(), std::back_inserter(features));
        //        }
        
        const auto svm = mlTrainedFrom(featuresAndLabels.first, featuresAndLabels.second);
        svm->save("svm_model.xml");
        //        auto svm = SVM::load<SVM>("svm_model.xml");
        
        
        cout << endl << endl << "Stats phase" << endl;
        {
            
            std::multiset<int> predictedSet;
            std::multiset<int> realSet;
            
            const auto featuresAndLabelsTestSet = featuresFromAllAudio("TestSet0");
            
            {
                Mat sampleData = matFromVectors(featuresAndLabelsTestSet.first, svm->getVarCount());
                // Set up training data
                Mat labelData = matFromVector(featuresAndLabelsTestSet.second);
                
                saveCSVFileWith("testSet0.csv", sampleData, labelData);
            }
            
            int   successful = 0;
            int unsuccessful = 0;
            for (int i = 0; i < featuresAndLabelsTestSet.first.size(); i++) {
                const auto featureVec = featuresAndLabelsTestSet.first.at(i);
                Mat featureMat = Mat::zeros(1, svm->getVarCount(), CV_32FC1);
                
                const int minCols = svm->getVarCount();
                for(int j = 0; j < minCols; j++)
                    featureMat.at<float>(j)=featureVec.at(j);
                
                //                std::cout << "feature:" << featureMat << std::endl;
                
                const auto prediction = svm->predict(featureMat);
                const auto realClass = featuresAndLabelsTestSet.second.at(i);
                
                predictedSet.insert((int)prediction);
                realSet.insert((int)realClass);
                
                bool proper  = (romajiFromClass.at(prediction) == romajiFromClass.at(realClass)) || (realClass == prediction);
                
                successful += proper ? 1 : 0;
                unsuccessful += proper ? 0 : 1;
            }
            
            for (int i = 0; i < romajiToHiragana.size(); i++) {
                if (realSet.count(i) != 0 || predictedSet.count(i) != 0)
                    std::cout << i << ": predicted:" << predictedSet.count(i) << ", real: " << realSet.count(i) << std::endl;
            }
            
            float successRate = (float)successful / ((float)successful+unsuccessful);
            
            cout << endl << "Successrate: " << successRate << endl;
        }
        
        
        
        
        
        
//        cout << endl << endl << "Test phase" << endl;
//        for (size_t i = startIndex; i < endIndex; i++) {
//            const auto word = japaneseWords.at(i);
//            
//            std::system(("say \"Your word is "s + word.explanation + ".\" "s).c_str());
//            
//            cout << endl << endl << "Your word is " << word.explanation << endl << flush;
//            
//            boost::locale::generator gen;
//            ssegment_index map(boundary_type::character, word.hiragna.begin(),word.hiragna.end(), gen("ja_JP.UTF-8"));
//            for(ssegment_index::iterator it=map.begin(), e=map.end(); it != e; ++it) {
//                const auto hiraganaCharacter = *it;
//                const auto romanjiCharacter = hiraganaToRomaji.at(hiraganaCharacter);
//                
//                std::cout << hiraganaCharacter << " " << romanjiCharacter << endl << flush;
//                
//                const auto sound = recordToBuffer();
//                const auto temporaryFileName = "./temp_sound.wav";
//                sound.saveToFile(temporaryFileName);
//                Aquila::WaveFile buffer(temporaryFileName);
//                const auto features_vec = featuresFromAudio(buffer, 100);
//                
//                const auto features = Mat(features_vec).t();
//                
//                std::cout << features << std::endl;
//                const auto prediction = svm->predict(features);
//                
//                std::cout << "Prediction: " << prediction << " " << romajiFromClass.at((size_t)prediction) << std::endl;
//                std::cout << "Correct: " << std::boolalpha << (romajiFromClass.at((size_t)prediction) == romanjiCharacter) << std::endl;
//            }
//        }
    }
    
    return 0;
}
