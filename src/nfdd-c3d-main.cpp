#include "opencv2/opencv.hpp"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    //Step 1: create tmp_c3d.txt
    string video_fname = argv[1];
    VideoCapture cap(video_fname.c_str());
    if(!cap.isOpened())
    {
        return -1;
    }

    int nframes = 0;
    while(true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        else
        {
            nframes++;
        }
    }

    //cout <<"nframes = " << nframes << endl;
    if(nframes < 16)
    {
        cout << "The length of frame squences is less than 16, and hence cannot fit in C3D." << endl;
        return -1;
    }

    ofstream c3d_data_file;
    c3d_data_file.open("tmp_c3d.txt");
    for(int i=0; i<nframes-15; i++)
    {
        c3d_data_file << video_fname << " " << i << " " << 0 << endl;
    }
    c3d_data_file.close();

    //Step 2: create tmp_run.sh.    
    ofstream bash_file;
    bash_file.open("tmp_run.sh");
    bash_file << "../build/nfdd-c3d \\" << endl;
    bash_file << "  tmp_c3d.prototxt \\" << endl;
    bash_file << "  iphone4_iter_206000.caffemodel \\" << endl;
    bash_file << "  0 \\" << endl;
    bash_file << "  1 \\" << endl;
    bash_file << "  " << (nframes-15)  << " \\" << endl;
    bash_file << "  tmp_c3d.output \\" << endl;
    bash_file << "  fc8" << endl;
    bash_file.close();

    
   // Step 3: execute the bash tmp_run.sh
    system("bash ./tmp_run.sh");
    system("rm tmp_run.sh");
    system("rm tmp_c3d.txt");

    // Step 4: analyze the output of the C3D netowrk and output the final confidence score.
    ifstream c3d_output_file;
    c3d_output_file.open("tmp_c3d.output");
    int id, label;
    float nonscore, score;
    vector<float> scoreList;
    while(c3d_output_file >> id >> nonscore >> score >> label)
    {
        //cout << id << " " << nonscore << " " << score << " " << label << endl;
        scoreList.push_back(score);
    }
    c3d_output_file.close();
    system("rm tmp_c3d.output");

    
    vector<float> peak_scoreList;
    for (int i=1; i<scoreList.size()-1; i++)
    {
        if(scoreList[i]>scoreList[i-1] && scoreList[i]>scoreList[i+1])
        {
            peak_scoreList.push_back(scoreList[i]);
        }
    }
    
    sort(peak_scoreList.begin(), peak_scoreList.end());
    
    float threshVal = peak_scoreList[int(0.98*peak_scoreList.size())];
//    cout << "threshVal = " << threshVal << endl;

    vector<int> bHpeaks(scoreList.size(), 0);

    for (int i=1; i<scoreList.size()-1; i++)
    {
        if(scoreList[i]>scoreList[i-1] && scoreList[i]>scoreList[i+1] && scoreList[i]>=threshVal)
        {
            bHpeaks[i] = 1;
        }
    }
    
//    for (int i=0; i<scoreList.size(); i++)
//    {
//        cout << "i = " << i << " ,  "  << scoreList[i] << " --- " << sorted_scoreList[i] << "  .....  " << bHpeaks[i]<< endl;
//    }

    double maxscore = -10.0;
    int wind = 250;
    for (int i=0; i<bHpeaks.size(); i++)
    {
        if (bHpeaks[i] == 0)
        {
            continue;
        }

        int w1 = (i-wind < 0)? 0:(i-wind);
        int w2 = (i+wind >= bHpeaks.size())? (bHpeaks.size()-1):(i+wind);

        vector<float> windScores;
        float maxPeakscore = scoreList[i];
        float npeaks = 1;
        for(int j=w1; j<=w2; j++)
        {
            windScores.push_back(scoreList[j]);
            if (bHpeaks[j] == 1 && j != i)
            {
                npeaks++;
                if(scoreList[j] > maxPeakscore)
                {
                    maxPeakscore = scoreList[j];
                }
            }
        }

        sort(windScores.begin(), windScores.end());
        float minVal = windScores[0];
        int midpos = windScores.size()/2;
        float medianVal = (windScores.size()%2 == 0)? ((windScores[midpos-1] + windScores[midpos])/2):windScores[midpos];

//        cout << "length: " << windScores.size() << " ---- ( " << windScores[midpos-1] << " , " << windScores[midpos] << " )." << endl; 

        float tmpscore = maxPeakscore/npeaks + (minVal - medianVal)*1.25;
//        cout << "minscore: " << minVal << endl;
//        cout << "medianscore: " << medianVal << endl;
//        cout << "tmpscore: " << maxPeakscore << " ... " << npeaks << " ... " << (minVal-medianVal) << " ---> " << tmpscore << endl;

        if (maxscore < tmpscore)
        {
            maxscore = tmpscore;
        }

    }


    cout << "Video confidence score: " << maxscore << endl;


    return 0;

}
