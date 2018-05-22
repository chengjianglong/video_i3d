#include "nfdd-c3d-dryrun.h"


int main(int argc, char** argv)
{
    //Step 1: create tmp_c3d.txt
    string video_fname = argv[1];
    
    video_detect_values output_res = detect_frame_drop(video_fname);

    // Write the results into a json file.
    ofstream json_file;
    json_file.open("demo.json");

    // write algorithom section.
    json_file << "{\"algorithom\": {"<< endl;
    json_file << "\t\"name\": \"kwc3d-framedrop\"," << endl;
    json_file << "\t\"version\": \"0.1.2\"," << endl;
    json_file << "\t\"description\": \"Frame drop detection based on C3D neural network. We propose a new approach for forensic analysis by exploiting the local spatio-temporal relationships within a portion of a video to robustly detect frame removals. A Convolutional 3D Neural Network (C3D) is adapted for frame drop detection. In order to further suppress the errors due by the network, we produce a refined video-level confidence score and demonstrate that it is superior to the raw output scores from the network. For more details, please see our workshop paper, ``A C3D-based Convolutional Neural Network for Frame Dropping Detection in a Single Video Shot with this link: http://www.chengjianglong.com/publications/C3D4FDD_CVPRW.pdf.\"," << endl;
    json_file << "\t\"metadata_usage\": [\"no-metadata\"]," << endl;
    json_file << "\t\"target_manipulations\": [\"Frame_drop\"], " << endl;
    json_file << "\t\"algorithm_type\": [\"integrity-indicator???\"]," << endl;
    json_file << "\t\"indicator_type\": [\"digital\"]," << endl;
    json_file << "\t\"media_type\": [\"video\"], " << endl;
    json_file << "\t\"file_type\": [\"video/avi\"], " << endl;
    json_file << "\t\"gpu_usage\": [\"required\"]," << endl;
    json_file << "\t\"ram_usage\": 1024," << endl;
    json_file << "\t\"expected_runtime\": 100000," << endl;
    json_file << "\t\"code_link\": \"TBA???\"" << endl;
    json_file << " }," << endl;


    // write detection section.
    json_file <<"\"detection\": {" << endl;
    json_file <<"\t\"input_filename\": \"" << video_fname << "\" ," << endl;
    json_file <<"\t\"indicator_score\": " << output_res.detection_score << "," << endl;
    json_file <<"\t\"confidence\": " << output_res.detection_score << "," << endl;
    json_file <<"\t\"output\": \"Processed\", " << endl;
    json_file <<"\t\"specificity\": [ \"global\" ]," << endl;
    json_file <<"\t\"target_manipulations\": [\"frame-drop\"], " << endl;
    json_file <<"\t\"explanation\": \"consistency mismatchs mismatch occured based on the video-level confidence score\", " << endl;
    // write video-localization section.
    json_file<<"\t\"video_localization\": {" << endl;
    json_file<<"\t\t\"frame_confidence\": [";
    for(int i=0; i<output_res.frame_scores.size(); i++)
    {
        if(i == output_res.frame_scores.size()-1)
        {
            json_file << output_res.frame_scores[i] << "]," << endl;
        }
        else
        {
            json_file << output_res.frame_scores[i] << ",";
        }
    }
    json_file<<"\t\t\"frame_output\": [";
    for(int i=0; i<output_res.frame_optout.size(); i++)
    {
        if(i == output_res.frame_optout.size()-1)
        {
            json_file << output_res.frame_optout[i] << "]" << endl;
        }
        else
        {
            json_file << output_res.frame_optout[i] << ",";
        }
    }

    json_file<<" \t}"<<endl;
    json_file << " }," << endl;

    // write estimated_properties
    json_file << "\"supplemental_information\": {" << endl;
    json_file << "\t\"name\": \"peaks\", " << endl;
    json_file << "\t\"description\": \"top 2% peaks determined by the output confidence scores among the whole video\"," << endl;
    json_file << "\t\"value\": {" << endl;
    json_file << "\t\t\"peak_threshold\": " << output_res.peak_thresh << "," << endl;
    json_file << "\t\t\"peak_ids\": [";
    for(int i=0; i<output_res.peak_ids.size(); i++)
    {
        if(i == output_res.peak_ids.size()-1)
        {
            json_file << output_res.peak_ids[i] << "]," << endl;
        }
        else
        {
            json_file << output_res.peak_ids[i] << ",";
        }
    }
    json_file << "\t\t\"peak_scores\": [";
    for(int i=0; i<output_res.peak_scores.size(); i++)
    {
        if(i == output_res.peak_scores.size()-1)
        {
            json_file << output_res.peak_scores[i] << "]" << endl;
        }
        else
        {
            json_file << output_res.peak_scores[i] << ",";
        }
    }
    json_file << "\t}" << endl;
    json_file << " }" << endl;
    json_file<<"}"<<endl;

    json_file.close();


    return 0;

}
