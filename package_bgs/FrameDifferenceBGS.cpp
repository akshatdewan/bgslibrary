/*
This file is part of BGSLibrary.

BGSLibrary is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BGSLibrary is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BGSLibrary.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "FrameDifferenceBGS.h"

FrameDifferenceBGS::FrameDifferenceBGS() : firstTime(true), enableThreshold(true), threshold(15), showOutput(true)
{
  std::cout << "FrameDifferenceBGS()" << std::endl;
  myfile.open("fg_intensity.txt");
}

FrameDifferenceBGS::~FrameDifferenceBGS()
{
  std::cout << "~FrameDifferenceBGS()" << std::endl;
  myfile.close();
}

void FrameDifferenceBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  

  if(img_input.empty())
    return;

  loadConfig();

  if(firstTime)
    saveConfig();

  if(img_input_prev.empty())
  {
    img_input.copyTo(img_input_prev);
    return;
  }

  cv::absdiff(img_input_prev, img_input, img_foreground);

  if(img_foreground.channels() == 3)
    cv::cvtColor(img_foreground, img_foreground, CV_BGR2GRAY);

  if(enableThreshold)
    cv::threshold(img_foreground, img_foreground, threshold, 255, cv::THRESH_BINARY);
  // TODO 
  float fg_intensity = 0.0;

  for(int j=0;j<img_foreground.rows;j++) 
    {
    for (int i=0;i<img_foreground.cols;i++)
      {
        fg_intensity += img_foreground.at<uchar>(i,j);
      }
    }
  std::ostringstream ss;
  int fontFace = 1;
  double fontScale = 1;
  int thickness = 2;  
  cv::Point textOrg(10, 130);
  ss << fg_intensity;
  std::string text (ss.str());
  std::string text_intrusion = " Possible Intrusion attempt";
  // It is a very ugly way of doing things because I am creating a socket for every frame
  try
    {
      // Create the client socket
      ClientSocket client_socket ( "localhost", 25001 ); //Server is listening on 25001
      try
      {
        client_socket << text_intrusion;
      }
      catch( SocketException& ) {}
    }
    
  catch ( SocketException& e )
    {
      std::cout << "Exception was caught:" << e.description() << "\n";
    }
  
  myfile << ss.str() << std::endl;
  if(fg_intensity - 1000000 > 0)
    {
    ss << text.append(text_intrusion); 
    }
  cv::putText(img_foreground, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
  
  // TODO
  if(showOutput)
    cv::imshow("Frame Difference", img_foreground);

  img_foreground.copyTo(img_output);

  img_input.copyTo(img_input_prev);

  firstTime = false;
}

void FrameDifferenceBGS::saveConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/FrameDifferenceBGS.xml", 0, CV_STORAGE_WRITE);

  cvWriteInt(fs, "enableThreshold", enableThreshold);
  cvWriteInt(fs, "threshold", threshold);
  cvWriteInt(fs, "showOutput", showOutput);

  cvReleaseFileStorage(&fs);
}

void FrameDifferenceBGS::loadConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/FrameDifferenceBGS.xml", 0, CV_STORAGE_READ);
  
  enableThreshold = cvReadIntByName(fs, 0, "enableThreshold", true);
  threshold = cvReadIntByName(fs, 0, "threshold", 15);
  showOutput = cvReadIntByName(fs, 0, "showOutput", true);

  cvReleaseFileStorage(&fs);
}
