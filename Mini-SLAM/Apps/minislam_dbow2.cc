/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

 #include <iostream>
 #include <vector>
 
 // DBoW2
 #include <DBoW2.h> // defines OrbVocabulary and OrbDatabase
 
 // OpenCV
 #include <opencv2/core.hpp>
 #include <opencv2/highgui.hpp>
 #include <opencv2/features2d.hpp>
 #include <experimental/filesystem> 
 
 
 using namespace DBoW2;
 using namespace std;
 namespace fs = std::experimental::filesystem;
 
 // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
 int loadFeatures(vector<vector<cv::Mat > > &features, const string &folderPath);
 void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
 void testDatabase(const vector<vector<cv::Mat > > &features, int nImages,  const string &folderPath);
 
 
 // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 
 void wait()
 {
   cout << endl << "Press enter to continue" << endl;
   getchar();
 }
 
 // ----------------------------------------------------------------------------
 
 int main()
 {
   vector<vector<cv::Mat > > features;
   string folderPath = "Datasets/V102_small";
   int nImages = loadFeatures(features, folderPath);
 
   wait();

   testDatabase(features, nImages, folderPath);
 
   return 0;
 }
 
 // ----------------------------------------------------------------------------

 int loadFeatures(vector<vector<cv::Mat>> &features, const string &folderPath) {
  features.clear();

  // Check if directory exists
  if (!fs::exists(folderPath)) {
      cerr << "Directory does not exist: " << folderPath << endl;
      return -1;
  }

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;

  vector<string> imagePaths;
  try {
      for (const auto &entry : fs::directory_iterator(folderPath)) {
          if (is_regular_file(entry.path())) {
              string ext = entry.path().extension().string();
              transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
              if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || 
                  ext == ".bmp" || ext == ".tiff" || ext == ".tif" || 
                  ext == ".ppm" || ext == ".pgm") {
                  imagePaths.push_back(entry.path().string());
              }
          }
      }
  } catch (const fs::filesystem_error &ex) {
      cerr << "Filesystem error: " << ex.what() << endl;
      return -1;
  }

  // Sort images to ensure consistent order
  sort(imagePaths.begin(), imagePaths.end());

  features.reserve(imagePaths.size());

  for (const auto &imagePath : imagePaths) {
    std::cout << "Loading image: " << imagePath << endl;
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
      if (image.empty()) {
          cerr << "Failed to load image: " << imagePath << endl;
          continue;
      }

      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;
      orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

      features.emplace_back();
      std::cout << "Desc size: " << descriptors.size() << std::endl;
      changeStructure(descriptors, features.back());
  }

  return imagePaths.size();
}
 
 // ----------------------------------------------------------------------------
 
 void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
 {
   out.resize(plain.rows);
 
   for(int i = 0; i < plain.rows; ++i)
   {
    // std::cout << "plain.row(i): " << plain.row(i).size() << std::endl;
     out[i] = plain.row(i);
   }
 }
 
 // ----------------------------------------------------------------------------
 
 void testDatabase(const vector<vector<cv::Mat > > &features, int nImages, const string &folderPath)
 {
   cout << "Creating a small database..." << endl;
 
   // load the vocabulary from disk
   std::cout << "Opening Vocab" << std::endl;
  //  OrbVocabulary voc("ORBvoc.txt"); // XAVI: use this
   OrbVocabulary voc("small_voc.yml.gz");
   
   std::cout << "Creating DB" << std::endl;
   OrbDatabase db(voc, false, 0); // false = do not use direct index
   // (so ignore the last param)
   // The direct index is useful if we want to retrieve the features that 
   // belong to some vocabulary node.
   // db creates a copy of the vocabulary, we may get rid of "voc" now
 
   // add images to the database
   std::cout << "Adding images" << std::endl;
   for(int i = 0; i < nImages; i++)
   {
     db.add(features[i]);
   }
 
   cout << "... done!" << endl;
 
   cout << "Database information: " << endl << db << endl;

   // Read the images
   vector<string> imagePaths;
    try {
        for (const auto &entry : fs::directory_iterator(folderPath)) {
            if (is_regular_file(entry.path())) {
                string ext = entry.path().extension().string();
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || 
                    ext == ".bmp" || ext == ".tiff" || ext == ".tif" || 
                    ext == ".ppm" || ext == ".pgm") {
                    imagePaths.push_back(entry.path().string());
                }
            }
        }
    } catch (const fs::filesystem_error &ex) {
        cerr << "Filesystem error: " << ex.what() << endl;
        return;
    }

  // Sort images to ensure consistent order
  sort(imagePaths.begin(), imagePaths.end());
 
   // and query the database
   cout << "Querying the database: " << endl;
 
  // Create display windows
  const string query_win = "Query Image";
  const string result_win = "Closest Match";
  cv::namedWindow(query_win, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(result_win, cv::WINDOW_AUTOSIZE);

  // Query loop
  for(int i = 0; i < nImages; i++) 
  {
    // Show query image
    cv::Mat query_img = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);
    if(query_img.empty()) {
      cerr << "Failed to load query image: " << imagePaths[i] << endl;
      continue;
    }
    cv::imshow(query_win, query_img);

    // Perform query
    QueryResults ret;
    db.query(features[i], ret, 4); // Get top 4 matches

    cout << "\nQuery Image " << i << " (" << imagePaths[i] << "):\n";

    // Show results (skip first match)
    for(size_t j = 0; j < ret.size(); ++j) {
      const Result& r = ret[j];
      
      // Show result image
      cv::Mat result_img = cv::imread(imagePaths[r.Id], cv::IMREAD_GRAYSCALE);
      if(result_img.empty()) {
        cerr << "Failed to load result image: " << imagePaths[r.Id] << endl;
        continue;
      }
      cv::imshow(result_win, result_img);

      cout << "- Match " << j << ": Image " << imagePaths[r.Id]
           << " (Score: " << r.Score << ")\n";

      // Wait for key press
      int key = cv::waitKey(0);
      if(key == 'q' || key == 27) { // Exit on 'q' or ESC
        cv::destroyAllWindows();
        return;
      }
    }

    // Clear results window
    cv::Mat blank = cv::Mat::zeros(200, 200, CV_8UC1);
    cv::imshow(result_win, blank);
  }

  cv::destroyAllWindows();
 
   cout << endl;
 
   // we can save the database. The created file includes the vocabulary
   // and the entries added
   cout << "Saving database..." << endl;
   db.save("testDB.yml.gz");
   cout << "... done!" << endl;
   
   // once saved, we can load it again  
   cout << "Retrieving database once again..." << endl;
   OrbDatabase db2("testDB.yml.gz");
   cout << "... done! This is: " << endl << db2 << endl;
 }
 
 // ----------------------------------------------------------------------------
 
 
 