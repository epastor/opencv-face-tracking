#include <iostream>
#include "opencv/cv.h"
#include "opencv/highgui.h"
using namespace std;
using namespace cv;

/** Function Headers */
void findAndRender( Mat frame );

String faceCascadeFile = "haarcascade_frontalface_default.xml";
CascadeClassifier faceClassifier;

String eyesCascadeFile = "haarcascade_eye.xml";
CascadeClassifier eyesClassifier;
 
int main()
{
    // Create the webcam window.
    cvNamedWindow( "CAMERA_STREAM", CV_WINDOW_AUTOSIZE );
    // Open the video stream using any connected cam.
 	CvCapture* stream = cvCaptureFromCAM( CV_CAP_ANY );

    if ( !stream )
	{
	    cout << "ERROR: The stream is null!\n";
	    return -1;
	}
 
    IplImage* frame = NULL;
    
    if( !faceClassifier.load( faceCascadeFile ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyesClassifier.load( eyesCascadeFile ) ){ printf("--(!)Error loading\n"); return -1; };
    
    char keypress;
    bool quit = false;
 
    while( !quit )
    {
        // Get a color frame from the cam.
        frame = cvQueryFrame( stream );
        // Find faces in the stream and render indicators.
        findAndRender( frame );
 		// Wait 20ms
        keypress = cvWaitKey(20);
 		// Turn on the exit flag if the user presses escape.
        if (keypress == 27) quit = true;
    }
 
    // Cleaning up.
    cvReleaseImage( &frame );
    cvDestroyAllWindows();
}

void findAndRender( Mat frame )
{
    std::vector<Rect> faces;
    Mat bwframe;

    // Get a black & white version of the frame
    cvtColor( frame, bwframe, CV_BGR2GRAY );

    // Equalize the image histogram to improve contrast.
    equalizeHist( bwframe, bwframe );

    // Try to find faces in the frame, discard any matches with a size < 300 x 300
    faceClassifier.detectMultiScale( bwframe, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size( 300, 300 ) );

    for( size_t i = 0; i < faces.size(); i++ )
    {
        // Find the center of each face match, draw an ellipse around them.
        Point center( faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5 );
        ellipse( frame, center, Size( faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar( 255, 0, 0 ), 4, 8, 0 );

        Mat faceRegion = bwframe( faces[i] );
        std::vector<Rect> eyes;

        // Try to find eyes in each face region, discard matches with a size < 30 x 30
        eyesClassifier.detectMultiScale( faceRegion, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(80, 80) );

        for( size_t j = 0; j < eyes.size(); j++ )
        {
            // Find the borders of each eye, draw a box around them.
            Point point1 = Point(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y);
            Point point2 = Point(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y + eyes[j].height);
            rectangle( frame, point1, point2, Scalar( 0, 0, 255 ), 4, 8, 0  );
        }
    }
    // Render the processed frame on screen
    imshow( "CAMERA_STREAM", frame );
}