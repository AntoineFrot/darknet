#ifdef OPENCV

#include "stdio.h"
#include "stdlib.h"
#include "opencv2/opencv.hpp"
#include "image.h"

using namespace cv;

extern "C" {

Mat image_to_mat(image im)
{
    assert(im.c == 3 || im.c == 1);
    int x,y,c;
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);
    unsigned char *data = (unsigned char *)malloc(im.w * im.h * im.c);
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = copy.data[c*im.h*im.w + y*im.w + x];
                data[y*im.w*im.c + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    Mat m(im.h, im.w, CV_MAKETYPE(CV_8U, im.c), data);
    free_image(copy);
    free(data);
    return m;
}

image mat_to_image(Mat m)
{
    int h = m.rows;
    int w = m.cols;
    int c = m.channels();
    image im = make_image(w, h, c);
    unsigned char *data = (uint8_t*)m.data;
    int step = m.step;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    rgbgr_image(im);
    return im;
}

void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CAP_PROP_FPS, w);
    return (void *) cap;
}

image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

image get_image_from_screen(Display* pDisplay, Window* pRoot, int x, int y, int width, int height)
{
    XImage* img = XGetImage(pDisplay, *pRoot, x, y, width, height, AllPlanes, ZPixmap);
    Mat cvImg = Mat(height, width, CV_8UC4, img->data);

    Mat cvImg3C;

    cvtColor(cvImg, cvImg3C, COLOR_BGRA2BGR);

    //imshow("img", cvImg3C);
    //waitKey(10000);
    //width += 10;
    //height += 10;
    //printf("%d - %d\n", width, height);
    //sleep(1);

    return mat_to_image(cvImg3C);
}

image load_image_cv(char *filename, int channels)
{
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    Mat m;
    m = imread(filename, flag);
    if(!m.data){
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        char buff[256];
        sprintf(buff, "echo %s >> bad.list", filename);
        system(buff);
        return make_image(10,10,3);
        //exit(0);
    }
    image im = mat_to_image(m);
    return im;
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}

void send_key(Display* pDisplay)
{
    unsigned int keycode;
    keycode = XKeysymToKeycode(pDisplay, XK_W);
    XTestFakeKeyEvent(pDisplay, keycode, True, 0);
    XTestFakeKeyEvent(pDisplay, keycode, False, 0);
}

void make_window(char *name, int w, int h, int fullscreen)
{
    namedWindow(name, WINDOW_NORMAL);
    if (fullscreen) {
        setWindowProperty(name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    } else {
        resizeWindow(name, w, h);
        if(strcmp(name, "Demo") == 0) moveWindow(name, 0, 0);
    }
}

}

#endif
