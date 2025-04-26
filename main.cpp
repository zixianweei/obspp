#include "logger.h"
#include "op_flip.h"
#include "tensor.h"

#include "homedir.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

int main()
{
#if defined(HAS_CUTE_LOGGER)
    CuteLogger::GetInstance().Init("cute_log.txt");
#endif
    CUTE_LOG_INFO("cute instance init.");

    std::string vpath = std::string(HOME_DIR) + "/data/lol.mp4";

    cv::VideoCapture cap(vpath);
    if (!cap.isOpened())
    {
        CUTE_LOG_ERROR("{}: capture is not opened", __func__);
        return 1;
    }

    cv::Mat frame;
    int idx = 0;
    while (true)
    {
        idx++;
        cap.read(frame);
        cv::resize(frame, frame, cv::Size(640, 360));

        // flip
        cv::Mat frame_f32;
        frame.convertTo(frame_f32, CV_32FC3);
        cute::Tensor input;
        std::vector<int> shape = {1, 3, 360, 640};
        input.fromBytes(frame_f32.data, shape, cute::Format::kFloat32);

        cute::Tensor output;
        output.fromBytes(nullptr, shape, cute::Format::kFloat32);

        cute::OpFlip op;
        op.Forward(input, output);

        cv::Mat frame_f32_res = cv::Mat::zeros(frame_f32.size(), CV_32FC3);
        output.toBytes((void**)&frame_f32_res.data, shape, cute::Format::kFloat32);

        cv::Mat frame_res;
        frame_f32_res.convertTo(frame_res, CV_8UC3);

        cv::imshow("frame", frame_res);
        int key = cv::waitKey(1);
        if (key == 27)
        {
            CUTE_LOG_INFO("{}: escape key pressed, exiting", __func__);
            break;
        }
    }

    return 0;
}
