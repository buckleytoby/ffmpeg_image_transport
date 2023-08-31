// -*-c++-*---------------------------------------------------------------------------------------
// Copyright 2023 Bernd Pfrommer <bernd.pfrommer@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ffmpeg_image_transport/ffmpeg_encoder.hpp"

#include <cv_bridge/cv_bridge.h>

#include <fstream>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ffmpeg_image_transport/safe_param.hpp"

bool DEBUG = false;

namespace ffmpeg_image_transport
{
FFMPEGEncoder::FFMPEGEncoder() : logger_(rclcpp::get_logger("FFMPEGEncoder")) {
  if (DEBUG){
    rcutils_logging_set_logger_level(logger_.get_name(), RCUTILS_LOG_SEVERITY_DEBUG);
    av_log_set_level(AV_LOG_DEBUG);
  }
}

FFMPEGEncoder::~FFMPEGEncoder()
{
  Lock lock(mutex_);
  closeCodec();
  // close the hw context

  if (use_hw_frames_){
    if (hw_device_context){ av_buffer_unref(&hw_device_context); }
    cudaFree(pYUV_);
    bayer2rgb_free(gpu_vars_);
  }
}

void FFMPEGEncoder::reset()
{
  Lock lock(mutex_);
  closeCodec();
}

void FFMPEGEncoder::closeCodec()
{
  if (codecContext_) {
    avcodec_close(codecContext_);
    codecContext_ = NULL;
  }
  if (frame_) {
    av_free(frame_);
    frame_ = 0;
  }
  if (packet_) {
    av_packet_free(&packet_);  // also unreferences the packet
    packet_ = nullptr;
  }
}

void FFMPEGEncoder::setParameters(rclcpp::Node * node)
{
  Lock lock(mutex_);
  const std::string ns = "ffmpeg_image_transport.";
  codecName_ = get_safe_param<std::string>(node, ns + "encoding", "libx264");
  profile_ = get_safe_param<std::string>(node, ns + "profile", "main");
  preset_ = get_safe_param<std::string>(node, ns + "preset", "slow");
  qmax_ = get_safe_param<int>(node, ns + "qmax", 10);   // 0: highest, 63: worst quality bound
  bitRate_ = get_safe_param<int64_t>(node, ns + "bit_rate", 8242880);
  GOPSize_ = get_safe_param<int64_t>(node, ns + "gop_size", 15);
  RCLCPP_INFO_STREAM(
    logger_, "enc: " << codecName_ << " prof: " << profile_ << " preset: " << preset_);
  RCLCPP_INFO_STREAM(
    logger_, "qmax: " << qmax_ << " bitrate: " << bitRate_ << " gop: " << GOPSize_);
}

bool FFMPEGEncoder::initialize(int width, int height, Callback callback)
{
  Lock lock(mutex_);
  callback_ = callback;

  auto ret = openCodec(width, height);
  // auto device_ctx = (AVHWDeviceContext*) codecContext_->hw_device_ctx->data;
  // auto av_cuda_ctx = (AVCUDADeviceContext*) device_ctx->hwctx;
  // auto cuda_ctx = (CUcontext*) av_cuda_ctx->cuda_ctx;

  // cudaSetDevice(0);


  return ret;
}

bool FFMPEGEncoder::setInputFormat(AVPixelFormat format){
  inputFormat_ = format;
  RCLCPP_INFO_STREAM(
    logger_, "inputFormat_: " << inputFormat_);
}

bool FFMPEGEncoder::setPixFormat(AVPixelFormat format){
  pixFormat_ = format;
  RCLCPP_INFO_STREAM(
    logger_, "pixFormat_: " << pixFormat_);
}

bool FFMPEGEncoder::setUseHWFrames(bool use_hw_frames){
  use_hw_frames_ = use_hw_frames;
  RCLCPP_INFO_STREAM(
    logger_, "use_hw_frames_: " << use_hw_frames_);
}

bool FFMPEGEncoder::openCodec(int width, int height)
{
  codecContext_ = NULL;
  try {
    if (codecName_.empty()) {
      throw(std::runtime_error("no codec set!"));
    }
    if ((width % 32) != 0) {
      RCLCPP_WARN(logger_, "horiz res must be multiple of 32!");
    }
    if (codecName_ == "h264_nvmpi" && ((width % 64) != 0)) {
      RCLCPP_WARN(logger_, "horiz res must be multiple of 64!");
      throw(std::runtime_error("h264_nvmpi must have horiz rez mult of 64"));
    }
    // find codec
    const AVCodec * codec = avcodec_find_encoder_by_name(codecName_.c_str());
    if (!codec) {
      throw(std::runtime_error("cannot find codec: " + codecName_));
    }
    // allocate codec context
    codecContext_ = avcodec_alloc_context3(codec);
    if (!codecContext_) {
      throw(std::runtime_error("cannot allocate codec context!"));
    }
    codecContext_->bit_rate = bitRate_;
    codecContext_->qmax = qmax_;  // 0: highest, 63: worst quality bound
    codecContext_->width = width;
    codecContext_->height = height;
    codecContext_->time_base = timeBase_;
    codecContext_->framerate = frameRate_;

    // gop size is number of frames between keyframes
    // small gop -> higher bandwidth, lower cpu consumption
    codecContext_->gop_size = GOPSize_;
    // number of bidirectional frames (per group?).
    // NVenc can only handle zero!
    codecContext_->max_b_frames = 0;

    // encoded pixel format. Must be supported by encoder
    // check with e.g.: ffmpeg -h encoder=h264_nvenc -pix_fmts

    if (use_hw_frames_){
      codecContext_->pix_fmt = AV_PIX_FMT_CUDA;
    } else {
      codecContext_->pix_fmt = pixFormat_;
    }

    if (
      av_opt_set(codecContext_->priv_data, "profile", profile_.c_str(), AV_OPT_SEARCH_CHILDREN) !=
      0) {
      RCLCPP_ERROR_STREAM(logger_, "cannot set profile: " << profile_);
    }

    if (
      av_opt_set(codecContext_->priv_data, "preset", preset_.c_str(), AV_OPT_SEARCH_CHILDREN) !=
      0) {
      RCLCPP_ERROR_STREAM(logger_, "cannot set preset: " << preset_);
    }
    RCLCPP_DEBUG(
      logger_,
      "codec: %10s, profile: %10s, preset: %10s,"
      " bit_rate: %10ld qmax: %2d",
      codecName_.c_str(), profile_.c_str(), preset_.c_str(), bitRate_, qmax_);
    /* other optimization options for nvenc
         if (av_opt_set_int(codecContext_->priv_data, "surfaces",
         0, AV_OPT_SEARCH_CHILDREN) != 0) {
         RCLCPP_ERROR_STREAM(logger_, "cannot set surfaces!");
         }
      */

    // Check if using hw frames
    if (use_hw_frames_){
      // must initialize the hw_frames_ctx
      AVBufferRef *hw_frames_ref;
      if (av_hwdevice_ctx_create(&hw_device_context, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0)){
        throw(std::runtime_error("cannot create hwdevice context!"));
      }
      hw_frames_ref = av_hwframe_ctx_alloc(hw_device_context); // hw_device_context remains in my ownership (I'm responsible for freeing it)

      if (hw_frames_ref == NULL){
        throw(std::runtime_error("cannot allocate hwframe context!"));
      }

      // set up frame info
      auto frame_ctx = (AVHWFramesContext*) hw_frames_ref->data;
      frame_ctx->format = AV_PIX_FMT_CUDA;
      frame_ctx->sw_format = AV_PIX_FMT_YUV420P;
      frame_ctx->width = width;
      frame_ctx->height = height;

      // finalize hw_frames_ref for use
      av_hwframe_ctx_init(hw_frames_ref); 

      // set the hw_frames_ctx
      codecContext_->hw_frames_ctx = hw_frames_ref;

      //// setup cuda
      // init the bayer2rgb kernel
      RCLCPP_INFO_STREAM(logger_, "initializing cuda_debayer. Width: " << width << " Height: " << height);
      auto ret_cuda = bayer2rgb_init(&gpu_vars_, width, height, 3, false); // RGB24 has 3 bytes per pixel
      if (ret_cuda != cudaSuccess) {
        RCLCPP_ERROR_STREAM(logger_, "cannot init cuda_debayer: " << cudaGetErrorString(ret_cuda));
        throw(std::runtime_error("cannot init cuda_debayer"));
      }

      // Allocate CUDA arrays in device memory
      cudaMalloc(&pYUV_, (size_t) width * height * 1.5);
    }

    // open the codec
    if (avcodec_open2(codecContext_, codec, NULL) < 0) {
      throw(std::runtime_error("cannot open codec!"));
    }
    RCLCPP_DEBUG_STREAM(logger_, "opened codec: " << codecName_);
    frame_ = av_frame_alloc();
    if (!frame_) {
      throw(std::runtime_error("cannot alloc frame!"));
    }
    frame_->width = width;
    frame_->height = height;
    frame_->quality = 10; // default for qmax
    frame_->format = pixFormat_;
    if (use_hw_frames_){
      RCLCPP_INFO_STREAM(logger_, "set frame_->hw_frames_ctx");
      frame_->hw_frames_ctx = codecContext_->hw_frames_ctx;
      frame_->buf[0] = av_buffer_alloc(1); // must be set so that avcodec_send_frame will copy over the hw_frames_ctx
      frame_->linesize[0] = width; // nb of bytes per line of the YUV image
    }
    // allocate image for frame
    if (!use_hw_frames_ &&
      av_image_alloc(
        frame_->data, frame_->linesize, width, height, static_cast<AVPixelFormat>(frame_->format),
        64) < 0) {
      throw(std::runtime_error("cannot alloc image!"));
    } else {
      // print linesizes
      RCLCPP_DEBUG_STREAM(
        logger_, "linesizes: " << frame_->linesize[0] << " " << frame_->linesize[1] << " "
                               << frame_->linesize[2]);
    }
    // Initialize packet
    packet_ = av_packet_alloc();
    packet_->data = NULL;
    packet_->size = 0;
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR_STREAM(logger_, e.what());
    if (codecContext_) {
      // release the cuda context
      if (use_hw_frames_){
        if (hw_device_context){ av_buffer_unref(&hw_device_context); }
        cudaFree(pYUV_);
        bayer2rgb_free(gpu_vars_);
      }
      avcodec_close(codecContext_);
      codecContext_ = NULL;
    }
    if (frame_) {
      av_free(frame_);
      frame_ = 0;
    }
    return (false);
  }
  RCLCPP_DEBUG_STREAM(
    logger_, "intialized codec " << codecName_ << " for image: " << width << "x" << height);
  return (true);
}

void FFMPEGEncoder::encodeImage(const Image & msg)
{
  rclcpp::Time t0;
  if (measurePerformance_) {
    t0 = rclcpp::Clock().now();
  }
  cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
  encodeImage(img, msg.header, t0);
  if (measurePerformance_) {
    const auto t1 = rclcpp::Clock().now();
    tdiffDebayer_.update((t1 - t0).seconds());
  }
}

void strided_copy(
  uint8_t * dest, const int stride_dest, const uint8_t * src, const int stride_src, const int n,
  const int length)
{
  if ((stride_dest == stride_src) && (stride_src == length)) {
    memcpy(dest, src, n * length);
  } else {
    for (int ii = 0; ii < n; ii++) {
      memcpy(dest + stride_dest * ii, src + stride_src * ii, length);
    }
  }
}

void channel_copy(
  uint8_t * dest, const int nb_channels, const uint8_t * src, const int src_nb_channels, const int width, const int height) {
  // assumes 8 bits per channel

    for (int ii = 0; ii < height; ii++) {
      for (int jj = 0; jj < width ; jj++){
        memcpy(dest + nb_channels * (width * ii + jj), src + src_nb_channels * (width * ii + jj), src_nb_channels);
      }
    }
}

void FFMPEGEncoder::encodeImage(const cv::Mat & img, const Header & header, const rclcpp::Time & t0)
{
  Lock lock(mutex_);
  rclcpp::Time t1, t2, t3;
  if (measurePerformance_) {
    frameCnt_++;
    t1 = rclcpp::Clock().now();
    totalInBytes_ += img.cols * img.rows;  // raw size!
  }

  const int width = img.cols;
  const int height = img.rows;
  const AVPixelFormat targetFmt = codecContext_->pix_fmt;

  if (use_hw_frames_){
    //// hw transcode
    // call cuda transcode function
    if (inputFormat_ == AV_PIX_FMT_BAYER_RGGB8 && hwFormat_ == AV_PIX_FMT_YUV420P){

	    cudaStream_t stream = NULL;
      cudaError_t ret_cuda = cudaSuccess;

      //run bayer->rgb kernel function
      // RCLCPP_INFO(logger_, "running bayer2rgb kernel");
      ret_cuda = bayer2rgb_process(gpu_vars_, (void*)img.data, &pRGB, &stream, true);
      if (ret_cuda != cudaSuccess) {
        RCLCPP_ERROR_STREAM(logger_, "cannot hw convert bayerRG8 to RGB24: " << cudaGetErrorString(ret_cuda));
        return;
      }
      // RCLCPP_INFO(logger_, "running rgb2yuv420p kernel");
      ret_cuda = rgb2yuv420p_process(pRGB, pYUV_, height, width);
      if (ret_cuda != cudaSuccess) {
        RCLCPP_ERROR_STREAM(logger_, "cannot hw convert RGB24 to YUV420P: " << cudaGetErrorString(ret_cuda));
        return;
      }

      // construct hw frame
      // set data[0] to the YUV surface cuda device pointer
      // RCLCPP_INFO(logger_, "set frame");
      cudaDeviceSynchronize(); // might not have to call this, or there might be a quicker function
      frame_->data[0] = (uint8_t*) pYUV_;
      // frame_->data[1] = (uint8_t*) pYUV_;
      // frame_->data[2] = (uint8_t*) pYUV_;
      frame_->linesize[0] = width; // nb of bytes per line of the YUV image
      // frame_->linesize[1] = width/2; // nb of bytes per line of the YUV image
      // frame_->linesize[2] = width/2; // nb of bytes per line of the YUV image
    } else {
      RCLCPP_ERROR_STREAM(logger_, "cannot hw convert format bayerRG8 -> " << (int)hwFormat_);
      return;
    }

  } else {
    // sw transcode
    if (inputFormat_ == AV_PIX_FMT_BGR24 && targetFmt == AV_PIX_FMT_BGR24) {
      const uint8_t * p = img.data;
      strided_copy(frame_->data[0], frame_->linesize[0], p, width * 3, height, width * 3);
    } else if (inputFormat_ == AV_PIX_FMT_RGB24 && targetFmt == AV_PIX_FMT_RGB0) {
        // 32 bits, RGBXRGBX X is unused
        const uint8_t * p = img.data;
        channel_copy(frame_->data[0], 4, p, 3, width, height);
    }
    else if (inputFormat_ == AV_PIX_FMT_BGR24 && targetFmt == AV_PIX_FMT_YUV420P) {
      cv::Mat yuv;
      cv::cvtColor(img, yuv, cv::COLOR_BGR2YUV_I420);
      const uint8_t * p = yuv.data;
      // Y
      strided_copy(frame_->data[0], frame_->linesize[0], p, width, height, width);
      // U
      strided_copy(
        frame_->data[1], frame_->linesize[1], p + width * height, width / 2, height / 2, width / 2);
      // V
      strided_copy(
        frame_->data[2], frame_->linesize[2], p + width * height + width / 2 * height / 2, width / 2,
        height / 2, width / 2);
    } else if (inputFormat_ == AV_PIX_FMT_YUV444P && targetFmt == AV_PIX_FMT_YUV444P) {
      // https://www.flir.com/support-center/iis/machine-vision/knowledge-base/understanding-yuv-data-formats/
      // The YUV444 data format transmits 24 bits per pixel. Each pixel is assigned unique Y, U and V values—1 byte for each value,
      // just check to make sure this is the format coming off the camera.
      // (for us it should be)
      frame_->data[0] = img.data;
    }
      else {
      RCLCPP_ERROR_STREAM(logger_, "cannot convert format bgr8 -> " << (int)codecContext_->pix_fmt);
      return;
    }
  }
  if (measurePerformance_) {
    t2 = rclcpp::Clock().now();
    tdiffFrameCopy_.update((t2 - t1).seconds());
  }

  frame_->pts = pts_++;  //
  ptsToStamp_.insert(PTSMap::value_type(frame_->pts, header.stamp));

  // ///// DEBUG /////
  // char buf[100];
  // int ret_i = avcodec_is_open(codecContext_); RCLCPP_INFO_STREAM(logger_, "\n\navcodec_is_open error: " << ret_i);
  // ret_i = av_codec_is_encoder(codecContext_->codec); RCLCPP_INFO_STREAM(logger_, "av_codec_is_encoder error: " << ret_i);
  // if (frame_->extended_data != frame_->data){ RCLCPP_INFO_STREAM(logger_, "frame_->extended_data != frame_->data"); }
  // auto avhwctx = (AVHWFramesContext*) frame_->hw_frames_ctx->data;
  // RCLCPP_INFO_STREAM(logger_, "frame_ format: " << av_get_pix_fmt_name(avhwctx->format) << " width: " << avhwctx->width << " height: " << avhwctx->height);
  // AVPixelFormat* fmts;
  // if (!av_hwframe_transfer_get_formats(frame_->hw_frames_ctx, AV_HWFRAME_TRANSFER_DIRECTION_FROM, &fmts, 0)){
  //   for (AVPixelFormat* it = fmts; *it != AV_PIX_FMT_NONE; it++){
  //     RCLCPP_INFO_STREAM(logger_, "av_hwframe_transfer_get_formats: " << av_get_pix_fmt_name(*it));
  //   }
  // }

  // AVFrame* frameb; AVFrame* framea;
  // frameb = av_frame_alloc(); framea = av_frame_alloc();
  // frameb->width = frame_->width; framea->width = frame_->width;
  // frameb->height = frame_->height; framea->height = frame_->height;
  // frameb->format = frame_->format; framea->format = frame_->format;
  // ret_i = av_frame_copy(frameb, frame_); av_strerror(ret_i, buf, 100); RCLCPP_INFO_STREAM(logger_, "av_frame_copy error: " << buf);
  // ret_i = av_frame_ref(frameb, frame_); av_strerror(ret_i, buf, 100); RCLCPP_INFO_STREAM(logger_, "av_frame_ref error: " << buf);
  // ret_i = av_hwframe_transfer_data(frameb, frame_, 0); av_strerror(ret_i, buf, 100); RCLCPP_INFO_STREAM(logger_, "av_hwframe_transfer_data error: " << buf);

  // // int ret1 = avcodec_receive_packet(codecContext_, packet_); av_strerror(ret1, buf, 100);
  // // RCLCPP_INFO_STREAM(logger_, "avcodec_receive_packet: " << buf );
  // /////////////////////

  int ret = avcodec_send_frame(codecContext_, frame_);
  if (ret != 0){
    char buf[100];
    av_strerror(ret, buf, 100);
    RCLCPP_INFO_STREAM(logger_, "avcodec_send_frame error: " << buf);
  }

  if (measurePerformance_) {
    t3 = rclcpp::Clock().now();
    tdiffSendFrame_.update((t3 - t2).seconds());
  }
  // now drain all packets
  while (ret == 0) {
    ret = drainPacket(header, width, height);
  }
  if (measurePerformance_) {
    const rclcpp::Time t4 = rclcpp::Clock().now();
    tdiffTotal_.update((t4 - t0).seconds());
  }
}

int FFMPEGEncoder::drainPacket(const Header & header, int width, int height)
{
  rclcpp::Time t0, t1, t2;
  if (measurePerformance_) {
    t0 = rclcpp::Clock().now();
  }
  int ret = avcodec_receive_packet(codecContext_, packet_);
  if (measurePerformance_) {
    t1 = rclcpp::Clock().now();
    tdiffReceivePacket_.update((t1 - t0).seconds());
  }
  const AVPacket & pk = *packet_;
  if (ret == 0 && pk.size > 0) {
    FFMPEGPacket * packet = new FFMPEGPacket;
    FFMPEGPacketConstPtr pptr(packet);
    packet->data.resize(pk.size);
    packet->width = width;
    packet->height = height;
    packet->pts = pk.pts;
    packet->flags = pk.flags;
    memcpy(&(packet->data[0]), pk.data, pk.size);
    if (measurePerformance_) {
      t2 = rclcpp::Clock().now();
      totalOutBytes_ += pk.size;
      tdiffCopyOut_.update((t2 - t1).seconds());
    }
    packet->header = header;
    auto it = ptsToStamp_.find(pk.pts);
    if (it != ptsToStamp_.end()) {
      packet->header.stamp = it->second;
      packet->encoding = codecName_;
      callback_(pptr);  // deliver packet callback
      if (measurePerformance_) {
        const auto t3 = rclcpp::Clock().now();
        tdiffPublish_.update((t3 - t2).seconds());
      }
      ptsToStamp_.erase(it);
    } else {
      RCLCPP_ERROR_STREAM(logger_, "pts " << pk.pts << " has no time stamp!");
    }
    av_packet_unref(packet_);  // free packet allocated by encoder
  }
  return (ret);
}

void FFMPEGEncoder::printTimers(const std::string & prefix) const
{
  Lock lock(mutex_);
  RCLCPP_INFO_STREAM(
    logger_, prefix << " pktsz: " << totalOutBytes_ / frameCnt_ << " compr: "
                    << totalInBytes_ / (double)totalOutBytes_ << " debay: " << tdiffDebayer_
                    << " fmcp: " << tdiffFrameCopy_ << " send: " << tdiffSendFrame_
                    << " recv: " << tdiffReceivePacket_ << " cout: " << tdiffCopyOut_
                    << " publ: " << tdiffPublish_ << " tot: " << tdiffTotal_);
}
void FFMPEGEncoder::resetTimers()
{
  Lock lock(mutex_);
  tdiffDebayer_.reset();
  tdiffFrameCopy_.reset();
  tdiffSendFrame_.reset();
  tdiffReceivePacket_.reset();
  tdiffCopyOut_.reset();
  tdiffPublish_.reset();
  tdiffTotal_.reset();
  frameCnt_ = 0;
  totalOutBytes_ = 0;
  totalInBytes_ = 0;
}
}  // namespace ffmpeg_image_transport
