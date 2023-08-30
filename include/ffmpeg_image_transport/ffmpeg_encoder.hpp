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

#ifndef FFMPEG_IMAGE_TRANSPORT__FFMPEG_ENCODER_HPP_
#define FFMPEG_IMAGE_TRANSPORT__FFMPEG_ENCODER_HPP_

#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unordered_map>

#include "ffmpeg_image_transport/tdiff.hpp"
#include "ffmpeg_image_transport/types.hpp"
#include "ffmpeg_image_transport/cuda_encoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/log.h>
}

namespace ffmpeg_image_transport
{
class FFMPEGEncoder
{
public:
  using Lock = std::unique_lock<std::recursive_mutex>;
  using Callback = std::function<void(const FFMPEGPacketConstPtr & pkt)>;

  FFMPEGEncoder();
  ~FFMPEGEncoder();
  // ------- various encoding settings
  void setCodec(const std::string & n)
  {
    Lock lock(mutex_);
    codecName_ = n;
  }
  void setProfile(const std::string & p)
  {
    Lock lock(mutex_);
    profile_ = p;
  }
  void setPreset(const std::string & p)
  {
    Lock lock(mutex_);
    preset_ = p;
  }
  void setQMax(int q)
  {
    Lock lock(mutex_);
    qmax_ = q;
  }
  void setBitRate(int r)
  {
    Lock lock(mutex_);
    bitRate_ = r;
  }
  int getGOPSize() const
  {
    Lock lock(mutex_);
    return (GOPSize_);
  }
  void setGOPSize(int g)
  {
    Lock lock(mutex_);
    GOPSize_ = g;
  }
  void setFrameRate(int frames, int second)
  {
    Lock lock(mutex_);
    frameRate_.num = frames;
    frameRate_.den = second;
    timeBase_.num = second;
    timeBase_.den = frames;
  }
  void setMeasurePerformance(bool p)
  {
    Lock lock(mutex_);
    measurePerformance_ = p;
  }
  // ------- teardown and startup
  bool isInitialized() const
  {
    Lock lock(mutex_);
    return (codecContext_ != NULL);
  }
  bool initialize(int width, int height, Callback callback);
  bool setInputFormat(AVPixelFormat format);
  bool setPixFormat(AVPixelFormat format);
  bool setUseHWFrames(bool use_hw_frames);
  void setLogger(rclcpp::Logger logger) { logger_ = logger; }
  void setParameters(rclcpp::Node * node);
  void reset();
  // encode image
  void encodeImage(const cv::Mat & img, const Header & header, const rclcpp::Time & t0);
  void encodeImage(const Image & msg);
  // ------- performance statistics
  void printTimers(const std::string & prefix) const;
  void resetTimers();

private:
  using PTSMap = std::unordered_map<int64_t, rclcpp::Time>;

  bool openCodec(int width, int height);
  void closeCodec();
  int drainPacket(const Header & hdr, int width, int height);
  // --------- variables
  rclcpp::Logger logger_;
  mutable std::recursive_mutex mutex_;
  std::function<void(const FFMPEGPacketConstPtr & pkt)> callback_;
  // config
  std::string codecName_;  // e.g. "libx264"
  std::string preset_;     // e.g. "slow", "medium", "lossless"
  std::string profile_;    // e.g. "main", "high", "rext"
  int qmax_{0};            // max allowed quantization. The lower the better quality
  int GOPSize_{15};        // distance between two keyframes
  AVPixelFormat pixFormat_{AV_PIX_FMT_YUV420P}; // use AV_PIX_FMT_CUDA for hw frames
  AVPixelFormat inputFormat_{AV_PIX_FMT_BGR24};
  AVPixelFormat hwFormat_{AV_PIX_FMT_YUV420P}; // standard nvenc format
  bool use_hw_frames_{false};
  AVRational timeBase_{1, 100};
  AVRational frameRate_{100, 1};
  int64_t bitRate_{1000000};
  // libav state
  AVCodecContext * codecContext_{NULL};
  AVFrame * frame_{NULL};
  AVPacket * packet_;
  int64_t pts_{0};
  PTSMap ptsToStamp_;
  // performance analysis
  bool measurePerformance_{true};
  int64_t totalInBytes_{0};
  int64_t totalOutBytes_{0};
  unsigned int frameCnt_{0};
  TDiff tdiffUncompress_;
  TDiff tdiffEncode_;
  TDiff tdiffDebayer_;
  TDiff tdiffFrameCopy_;
  TDiff tdiffSendFrame_;
  TDiff tdiffReceivePacket_;
  TDiff tdiffCopyOut_;
  TDiff tdiffPublish_;
  TDiff tdiffTotal_;
  // cuda things
	struct cuda_vars *gpu_vars_ = NULL;
  uint8_t* pRGB; 
  uint8_t* pYUV_; 
  cudaSurfaceObject_t pYUVSurf_ = 0;
};
}  // namespace ffmpeg_image_transport
#endif  // FFMPEG_IMAGE_TRANSPORT__FFMPEG_ENCODER_HPP_
