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

#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

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

#include <fstream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>



const int width = 640;
const int height = 480;

AVCodecContext * codecContext_{NULL};
AVFrame * frame_{NULL};
AVPacket * packet_;
int64_t pts_{0};

struct cuda_vars *gpu_vars_ = NULL;
uint8_t* pRGB; 
uint8_t* pYUV_; 
cudaSurfaceObject_t pYUVSurf_ = 0;

// config
std::string codecName_{"hevc_nvenc"};  // e.g. "libx264"
std::string preset_{"slow"};     // e.g. "slow", "medium", "lossless"
std::string profile_{"main"};    // e.g. "main", "high", "rext"
int qmax_{0};            // max allowed quantization. The lower the better quality
int GOPSize_{15};        // distance between two keyframes
AVPixelFormat pixFormat_{AV_PIX_FMT_YUV420P}; // use AV_PIX_FMT_CUDA for hw frames
AVPixelFormat inputFormat_{AV_PIX_FMT_BGR24};
AVPixelFormat hwFormat_{AV_PIX_FMT_CUDA}; // standard nvenc format

AVRational timeBase_{1, 100};
AVRational frameRate_{100, 1};
int64_t bitRate_{1000000};

void closeCodec()
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

bool openCodec()
{
  codecContext_ = NULL;
  try {
    if (codecName_.empty()) {
      throw(std::runtime_error("no codec set!"));
    }
    // find codec
    AVCodec * codec = avcodec_find_encoder_by_name(codecName_.c_str());
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

    codecContext_->pix_fmt = AV_PIX_FMT_CUDA;

    // must initialize the hw_frames_ctx
    AVBufferRef *hw_frames_ref;
    AVBufferRef *hw_device_context;
    if (av_hwdevice_ctx_create(&hw_device_context, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0)){
      throw(std::runtime_error("cannot create hwdevice context!"));
    }
    hw_frames_ref = av_hwframe_ctx_alloc(hw_device_context); // hw_device_context remains in my ownership

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

    cudaMalloc(&pYUV_, (size_t) width * height * 1.5);


    // open the codec
    if (avcodec_open2(codecContext_, codec, NULL) < 0) {
      throw(std::runtime_error("cannot open codec!"));
    }
    frame_ = av_frame_alloc();
    if (!frame_) {
      throw(std::runtime_error("cannot alloc frame!"));
    }
    frame_->width = width;
    frame_->height = height;
    frame_->format = pixFormat_;
    frame_->hw_frames_ctx = codecContext_->hw_frames_ctx;
    
  } catch (const std::runtime_error & e) {
    if (codecContext_) {
      // release the cuda context
      av_buffer_unref(&codecContext_->hw_frames_ctx);
      cudaDestroySurfaceObject(pYUVSurf_);
      avcodec_close(codecContext_);
      codecContext_ = NULL;
    }
    if (frame_) {
      av_free(frame_);
      frame_ = 0;
    }
    return (false);
  }
  return (true);
}

void encodeImage()
{
  const AVPixelFormat targetFmt = codecContext_->pix_fmt;
  frame_->data[0] = (uint8_t*) pYUV_;
  frame_->pts = pts_++;  //

  char buf[100];
  int ret_i = avcodec_is_open(codecContext_); std::cout<< "\n\navcodec_is_open error: " << ret_i << "\n";
  ret_i = av_codec_is_encoder(codecContext_->codec); std::cout<< "\n\av_codec_is_encoder error: " << ret_i << "\n";
  if (frame_->extended_data != frame_->data){ std::cout<< "frame_->extended_data != frame_->data"; }
  auto avhwctx = (AVHWFramesContext*) frame_->hw_frames_ctx->data;
  std::cout<<"frame_ format: " << av_get_pix_fmt_name(avhwctx->format) << " width: " << avhwctx->width << " height: " << avhwctx->height <<std::endl;
  AVPixelFormat* fmts;
  if (!av_hwframe_transfer_get_formats(frame_->hw_frames_ctx, AV_HWFRAME_TRANSFER_DIRECTION_FROM, &fmts, 0)){
    for (AVPixelFormat* it = fmts; *it != AV_PIX_FMT_NONE; it++){
      std::cout<< "av_hwframe_transfer_get_formats: " << av_get_pix_fmt_name(*it)<<std::endl;
    }
  }

  AVFrame* frameb; AVFrame* framea;
  frameb = av_frame_alloc(); framea = av_frame_alloc();
  frameb->width = frame_->width; framea->width = frame_->width;
  frameb->height = frame_->height; framea->height = frame_->height;
  frameb->format = frame_->format; framea->format = frame_->format;
  ret_i = av_frame_copy(frameb, frame_); av_strerror(ret_i, buf, 100); std::cout<< "av_frame_copy error: " << buf<<std::endl;
  ret_i = av_frame_ref(frameb, frame_); av_strerror(ret_i, buf, 100); std::cout<< "av_frame_ref error: " << buf<<std::endl;
  ret_i = av_hwframe_transfer_data(frameb, frame_, 0); av_strerror(ret_i, buf, 100); std::cout<<  "av_hwframe_transfer_data error: " << buf<<std::endl;


  int ret = avcodec_send_frame(codecContext_, frame_);
  if (ret != 0){
    char buf[100];
    av_strerror(ret, buf, 100);
  }
}

// main function
int main(int argc, char ** argv)
{
  av_log_set_level(AV_LOG_DEBUG);
  openCodec();
  encodeImage();
  // cleanup
  closeCodec();
}