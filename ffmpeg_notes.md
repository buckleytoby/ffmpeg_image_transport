note: avcodec_alloc_context3() doesn't actually make a device/context, it just allocates the struct AVCodecContext


must set avctx->hw_frames_ctx before call open2
avcodec_open2 actually makes the device/context, if one doesn't exist

in nvenc.c if hw_frames_ctx is set then we can use GPU frames as input (we want this)

can call av_hwframe_ctx_alloc to make the AVHWFramesContext, input device_ctx: a reference to a AVHWDeviceContext
can call av_hwdevice_ctx_create to make the AVHWDeviceContext

nvenc.c:2107 must upload an input hw frame
this frame is sent from ff_nvenc_receive_packet, set from ctx->frame which is avctx->priv_data which is set in avcodec_open2 which is set from codec->privdata which is an input to avcodec_open2 (set in ffmpeg_encoder for example)
however, inside nvenc_setup_device some avctx->priv_data gets overwritten from some of the hw_frames_ctx data

frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;
ctx->data_pix_fmt = frames_ctx->sw_format;
cuda_device_hwctx = frames_ctx->device_ctx->hwctx;
ctx->cu_context = cuda_device_hwctx->cuda_ctx;
ctx->cu_stream = cuda_device_hwctx->stream;

NvencContext *ctx = avctx->priv_data;
AVFrame *frame = ctx->frame;
reg.resourceToRegister = frame->data[0];

ok... so I think I'd still use frame_ from ffmpeg_encoder, I just set data[0] to the cuda resource and still use avcodec_send_frame
yes. avcodec_send_frame calls something internally which copies the input frame_ to avci->buffer_frame, and then in ff_encode_get_frame av_frame_move_ref copies avci->buffer_frame into frame
resourceToRegister = void* handle to the resource that is being registered of type NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR (input resource type is a cuda device pointer surface)

a cuda surface is r/w by your kernel (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-memory
)
great example on how to allocate device memory, surfaces, and invoke kernel


sidenotes:
https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/nvenc-video-encoder-api-prog-guide/index.html

"If the client has used a CUDA device to initialize the encoder session and wishes to use input buffers NOT allocated through the NVIDIA Video Encoder Interface, the client is required to use buffers allocated using the cuMemAlloc family of APIs. NVIDIA Video Encoder Interface supports CUdeviceptr and CUarray input formats."


inside hwcontext.c there is av_hwframe_transfer_data which is called if frame_->hw_frames_ctx is non-NULL when you avcodec_send_frame. The dst is avci->buffer_frame

encode.c: ffcodec(avctx->codec)->cb.receive_packet(avctx, avpkt): calls ff_nvenc_receive_packet
inside nvenc.c: ff_nvenc_receive_packet encodes the packet by calling ff_encode_get_frame (copies from the avci->buffer_frame)

nvenc.c: nvenc_upload_frame checks if avctx->pix_fmt == AV_PIX_FMT_CUDA for hw-frame
calls nvenc_register_frame queries both data[0] and linesize[0] for CUDA


ffmpeg installation:
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include -I/usr/local/cuda/include -fPIC" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib -L/usr/local/cuda/lib64" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-gnutls \
  --enable-libass \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libvorbis \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree \
  --enable-debug=3 \
  --enable-pic \
  --enable-shared \
  --enable-cuda-nvcc \
  --disable-static \
  --enable-libnpp \
  --disable-stripping

PATH="$HOME/bin:$PATH" make && \
make install && \
hash -r

  or follow this: https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/ffmpeg-with-nvidia-gpu/index.html

export LD_LIBRARY_PATH=/home/nuc-haptics/ffmpeg_build/lib:$LD_LIBRARY_PATH
  
