kernel_forward = '''
    extern "C"
    __global__ void roi_forward(const float* const bottom_data,const float* const bottom_rois,
                float* top_data, int* argmax_data,
                const double spatial_scale,const int channels,const int height, 
                const int width, const int pooled_height, 
                const int pooled_width,const int NN
    ){
        
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=NN)
        return;
    const int pw = idx % pooled_width;
    const int ph = (idx / pooled_width) % pooled_height;
    const int c = (idx / pooled_width / pooled_height) % channels;
    int num = idx / pooled_width / pooled_height / channels;
    const int roi_batch_ind = bottom_rois[num * 5 + 0];
    const int roi_start_w = round(bottom_rois[num * 5 + 1] * spatial_scale);
    const int roi_start_h = round(bottom_rois[num * 5 + 2] * spatial_scale);
    const int roi_end_w = round(bottom_rois[num * 5 + 3] * spatial_scale);
    const int roi_end_h = round(bottom_rois[num * 5 + 4] * spatial_scale);
    // Force malformed ROIs to be 1x1
    const int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    const float bin_size_h = static_cast<float>(roi_height)
                    / static_cast<float>(pooled_height);
    const float bin_size_w = static_cast<float>(roi_width)
                    / static_cast<float>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                    * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                    * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    float maxval = is_empty ? 0 : -1E+37;
    // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
    int maxidx = -1;
    const int data_offset = (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width + w;
            if (bottom_data[data_offset + bottom_index] > maxval) {
                maxval = bottom_data[data_offset + bottom_index];
                maxidx = bottom_index;
            }
        }
    }
    top_data[idx]=maxval;
    argmax_data[idx]=maxidx;
    }
'''
kernel_backward = '''
    extern "C"
    __global__ void roi_backward(const float* const top_diff,
         const int* const argmax_data,const float* const bottom_rois,
         float* bottom_diff, const int num_rois,
         const double spatial_scale, int channels,
         int height, int width, int pooled_height,
          int pooled_width,const int NN)
    {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    ////Importtan >= instead of >
    if(idx>=NN)
        return;
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx/ (width * height)) % channels;
    int num = idx / (width * height * channels);

    float gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
        // Skip if ROI's batch index doesn't match num
        if (num != static_cast<int>(bottom_rois[roi_n * 5])) {
            continue;
        }

        int roi_start_w = round(bottom_rois[roi_n * 5 + 1]
                                * spatial_scale);
        int roi_start_h = round(bottom_rois[roi_n * 5 + 2]
                                * spatial_scale);
        int roi_end_w = round(bottom_rois[roi_n * 5 + 3]
                                * spatial_scale);
        int roi_end_h = round(bottom_rois[roi_n * 5 + 4]
                                * spatial_scale);

        // Skip if ROI doesn't include (h, w)
        const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                h >= roi_start_h && h <= roi_end_h);
        if (!in_roi) {
            continue;
        }

        int offset = (roi_n * channels + c) * pooled_height
                        * pooled_width;

        // Compute feasible set of pooled units that could have pooled
        // this bottom unit

        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);

        float bin_size_h = static_cast<float>(roi_height)
                        / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width)
                        / static_cast<float>(pooled_width);

        int phstart = floor(static_cast<float>(h - roi_start_h)
                            / bin_size_h);
        int phend = ceil(static_cast<float>(h - roi_start_h + 1)
                            / bin_size_h);
        int pwstart = floor(static_cast<float>(w - roi_start_w)
                            / bin_size_w);
        int pwend = ceil(static_cast<float>(w - roi_start_w + 1)
                            / bin_size_w);

        phstart = min(max(phstart, 0), pooled_height);
        phend = min(max(phend, 0), pooled_height);
        pwstart = min(max(pwstart, 0), pooled_width);
        pwend = min(max(pwend, 0), pooled_width);
        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                int index_ = ph * pooled_width + pw + offset;
                if (argmax_data[index_] == (h * width + w)) {
                    gradient += top_diff[index_];
                }
            }
        }
    }
    bottom_diff[idx] = gradient;
    }
'''
