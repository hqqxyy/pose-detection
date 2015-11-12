function aboxes = proposal_test(conf, imdb, varargin)
% aboxes = proposal_test(conf, imdb, varargin)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------   

%% inputs
    ip = inputParser;
    ip.addRequired('conf',                              @isstruct);
    ip.addRequired('imdb',                              @isstruct);
    ip.addParamValue('net_def_file',    fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'test.prototxt'), ...
                                                        @isstr);
    ip.addParamValue('net_file',        fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'Zeiler_conv5.caffemodel'), ...
                                                        @isstr);
    ip.addParamValue('cache_name',      'Zeiler_conv5', ...
                                                        @isstr);
                                                    
    ip.addParamValue('suffix',          '',             @isstr);
    
    ip.addParamValue('model',           '',             @isstruct);
    
    ip.parse(conf, imdb, varargin{:});
    opts = ip.Results;
    

    cache_dir = fullfile(pwd, 'output', 'rpn_cachedir', opts.cache_name, imdb.name);
%     try
%         % try to load cache
%         ld = load(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]));
%         aboxes = ld.aboxes;
%         clear ld;
%     catch    
        %% init net
        % init caffe net
    mkdir_if_missing(cache_dir);
    caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
    caffe.init_log(caffe_log_file_base);
    caffe_net = caffe.Net(opts.net_def_file, 'test');
    caffe_net.copy_from(opts.net_file);

    % init log
    timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
    mkdir_if_missing(fullfile(cache_dir, 'log'));
    log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
    diary(log_file);

    % set random seed
    prev_rng = seed_rand(conf.rng_seed);
    caffe.set_random_seed(conf.rng_seed);

    % set gpu/cpu
    if conf.use_gpu
        caffe.set_mode_gpu();
    else
        caffe.set_mode_cpu();
    end             

    disp('opts:');
    disp(opts);
    disp('conf:');
    disp(conf);

%% testing
    num_images = length(imdb.image_ids);
    % all detections are collected into:
    %    all_boxes[image] = N x 5 array of detections in
    %    (x1, y1, x2, y2, score)
    aboxes = cell(num_images, 1);
    abox_deltas = cell(num_images, 1);
    aanchors = cell(num_images, 1);
    ascores = cell(num_images, 1);

    count = 0;
    for i = 1:num_images
        count = count + 1;
        fprintf('%s: test (%s) %d/%d ', procid(), imdb.name, count, num_images);
        th = tic;
        im = imread(imdb.image_at(i));

        [boxes, scores, abox_deltas{i}, aanchors{i}, ascores{i}] = proposal_im_detect(conf, caffe_net, im);

        fprintf(' time: %.3fs\n', toc(th));  

        if isempty(opts.model)
            aboxes{i} = [boxes, scores];
        else
            % boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
            aboxes{i} = boxes_filter([boxes, scores], opts.model.nms.per_nms_topN,...
                                                      opts.model.nms.nms_overlap_thres,...
                                                      opts.model.nms.after_nms_topN,...
                                                      conf.use_gpu);
        end
    end  

%         tic;
%         % save(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]), 'aboxes', '-v7.3');
%         savefast(fullfile(cache_dir, ['proposal_boxes_' imdb.name opts.suffix]), 'aboxes');
%         toc;
    diary off;
    caffe.reset_all(); 
    rng(prev_rng);
%     end
end


function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end




