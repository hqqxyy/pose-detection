function roidb = roidb_from_lsp(imdb, varargin)
% roidb = roidb_from_lsp(imdb, rootDir)
%   Builds an regions of interest database from imdb image
%   database. Uses precomputed selective search boxes available
%   in the R-CNN data package.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

ip = inputParser;
ip.addRequired('imdb',                  @isstruct);
% ip.addRequired('joint_dir',             @ischar);
% ip.addParamValue('extended',            false,          @islogical);
% ip.addParamValue('extended_joint_dir', 	'.',            @ischar);
ip.parse(imdb, varargin{:});
opts = ip.Results;

disp('using roidb...');
disp(opts);

roidb.name = imdb.name;
cache_file = ['./imdb/cache/roidb_' imdb.name];
if imdb.flip
    cache_file = [cache_file, '_flip'];
end
if ~isempty(imdb.degree)
    cache_file = [cache_file, '_degree', num2str(length(imdb.degree))];
end
cache_file = [cache_file, '_joint', num2str(imdb.num_joints)];
cache_file = [cache_file, '_scale', num2str(imdb.scale_factor)];
cache_file = [cache_file, '.mat'];

try
    load(cache_file);
catch
    addpath(fullfile(fileparts(imdb.image_dir), 'LSPcode')); 
    joint_order = [14,13,9,8,7,3,2,1,10,11,12,4,5,6];
    roidb.name = imdb.name;
    lsp_joints = parload(fullfile(fileparts(imdb.image_dir), 'joints.mat'), 'joints');
    % convert to person-centric
    lsp_joints = lsp_pc2oc(lsp_joints);
    pos = struct('joints', cell(1000, 1), ...
    'r_degree', cell(1000, 1), 'isflip', cell(1000,1));
    switch imdb.image_set
        case('trainval')
            for i = 1:1000
                pos(i).joints = lsp_joints(1:2,joint_order, i)';
                pos(i).r_degree = 0;
                pos(i).isflip = 0;
            end
        case('test')
            for i = 1001:2000
                pos(i - 1000).joints = lsp_joints(1:2,joint_order, i)';
                pos(i - 1000).r_degree = 0;
                pos(i - 1000).isflip = 0;
            end
        otherwise
            error('usage = ''trainval'' or ''test''');
    end
    
    
    %% extended
    if imdb.extended
        extended_lsp_joints = parload(fullfile(fileparts(imdb.extended_dir), 'joints.mat'), 'joints');
        extended_lsp_joints = lsp_pc2oc(extended_lsp_joints);
        extended_pos = struct('joints', cell(10000, 1), ...
            'r_degree', cell(10000, 1), 'isflip', cell(10000,1));
        for i = 1:10000
        	extended_pos(i).joints = extended_lsp_joints(1:2,joint_order, i)';
            extended_pos(i).r_degree = 0;
            extended_pos(i).isflip = 0;
        end
        pos = cat(1, pos, extended_pos);
    end 
    
    %% convert to 14 or 26 joints
    switch imdb.num_joints
        case 14
            Trans = eye(14,14);
            mirror = [1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8];
        case 26
            % -------------------
            % create ground truth joints for model training
            % We augment the original 14 joint positions with midpoints of joints,
            % defining a total of 26 joints
            I = [1  2  3  4   4   5  6   6   7  8   8   9   9   10 11  11  12 13  13  14 ...
            15 16  16  17 18  18  19 20  20  21  21  22 23  23  24 25  25  26];
            J = [1  2  3  3   4   4  4   5   5  3   6   3   6   6  6   7   7  7   8   8 ...
            9  9  10  10  10  11  11 9   12  9   12  12 12  13  13 13  14  14];
            A = [1  1  1  1/2 1/2 1  1/2 1/2 1  2/3 1/3 1/3 2/3 1  1/2 1/2 1  1/2 1/2 1 ...
            1  1/2 1/2 1  1/2 1/2 1  2/3 1/3 1/3 2/3 1  1/2 1/2 1  1/2 1/2 1];
            Trans = full(sparse(I,J,A,26,14));
            mirror = [1,2,15,16,17,18,19,20,21,22,23,24,25,26,3,4,5,6,7,8,9,10,11,12,13,14];
            joint_parent_id = [0 1 2 3 4 5 6 3 8 9 10 11 12 13 2 15 16 17 18 15 20 21 22 23 24 25];
        otherwise
            error('num_joints = %d is not supported!!', p_no);
    end
    
    for i = 1:numel(pos)
        pos(i).joints = Trans * pos(i).joints; % linear combination
    end
    
    % same to xianjie's code
    [pos] = init_scale(pos, joint_parent_id, 4);
    
    if 0 % debug
        for i = 1:numel(pos)
            %showskeletons(imread( imdb.image_at(i)), roidb.rois(i).boxes, joint_parent_id);
            show_imjoints(imread( imdb.image_at(80*i - 79)), pos(i).joints');
            pause(0.5);
        end
    end
    
    %% add rotate
    if ~isempty(imdb.degree)
        num_rotate_pos = (length(imdb.degree) + 1) * numel(pos);
        rotate_pos = repmat(pos, length(imdb.degree) + 1, 1);
        rotate_pos(1:((length(imdb.degree) + 1)):num_rotate_pos) = pos;
        for i = 1:numel(pos)
            tic_toc_print('rotate rois (%s): %d/%d\n', imdb.name, i, numel(pos));           
            for j = 1:length(imdb.degree)
                rotate_id = (i - 1)*(length(imdb.degree) + 1) + j + 1;
                ori_imsize_id = 2*(i - 1)*(length(imdb.degree) + 1) + 1;
                im_size = imdb.sizes(ori_imsize_id, :);
                rotate_pos(rotate_id).joints = lsp_map_rotate_points(pos(i).joints, im_size, imdb.degree(j),'ori2new');
                rotate_pos(rotate_id).r_degree = imdb.degree(j);
                if 0
                    show_imjoints(imdb.image_at(2*rotate_id - 1), rotate_pos(rotate_id).joints');
                end
            end
        end
        pos = rotate_pos;
    end
    
    
        
    if 0 % debug
        for i = 1:numel(pos)
            %showskeletons(imread( imdb.image_at(i)), roidb.rois(i).boxes, joint_parent_id);
            show_imjoints(imread( imdb.image_at(2*i - 1)), pos(i).joints');
            pause(0.2);
        end
    end
    
    %% add flip
    if imdb.flip
        flip_pos = repmat(pos, 2, 1);
        flip_pos(1:2:numel(flip_pos)) = pos;
        for i = 1:numel(pos)
            tic_toc_print('flip rois (%s): %d/%d\n', imdb.name, i, numel(pos)); 
            imsize_id = 2*i - 1;
            imsize = imdb.sizes(imsize_id, :);
            flip_pos(2*i).joints(mirror, 1) = imsize(2) - pos(i).joints(:, 1) + 1;
            flip_pos(2*i).joints(mirror, 2) = pos(i).joints(:, 2);
        end
        pos = flip_pos;
    end
    

    %% convert to bbox
    scale_factor = imdb.scale_factor;
    for i = 1:numel(pos)
        tic_toc_print('convert joint to bbox (%s): %d/%d\n', imdb.name, i, numel(pos));
        bbox(:,1) = max(1, pos(i).joints(:, 1) - scale_factor * pos(i).scale_x);
        bbox(:,2) = max(1, pos(i).joints(:, 2) - scale_factor * pos(i).scale_y);
        bbox(:,3) = min(imdb.sizes(i, 2), pos(i).joints(:, 1) + scale_factor * pos(i).scale_x);
        bbox(:,4) = min(imdb.sizes(i, 1), pos(i).joints(:, 2) + scale_factor * pos(i).scale_y);
        pos(i).bbox = round(bbox);
        roidb.rois(i) = attach_lsp(pos(i).bbox, imdb.class_to_id);        
    end
        
    if 0 % debug
        for i = 1:numel(imdb.image_ids)
%             show_imjoints(imread( imdb.image_at(i)), pos(i).joints');
            show_boxes(imread( imdb.image_at(i)), pos(i).bbox, joint_parent_id);
            pause(0.5);
        end
    end
    
    rmpath(fullfile(fileparts(imdb.image_dir), 'LSPcode')); 
    
    fprintf('Saving roidb to cache...');
    save(cache_file, 'roidb', '-v7.3');
    fprintf('done\n');
end



% ------------------------------------------------------------------------
function rec = attach_lsp(boxes, class_to_id)
% ------------------------------------------------------------------------

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]

gt_boxes = boxes;
all_boxes = boxes;
num_gt_boxes = size(boxes, 1);
gt_classes = (1:num_gt_boxes)';
num_boxes = 0;
rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, class_to_id.Count, 'single');
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));

