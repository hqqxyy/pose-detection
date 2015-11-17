function imdb = imdb_from_lsp(root_dir, image_set, varargin)
% imdb = imdb_from_voc(root_dir, image_set, year)
%   Builds an image database for the LSP and LSP extension devkit located
%   at root_dir using the image_set and year.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2015, Qiqi Hou
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

%imdb.name = 'voc_train_2007'
%imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

ip = inputParser;
ip.addRequired('root_dir',          @ischar);
ip.addRequired('image_set',         @ischar);
ip.addParamValue('extended',        false,          @islogical);
ip.addParamValue('extendedDir',     '.',            @ischar);
ip.addParamValue('extension',       '',             @ischar);
ip.addParamValue('flip',            true,           @islogical);
ip.addParamValue('degree',          [],             @ismatrix);
ip.addParamValue('num_joints',      26);
ip.addParamValue('scale_factor',    1);
ip.parse(root_dir, image_set, varargin{:});
opts = ip.Results;

disp('using imdb...');
disp(opts);

cache_file = './imdb/cache/imdb_lsp';
if opts.extended
    cache_file = [cache_file, '_extended'];
end    
cache_file = [cache_file, '_', opts.image_set];
if opts.flip
    cache_file = [cache_file, '_flip'];
end
if ~isempty(opts.degree)
    cache_file = [cache_file, '_degree', num2str(length(opts.degree))];
end
cache_file = [cache_file, '_scale', num2str(length(opts.scale_factor))];
cache_file = [cache_file, '.mat'];

try
    load(cache_file);
catch
    imdb.name = 'lsp';
    if opts.extended
        imdb.name = [imdb.name, '_extended'];
    end  
    imdb.name = [imdb.name, '_', opts.image_set];
    imdb.image_dir = fullfile(opts.root_dir, 'images');
    imdb.extension = 'jpg';
    imdb.num_joints = opts.num_joints;
    image_ids = cell(1000, 1);
    image_ids_at = @(i) sprintf('im%04d', i);
    imdb.image_set = opts.image_set;
    imdb.scale_factor = opts.scale_factor;
    switch opts.image_set
        case('trainval')
            for i = 1:1000
                image_ids{i} = image_ids_at(i);
            end
        case('test')
            for i = 1001:2000
                image_ids{i - 1000} = image_ids_at(i);
            end
        otherwise
            error('usage = ''trainval'' or ''test''');
    end

    
    %% extended LSP
    imdb.extended = opts.extended;
    if opts.extended
        imdb.extended_dir = opts.extendedDir;
        imdb.extended_image_dir = fullfile(opts.extendedDir, 'images');
        extended_image_ids_at = @(i) sprintf('im%05d_et', i);
        src_extended_image_at = @(i) sprintf('%s/im%05d.%s', imdb.extended_image_dir, i, imdb.extension);
        dst_extended_image_at = @(i) sprintf('%s/im%05d_et.%s', imdb.image_dir, i, imdb.extension);
        extended_image_ids = cell(10000, 1);
        for i = 1:10000
            tic_toc_print('extend imdb (%s): %d/%d\n', imdb.name, i, length(extended_image_ids));
            extended_image_ids{i} = extended_image_ids_at(i);
            if ~exist(dst_extended_image_at(i), 'file')
                im = imread(src_extended_image_at(i));
                imwrite(im, dst_extended_image_at(i));
            end
        end
        image_ids = cat(1, image_ids, extended_image_ids);
    end  
    imdb.image_ids = image_ids;

    
    
    %% add rotate
    imdb.degree = opts.degree;
    if ~isempty(opts.degree)
        image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
        rotated_image_at = @(i, j) sprintf('%s/%s_rotate%04d.%s', imdb.image_dir, imdb.image_ids{i}, j, imdb.extension);
        for i = 1:length(imdb.image_ids)
            tic_toc_print('rotate imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));           
            for j = 1:length(opts.degree)
                if ~exist(rotated_image_at(i, opts.degree(j)), 'file')
                    im = imread(image_at(i));
                    im_rotate = imrotate(im, opts.degree(j));
                    imwrite(im_rotate, rotated_image_at(i, opts.degree(j)));
                end
            end
        end
        img_num = length(imdb.image_ids)*(length(opts.degree) + 1);
        image_ids = imdb.image_ids;
        imdb.image_ids(1:(length(opts.degree) + 1):img_num) = image_ids;
        for i = 1:length(opts.degree)
            imdb.image_ids(i+1:(length(opts.degree) + 1):img_num) = ...
                cellfun(@(x)  sprintf('%s_rotate%04d', x, opts.degree(i)),...
                image_ids, 'UniformOutput', false);
        end
    end
    
    %% add flip
    imdb.flip = opts.flip;
    if opts.flip
        image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
        flip_image_at = @(i) sprintf('%s/%s_flip.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
        for i = 1:length(imdb.image_ids)
            tic_toc_print('flip imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids)); 
            if ~exist(flip_image_at(i), 'file')
                im = imread(image_at(i));
                imwrite(fliplr(im), flip_image_at(i));
            end
        end
        img_num = length(imdb.image_ids)*2;
        image_ids = imdb.image_ids;
        imdb.image_ids(1:2:img_num) = image_ids;
        imdb.image_ids(2:2:img_num) = cellfun(@(x) [x, '_flip'], image_ids, 'UniformOutput', false);
        imdb.flip_from = zeros(img_num, 1);
        imdb.flip_from(2:2:img_num) = 1:2:img_num;
    end
    
    imdb.num_classes = opts.num_joints;
    imdb.classes = cell(imdb.num_classes, 1);
    for i = 1:imdb.num_classes
        imdb.classes{i} = sprintf('joint_%02d', i);
    end
    imdb.class_to_id = ...
    containers.Map(imdb.classes, 1:imdb.num_classes);
    imdb.class_ids = 1:imdb.num_classes;

    % private VOC details
    % imdb.details.VOCopts = VOCopts;

    % LSP specific functions for evaluation and region of interest DB
    imdb.eval_func = @imdb_eval_lsp;
    imdb.roidb_func = @roidb_from_lsp;
    imdb.image_at = @(i) ...
    sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);

    image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
    for i = 1:length(imdb.image_ids)
        tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));       
        im = imread(image_at(i));
        imsize = size(im);
        imdb.sizes(i, :) = [imsize(1) imsize(2)];
    end

    fprintf('Saving imdb to cache...');
    save(cache_file, 'imdb', '-v7.3');
    fprintf('done\n');
end
