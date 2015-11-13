function gpu_id = auto_select_gpu()
% gpu_id = auto_select_gpu()
% Select the gpu which has the maximum free memory 
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    % deselects all GPU devices
    gpuDevice([]);

    maxFreeMemory = 0;
    if gpuDeviceCount == 1 % my lab computer
        fprintf('Lab computer...\n');
        gpu_cand_id = [1];
    else
        fprintf('Server computer...\n');
        gpu_cand_id = [5, 6, 7, 8];
    end
        
    for i = gpu_cand_id
        g = gpuDevice(i);
        freeMemory = g.FreeMemory();
        fprintf('GPU %d: free memory %d\n', i, freeMemory);
        if freeMemory > maxFreeMemory
            maxFreeMemory = freeMemory;
            gpu_id = i;
        end
    end
    fprintf('Use GPU %d\n', gpu_id);
    
    % deselects all GPU devices
    gpuDevice([]);
end
