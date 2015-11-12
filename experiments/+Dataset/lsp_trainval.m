function dataset = lsp_trainval(dataset, usage, use_flip)
% LSP trainval set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install

d_step = 18;
degree = [-180+d_step:d_step:-d_step,d_step:d_step:180];
switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_lsp(fullfile(pwd, 'datasets', 'LSP'),...
            'trainval',...
            'extended',        false,...
            'extendedDir',     fullfile(pwd, 'datasets', 'LSPET'),...
            'extension',       'jpg',...
            'flip',            use_flip,...
            'degree',          degree... degree
            ) };       
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
        
    case {'test'}
    otherwise
        error('usage = ''train'' or ''test''');
end

end