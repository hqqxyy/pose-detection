function dataset = lsp_test(dataset, usage, use_flip)
% Pascal voc 2007 test set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
degree = [];
switch usage
    case {'train'}
        dataset.imdb_train    = {  imdb_from_voc(devkit, 'test', '2007', use_flip) };
        dataset.roidb_train   = cellfun(@(x) x.roidb_func(x), dataset.imdb_train, 'UniformOutput', false);
    case {'test'}
        dataset.imdb_test    = imdb_from_lsp(fullfile(pwd, 'datasets', 'LSP'),...
            'test',...
            'extended',        false,...
            'extendedDir',     fullfile(pwd, 'datasets', 'LSPET'),...
            'extension',       'jpg',...
            'flip',            use_flip,...
            'degree',          degree... degree
            );       
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    otherwise
        error('usage = ''train'' or ''test''');
end

end