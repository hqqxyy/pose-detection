function res = imdb_eval_lsp(cls, boxes, imdb, cache_name, suffix)

res.recall = 0;
res.prec = 0;
res.ap = 0;
res.ap_auc = 0;
