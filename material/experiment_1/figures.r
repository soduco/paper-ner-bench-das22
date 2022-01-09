library(dplyr)
library(ggplot2)
library(xtable)
library(kableExtra)

# BERT : jq -r ' . | [ .[] ]  | @csv' $(l -1v ./*_test.json)
# CNN : jq -r 'with_entries(select([.key] | inside( ["ents_p","ents_r","ents_f"] ) ) ) | [ .[] ]  | @csv' $(l -1v ./*_test.json)


# Spacy CNN
cnn = read.csv("cnn/metrics.csv",header=T)
cnn = cnn %>% arrange(trainsize) %>% group_by(trainsize) %>% filter(dataset == 'test') %>% summarise(across(everything(), mean))

# HF Camembert 
camembert = read.csv("camembert/metrics.csv",header=T)
camembert = camembert %>% arrange(trainsize) %>% group_by(trainsize) %>% filter(dataset == 'test') %>% summarise(across(everything(), mean))

# HF Camembert + pretraining
camembert_pretrained = read.csv("camembert_pretrained/metrics.csv",header=T)
camembert_pretrained = camembert_pretrained %>% arrange(trainsize) %>% group_by(trainsize) %>% filter(dataset == 'test') %>% summarise(across(everything(), mean))


# Reshape for LaTeX export
tbl = cbind(cnn %>% select(eval_f1), camembert %>% select(eval_f1), camembert_pretrained %>% select(eval_f1))
colnames(tbl) <- c("Cnn", "Camembert", "Camembert+pretraining")

tbl = t(tbl)
colnames(tbl) <- camembert[['trainsize']]


# TODO : max value of each column in bold
#fn = function(x) cell_spec(x, bold = T)  
#tbl[, 2:8] = t(apply(tbl[,2:8], 1, fn))    

tbl %>%
    kable(booktabs = TRUE, digits=3, caption = "Train size vs F1 score", align = "c", 'latex') %>% 
    kable_styling(latex_options = c("hold_position")) %>%
    cat(., file = "table.tex")


png("experiment1.png",
  width     = 20,
  height    = 10,
  units     = "cm",
  res       = 300,
  pointsize = 4)

cnn = cnn %>% select(eval_precision, eval_recall, eval_f1, trainsize)
camembert = camembert %>% select(eval_precision, eval_recall, eval_f1, trainsize)
camembert_pretrained = camembert_pretrained %>% select(eval_precision, eval_recall, eval_f1, trainsize)


cnn$model = 'Cnn'
camembert$model = 'Camembert'
camembert_pretrained$model = 'Camembert+pretraining'

visuals = rbind(cnn, camembert,camembert_pretrained)

ggplot(visuals, aes(y=eval_f1, x=trainsize, group=interaction(model), col=model)) + 
   geom_point(aes(shape = model)) + geom_line(aes(linetype = model)) + theme_light()

dev.off()
browseURL("experiment1.png") 


