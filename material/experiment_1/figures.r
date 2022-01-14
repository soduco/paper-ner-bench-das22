library(dplyr)
library(gridExtra)
library(ggplot2)
library(xtable)
library(kableExtra)
library(cowplot)

# BERT : jq -r ' . | [ .[] ]  | @csv' $(l -1v ./*_test.json)
# CNN : jq -r 'with_entries(select([.key] | inside( ["ents_p","ents_r","ents_f"] ) ) ) | [ .[] ]  | @csv' $(l -1v ./*_test.json)


# Spacy CNN
cnn = read.csv("spacy_ner/metrics.csv",header=T)
cnn = cnn %>% arrange(trainsize) %>% group_by(trainsize) %>% filter(dataset == 'test') %>% summarise(across(everything(), mean))

# HF Camembert 
camembert = read.csv("camembert/metrics.csv",header=T)
camembert = camembert %>% arrange(trainsize) %>% group_by(trainsize) %>% filter(dataset == 'test') %>% summarise(across(everything(), mean))

# HF Camembert + pretraining
camembert_pretrained = read.csv("camembert_pretrained/metrics.csv",header=T)
camembert_pretrained = camembert_pretrained %>% arrange(trainsize) %>% group_by(trainsize) %>% filter(dataset == 'test') %>% summarise(across(everything(), mean))



# EXPORT TABLE 1

# Reshape
tbl = cbind(cnn %>% select(eval_f1), camembert %>% select(eval_f1), camembert_pretrained %>% select(eval_f1), cnn %>% select(eval_precision), camembert %>% select(eval_precision), camembert_pretrained %>% select(eval_precision), cnn %>% select(eval_recall), camembert %>% select(eval_recall), camembert_pretrained %>% select(eval_recall))
colnames(tbl) <- c("SpaCy NER F1", "CamemBERT F1", "CamemBERT+pretraining F1", "SpaCy NER precision", "CamemBERT precision", "CamemBERT+pretraining precision", "SpaCy NER recall", "CamemBERT recall", "CamemBERT+pretraining recall")

tbl = t(tbl)
colnames(tbl) <- camembert[['trainsize']]


# TODO : max value of each column in bold
#fn = function(x) cell_spec(x, bold = T)  
#tbl[, 2:8] = t(apply(tbl[,2:8], 1, fn))    

tbl %>%
    kable(booktabs = TRUE, digits=3, caption = "Trainset cardinality vs F1 score", align = "c", 'latex') %>% 
    kable_styling(latex_options = c("hold_position")) %>%
    cat(., file = "table_1.tex")



cnn = cnn %>% select(eval_precision, eval_recall, eval_f1, trainsize, trainsize_percent)
camembert = camembert %>% select(eval_precision, eval_recall, eval_f1, trainsize, trainsize_percent)
camembert_pretrained = camembert_pretrained %>% select(eval_precision, eval_recall, eval_f1, trainsize, trainsize_percent)


cnn$model = 'SpaCy NER'
camembert$model = 'CamemBERT'
camembert_pretrained$model = 'CamemBERT+pretraining'

visuals = rbind(cnn, camembert,camembert_pretrained)

# F1 SCORE
png("f1_vs_trainsize.png",
  width     = 20,
  height    = 10,
  units     = "cm",
  res       = 300,
  pointsize = 4)


plot1 = ggplot(visuals, aes(y=eval_f1, x=trainsize, group=interaction(model), col=model)) + 
   ylab("F1 score") + xlab("|trainset|") +
   geom_point(aes(shape = model)) + geom_line(aes(linetype = model)) + theme_light()# +theme(legend.position = "none")   

plot2 = ggplot(visuals, aes(y=eval_precision, x=trainsize, group=interaction(model), col=model)) + 
   ylab("Precision") + xlab("|trainset|") +
   geom_point(aes(shape = model)) + geom_line(aes(linetype = model)) + theme_light() + theme(legend.position = "none")   

plot3 = ggplot(visuals, aes(y=eval_recall, x=trainsize, group=interaction(model), col=model)) + 
   ylab("Recall") + xlab("|trainset|") +
   geom_point(aes(shape = model)) + geom_line(aes(linetype = model)) + theme_light()  + theme(legend.position = "none")   

legend = get_legend(
  # create some space to the left of the legend
  plot1 + theme(legend.box.margin = margin(0, 0, 0, 100))
)

#plots = plot_grid(plot1, plot2, plot3,align = "h", ncol = 3, rel_heights = c(1/3, 1/3, 1/3))
#plot_grid(plots, legend,rel_widths = c(3, .4))
plot_grid(plot1)
dev.off()


