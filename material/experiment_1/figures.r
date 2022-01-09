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


#png("mygraphic.png",  width=1000)
#cnn$group = 'Cnn'
#camembert$group = 'Camembert'
#camembert_pretrained$group = 'Camembert+pretraining'
#visuals = rbind(cnn, camembert,camembert_pretrained)
#visuals$vis=c(rep("red",8),rep("blue",8),rep("green",8))
#ggplot(visuals, aes(y=eval_f1, x=trainsize,group=interaction(group),col=group)) + 
#   geom_point() + geom_line()

#dev.off()
#browseURL("mygraphic.png") 


