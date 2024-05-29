#install.packages("ggplot2")
#install.packages("tidyverse")
#install.packages("ggrepel")

library(ggplot2)
library(tidyverse)
library(ggrepel)

# before running this script, please run create_draw_predict_data.py first.

for (name in c("BA5","XBB15")) {
    data <- read.csv(paste("./",name,"_data.csv",sep=""))
    test <- data %>% filter(kind == 'target' | kind == 'mutation') %>% filter(num > 2)
    test <- test[,3]
    color_ <- c('target'=alpha(c("white"),0.5), 'mutation'=alpha(c("yellow"),0.8))
    force_label <- c()
    pdf(paste("./plot_",name,".pdf",sep=""), width=12, height=3)
    for (tp in c("1")) { #"BA5_IC50", "XBB1_5_IC50"
        p <- ggplot(data , aes(index, num)) + 
            geom_line(color="#E99793", size=0.8, alpha=0.8) + geom_point(color="#DD6C70", shape=21)+ theme_classic() + theme(
                axis.text.y=element_blank(),
                axis.ticks.y=element_blank(),
                axis.text.x=element_text(angle=90, vjust=0.5))+
                scale_x_continuous(breaks=seq(331,520,2))+
                scale_color_manual(values=colors)+
            ylab('Screened variants number')+theme(axis.title.y = element_text(size = 13))+
            xlab('')+
            geom_label_repel(data=data %>% filter(kind == 'target' | kind == 'mutation') %>% filter(num > 2) %>% group_by(index) %>% summarise(num=max(num)), 
                            aes(label=index), min.segment.length = 0, direction="both", fill = color_[test])
        print(p)
    }
    dev.off()
}
