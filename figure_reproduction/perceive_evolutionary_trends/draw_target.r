#install.packages("ggplot2")
#install.packages("tidyverse")
#install.packages("ggrepel")

library(ggplot2)
library(tidyverse)
library(ggrepel)

# before running this script, please run calculate_preference_target.ipynb first.

suff = "sum"
for (use_weight in c("BA5_IC50", "XBB1_5_IC50")) {
    data <- read.csv(paste("./tmp_data-",suff,"331_520.csv",sep=""))
    force_label <- c()
    pdf(paste("./target_",use_weight,".pdf",sep=""), width=12, height=3)
    for (use_src in c("BA.5+BF.7")) {

        p <- ggplot(data %>% filter(absrc==use_src & weight == use_weight), aes(site, mut_escape_adj)) + 
            #geom_line(color="#A03429", size=0.8, alpha=0.8) + geom_point(color="#A03429", shape=21)+ theme_classic() + theme(
            geom_line(color="#E99793", size=0.8, alpha=0.8) + geom_point(color="#DD6C70", shape=21)+ theme_classic() + theme(
                axis.text.y=element_blank(),
                axis.ticks.y=element_blank(),
                axis.text.x=element_text(angle=90, vjust=0.5))+
                #scale_x_continuous(breaks=seq(331,531,2))+
                scale_x_continuous(breaks=seq(331,520,2))+
                scale_color_manual(values=colors)+
            ylab('Normalized weighted\nescape score')+theme(axis.title.y = element_text(size = 13))+
            xlab('')+
            #ggtitle(paste("Source:", use_src, 
            #              ' Weight:', use_weight, 
            #              ' Expr:', data$expr_coef[1],
            #              ' Bind:', data$bind_coef[1],
            #              ' Codon:', data$is_codon[1],
            #              ' LogTrans:', data$is_neut_log[1],
            #              ' SiteMax:', data$is_max[1]))+
            geom_label_repel(data=data %>% filter(absrc == use_src & weight == use_weight & (mut_escape_adj > 0.2 | (mut_escape_adj > 0.03 & site %in% force_label))) %>% group_by(site) %>% summarise(mut_escape_adj=max(mut_escape_adj)), 
                            aes(label=site), min.segment.length = 0, direction="both", fill = alpha(c("white"),0.5))
        print(p)
    }
    dev.off()
}
