# iANP-EC: Identifying Anticancer Natural Products using Ensemble Learning Incorporated with Evolutionary Computing

#### L. Nguyen, T-H Nguyen-Vo, Q. H. Trinh, B. H. Nguyen, P-U. Nguyen-Hoang, [L. Le](http://cbc.bio.hcmiu.edu.vn/)∗, and [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗

![alt text](https://github.com/mldlproject/2021-iANP-EC/blob/main/iANP-EC_abs.svg)

## Motivation
Cancer is one of the most deadly diseases that annually kill millions of people
worldwide. The investigation on anticancer medicines has never ceased to seek better
and more adaptive ones with fewer side effects. Besides chemically synthetic anticancer
drugs, natural products are scientifically proved as a highly potential alternative source
for anticancer drug discovery. Along with experimental approaches to find anticancer
 drug candidates, computational approaches have been developed to virtually screen
for potential anticancer compounds. Most of current computational approaches, however, focus on
anticancer activity of peptides only. In this study, we construct an ensemble computational framework 
called iANP-EC using machine learning approaches incorporated with evolutionary computing. 
Four learning algorithms (k-NN, SVM, RF, and XGB) are used incorporated with four molecular representation 
schemes to build a set of classifiers, among which the top-four best-performing classifiers are selected 
to form an ensemble classifier. The proposed ensemble model uses a set of weights that are
tuned with respect to the ROC-AUC measure using Particle Swarm Optimization. The number of curated 
chemical data used is 1011 samples collected from the NPACT and CancerHSP databases.

## Results
The results show that iANP-EC is a stable, robust, and effective framework which achieves a 
ROC-AUC of 0.8812 and a PR-AUC of 0.8446. The comparative analysis of molecular substructures 
between natural anticarcinogen and non-anticarcinogen partially unveil several key substructures 
that have essential impacts on acting as anticancer agents.


## Availability and implementation
Source code and data are available on [GitHub](https://github.com/mldlproject/2021-iANP-EC).

## Web-based Application
[Click here](http://192.168.1.19:8003/) 

## Contact 
[Go to contact information](https://homepages.ecs.vuw.ac.nz/~nguyenb5/contact.html)
