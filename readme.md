# iANP-EC: Identifying Anticancer Natural Products Using Ensemble Learning Incorporated with Evolutionary Computation

#### L. Nguyen, T-H Nguyen-Vo, Q. H. Trinh, B. H. Nguyen, P-U. Nguyen-Hoang, [L. Le](http://cbc.bio.hcmiu.edu.vn/)∗, and [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗

![alt text](https://github.com/mldlproject/2021-iANP-EC/blob/main/iANP-EC_abs.svg)

## Motivation
Cancer is one of the most deadly diseases that annually kills millions of people worldwide. The investigation on anticancer medicines has never ceased to seek better 
and more adaptive agents with fewer side effects. Besides chemically synthetic anticancer compounds, natural products are scientifically proved as a highly potential 
alternative source for anticancer drug discovery. Along with experimental approaches being used to find anticancer drug candidates, computational approaches have been 
developed to virtually screen for potential anticancer compounds. In this study, we construct an ensemble computational framework, called iANP-EC, using machine learning 
approaches incorporated with evolutionary computation. Four learning algorithms (k-NN, SVM, RF, and XGB) and four molecular representation schemes are used to build a set 
of classifiers, among which the top-four best-performing classifiers are selected to form an ensemble classifier. Particle swarm optimization (PSO) is used to optimise 
the weights used to combined the four top classifiers. The models are developed by a set of curated 997 compounds which are collected from the NPACT and CancerHSP databases. 

## Results
The results show that iANP-EC is a stable, robust, and effective framework that achieves an AUC-ROC value of 0.9193 and an AUC-PR value of 0.8366. The comparative analysis of 
molecular substructures between natural anticarcinogens and nonanticarcinogens partially unveils several key substructures that drive anticancerous activities. We also deploy 
the proposed ensemble model as an online web server with a user-friendly interface to support the research community in identifying natural products with anticancer activities.

## Availability and implementation
Source code and data are available on [GitHub](https://github.com/mldlproject/2021-iANP-EC).

## Web-based Application
- Source 1 ([Click here](http://124.197.54.240:8002/))
- Source 2 ([Click here](http://14.177.208.167:8002/))

## Citation
Loc Nguyen, Thanh-Hoang Nguyen Vo, Quang H. Trinh, Bach Hoai Nguyen, Phuong-Uyen Nguyen-Hoang, Ly Le*, and Binh P. Nguyen* (2022). iANP-EC: Identifying Anticancer Natural Products Using Ensemble Learning Incorporated with Evolutionary Computation. 
*Journal of Chemical Information and Modeling, 62(21), 5080-5089*. [DOI: 10.1021/acs.jcim.1c00920](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00920).

## Contact 
[Go to contact information](https://homepages.ecs.vuw.ac.nz/~nguyenb5/contact.html)
