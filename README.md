# mean-teacher-medical-imaging
Mean Teacher with Medical Imaging Data Augmentation (torchio) 

FETA dataset with 5% labeled data | Dice Score  
---------------------------------|------------------------
Mean Teacher ResNet-34          |  **0.5682 ± 0.22**
Supervised Approach ResNet-34|  0.5451 ± 0.19

Original Image           |  Elastic Deformation  
:-------------------------:|:-------------------------:
![](https://github.com/marwankefah/mean-teacher-medical-imaging/blob/master/readme/original.png)  |  ![](https://github.com/marwankefah/mean-teacher-medical-imaging/blob/master/readme/elastic_deformation.png)
 
 Random Affine        |  Random Motion
:-------------------------:|:-------------------------:
![](https://github.com/marwankefah/mean-teacher-medical-imaging/blob/master/readme/random_affine.png)  |  ![](https://github.com/marwankefah/mean-teacher-medical-imaging/blob/master/readme/random_motion.png)

  Random Flip         |  Random Blur
:-------------------------:|:-------------------------:
![](https://github.com/marwankefah/mean-teacher-medical-imaging/blob/master/readme/random_flip.png)  |  ![](https://github.com/marwankefah/mean-teacher-medical-imaging/blob/master/readme/random_blur.png)

Bano, S. et al. (2020). Deep Placental Vessel Segmentation for Fetoscopic Mosaicking. In: , et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2020. MICCAI 2020. Lecture Notes in Computer Science(), vol 12263. Springer, Cham. https://doi.org/10.1007/978-3-030-59716-0_73
