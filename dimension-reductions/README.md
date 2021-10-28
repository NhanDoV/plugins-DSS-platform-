# Instructions
- This recipes is used to reduce the dimenison of the input dataset by the PCA (Principal Component Analysis), refered [sklearn.pca](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) then displayed on DSS DataIku.
- Icon of this plugin on DSS, choose your dataset then find
<p align="center">
<img width="10%" src="plugin_icon.jpg">
</p>

after set the name of the output datasets, you will have

<p align="center">
<img width="40%" src="plugins.jpg"
</p>

# Input parameters
<p align="center">
<img width="120%" src="plugins_params.jpg"
</p>
 
# Outputs
## The pca_dataset
This is the dataset after you implement the PCA algorithm 
<p align="center">
<img width="70%" src="output1_pca-dataset.jpg"
</p>  
  
## The PC_weights
This is the weights of each PC, you can recover the original dataset or verify the `pca_dataset` by the multiplication with the `weight_matrix` or its inverse matrix.
<p align="center">
<img width="70%" src="output1_pca-dataset.jpg"
</p>  
