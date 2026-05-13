**A LoD of Gaussians** features many different parameters to tune for reconstruction. The following gives an overview on how to achieve best reconstruction quality and speed. All of the hyperparameters are grouped into one .json file in the /config folder. The config file should then be passed with the ```--config``` flag for training, rendering and evaluation.
## Essential Hyperparameters
Always set these hyperparameters for your specific dataset
- *iterations:* How many iterations fine training should be run. As a rough guideline, we recommend using ten times the number of training images (or at least 30,000 for smaller data sets). 
- *densify_until_iter:* We recommend setting this to about 85% of training iterations. Disabling densification for the last training steps leads to better convergence.
- *cap_max:* Determines the maximal number of Gaussians to train. For large datasets, this should be as high as possible under your VRAM constraints. How many Gaussians can be handled highly depends on the scale of the scene, with vast datasets like MatrixCity supporting more than 150 million Gaussians, while smaller scale datasets like Uni10k are able to support 50 million on 24 GB of VRAM.Generally, the larger the area over which Gaussians are distributed, the higher this should be.
- *SPT_root_volume:* This parameter defines the volume that an SPT should take up in the scene. If this is large, there will be fewer SPTs with more Gaussians each and vice versa. At the start of training, the message "Built N SPTs, which contain P % of Gaussians" will be printed. As a guideline, P should be at least 95 % and N should be proportional to the scene size (e.g. 500 SPTs for a single chunk of H-3DGS dataset, 25k SPTs for the campus scene, 60k for full MatrixCity).
## Helpful Hyperparameters
- *target_granularity_pixels:* Sets the base level of detail. (i.e. the radius of a Gaussian of the correct level of detail should be *target_granularity_pixels*). Lowering this will increase Memory usage and training time, but would lead to better results at the highest detail. 
- *densify_grad_threshold:* Lowering this value increases densification (for example if the desired number of Gaussians is not spawned during training). See Hierarchical-3DGS (Kerbl et al. 2024) for details.
- *graph_view_select:* If True, selects subsequent training views according to the paper. Generally, this is a speed optimization, disabling it will slightly increase visual quality. 
- *view_graph_k:* With how many training views each view is connected in the view graph. 
- *llff_hold:* If you want to use evaluation on a dataset that does not provide a train/test split, this will use every nth image for the training set.
- *prune_unused:* This will cull Gaussians that have not contributed to any training views. 
- *densification interval:* Setting this value too small causes performance problems, because the hierarchy is rebuilt each time for densification. 
- *use_GPU_caching:* Saves a lot of performance at a small memory cost, generally should be enabled
- *cache_size:* The number of Gaussians that can be stored in the GPU cache, trades off memory for performance
- *cache_size_after_reduction:* Should be set about 20% lower than *cache_size*
## Finetuning Hyperparameters
- *densification_interval:* Since densification also rebuilds the hierarchy structure, this is a performance concern. We recommend having about 100 densification iterations throughout the entire training.
- *min_SPT_size:* This is the minimum number of Gaussians that can form an SPT. This simply prohibits the formation of many tiny SPTs, which would not be worth the overhead.
- *coarse_iterations:* How many iterations coarse training should be run. For extremely large datasets, increase this to 100-150k.
- *SH_degree:* Generally, we recommend setting this to 1. Increasing it to 2 or 3 might slightly improve image quality at great performance cost.
- *densification:* Either "classic" (Standard 3DGS) or "MCMC" (Kheradmand et al. 2024). We have found that "classic" produces better results on ultra-large scale datasets, but MCMC may increase quality if the scene is small and Gaussian Budget is high. **Note that MCMC densification is not compatible with gsplat rasterizer, because it requires per-primitive contribution**.
- *noise_lr, lambda_scaling, lambda_opacity, lambda_distance_sigma, densify_percent:* Parameters for MCMC densification (See Kheradmand et al. 2024), only used if *densification == "MCMC"*. 
- *use_bounding_spheres:* Use bounding sphere over the entire SPT subtree instead of three times the covariance of the root for frustum culling SPTs. Technically more accurate, but little difference in practice.
- *clear_cache_interval:* Clears the cache every *clear_cache_interval* iterations to prevent stagnation in the cache.
## Expert-only
- *_lr:* Anything ending in lr is a learning rate. Can affect the convergence behaviour, modify with caution, potentially if the XYZ dimensions of your dataset are significantly smaller or larger than standard
- *opacity_reset_interval:* Do not touch, opacity resets will destroy the LoD structure
- *lambda_dssim:* Trades off SSIM and L1 loss like in standard 3DGS
- *use_frustum_culling:* Generally should only be disabled for debugging purposes, as it increases performance by a lot without any negative effects.
- *storage_device:* Only CPU currently supported

