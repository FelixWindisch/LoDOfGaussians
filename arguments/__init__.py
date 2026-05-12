#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


from argparse import ArgumentParser, Namespace
#from stp_gaussian_rasterization import ExtendedSettings, GlobalSortOrder, SortMode
import json
import sys
import os
from distutils.util import strtobool
class SplattingSettings():
    
    group_config = None
    group_settings = None
    #settings = ExtendedSettings()
    parser = None
    render = False
    
    def __init__(self, parser, render=False):
        self.parser = parser
        self.render = render
        if not render:
            self.group_config = parser.add_argument_group("Splatting Config")
            self.group_config.add_argument("--splatting_config", type=str)
            
        bool_ = lambda x: bool(strtobool(x))
        
        self.group_settings = parser.add_argument_group("Splatting Settings")
        self.group_settings.add_argument("--sort_mode", type=lambda sortmode: SortMode[sortmode], choices=list(SortMode))
        self.group_settings.add_argument("--sort_order", type=lambda sortorder: GlobalSortOrder[sortorder], choices=list(GlobalSortOrder))
        self.group_settings.add_argument("--tile_4x4", type=int, choices=[64], help='only needed if using sort_mode HIER')
        self.group_settings.add_argument("--tile_2x2", type=int, choices=[8,12,20], help='only needed if using sort_mode HIER')
        self.group_settings.add_argument("--per_pixel", type=int, choices=[1,2,4,8,12,16,20,24], help='if using sort_mode HIER, only {4,8,16} are valid')
        self.group_settings.add_argument("--rect_bounding", type=bool_, choices=[True, False], help="Bound 2D Gaussians with a rectangle instead of a circle")
        self.group_settings.add_argument("--tight_opacity_bounding", type=bool_, choices=[True, False], help="Bound 2D Gaussians by considering their opacity")
        self.group_settings.add_argument("--tile_based_culling", type=bool_, choices=[True, False], help="Cull complete tiles based on opacity")
        self.group_settings.add_argument("--hierarchical_4x4_culling", type=bool_, choices=[True, False], help="Cull Gaussians for 4x4 subtiles, only when using sort_mode HIER")
        self.group_settings.add_argument("--load_balancing", type=bool_, choices=[True, False], help="Perform per-tile computations cooperatively (e.g. duplication)")
        self.group_settings.add_argument("--proper_ewa_scaling", type=bool_, choices=[True, False], help='Dilation of 2D Gaussians as proposed by Yu et al. ("Mip-Splatting")')
    
    def get_settings(self, arguments):
        # get valid choices from configargparse
        config = None
        
        # load default dict, if passed
        if self.render:
            cmdlne_string = sys.argv[1:]
            args_cmdline = self.parser.parse_args(cmdlne_string)
            cfgfilepath = os.path.join(args_cmdline.model_path, "config.json")
            print("Looking for splatting config file in", cfgfilepath)
            if os.path.exists(cfgfilepath):
                print("Config file found: {}".format(cfgfilepath))
                self.settings = ExtendedSettings.from_json(cfgfilepath)
            else:
                print("No config file found, assuming default values")
        else:
            for arg in vars(arguments).items():
                if any([arg[0] in z.option_strings[0] for z in self.group_config._group_actions]):
                    # json passed, load it
                    if arg[1] is None:
                        continue
                    with open(arg[1], 'r') as json_file:
                        config = json.load(json_file)
                        self.settings = ExtendedSettings.from_dict(config)
                    
        for arg in vars(arguments).items():
            if any([arg[0] in z.option_strings[0] for z in self.group_settings._group_actions]):
                # pass any options which were not given
                if arg[1] is None:
                    continue
                self.settings.set_value(arg[0], arg[1])
                
        return self.settings

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(self).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 1
        self._source_path = ""
        self._model_path = ""
        self._exp_name = ""
        self._images = "images"
        self._alpha_masks = "masks"
        self._depths = "depths"
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False # Include the left half of the test images in the train set to optimize exposures
        self.data_device = "cuda"
        self.eval = False
        self.skip_scale_big_gauss = False
        self.hierarchy = ""
        self.pretrained = ""
        self.skybox_num = 100_000
        self.scaffold_file = ""
        self.bounds_file = ""
        self.skybox_locked = False
        self.anti_aliasing = False
        # MCMC
        #self.eval = True
        
        #MCMC
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        # TODO: Fix this
        #g.source_path = "/home/felix-windisch/Datasets/example_dataset_LOD/example_dataset/camera_calibration/aligned"
        # What is up with this?
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 100_000
        self.coarse_iterations = 60_000
        self.SH_degree = 1
        self.SH_increase_after_train_percent = 0.1
        self.lr_multiplier = 1.0
        self.position_lr_init = 0.00002
        self.position_lr_final = 0.0000002
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.iterations
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.001
        self.exposure_lr_final = 0.0001
        self.exposure_lr_delay_steps = 5000
        self.exposure_lr_delay_mult = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 300
        self.densification = ["MCMC", "classic"][0]
        self.opacity_reset_interval = 10e10
        self.densify_from_iter = 100
        self.densify_until_iter = self.iterations - 10000
        self.densify_grad_threshold = 0.0015
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.llff_hold = -1
        self.vary_distance_multiplier = False
        self.output_file_name = "result.dhier"
        #MCMC
        if self.densification == "MCMC":
            self.noise_lr = 0 
            self.lambda_scaling = 0.00
            self.lambda_opacity = 0.00
        else:
            self.noise_lr = 5e5
            self.lambda_scaling = 0.01
            self.lambda_opacity = 0.01
        self.densify_percent =  1.02
        self.cap_max = 12_000_000
        #MCMC
        #A LoD of Gaussians
        self.graph_view_select = False
        self.view_graph_k = 100
        self.use_bounding_spheres = False
        self.use_frustum_culling = True
        self.use_occlusion_culling = False
        self.prune_unused = True
        self.dampen_scale_grad = False
        self.storage_device = 'cpu'
        self.SPT_root_volume = 10
        self.target_granularity_pixels = 2
        self.min_SPT_size = 256
        self.use_GPU_caching = True
        self.cache_size = 15_000_000
        self.cache_size_after_reduction = 12_000_000
        self.clear_cache_interval = 1000
        self.optimize_exposure = False
        #A LoD of Gaussians
        super().__init__(parser, "Optimization Parameters")
    
    def to_dict(self):
        return self.__dict__
    
def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
