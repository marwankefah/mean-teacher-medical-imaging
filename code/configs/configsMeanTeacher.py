# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:47:42 2021

@author: Prinzessin
"""
import configparser
import os

class Configs:
    def __init__(self, filename):

        # =============================================================================
        # Readable ini file
        # =============================================================================
        self.config_filename = filename
        config_file = configparser.ConfigParser(allow_no_value=True)
        config_file.read(self.config_filename)

        self.root_path = config_file.get('path', 'root_path', fallback='../data/FETA/')
        self.linux_gpu_id=config_file.getint('path','linux_gpu_id',fallback=1)
        self.linux=config_file.getboolean('path','linux',fallback=False)

        self.exp = config_file.get('path','exp' ,fallback='FETA/Mean_Teacher')
        self.model = config_file.get('path','model', fallback='unetResnet34')
        self.psuedoLabelsGenerationEpoch = config_file.getint('network','psuedoLabelsGenerationEpoch',fallback=3)
        self.meanTeacherEpoch = config_file.getint('network','meanTeacherEpoch',fallback=3)
        self.num_workers = config_file.getint('network','num_workers',fallback=0)

        self.val_batch_size = config_file.getint('network','val_batch_size',fallback=16)

        self.generationLowerThreshold = config_file.getfloat('network','generationLowerThreshold',fallback=0.05)
        self.generationHigherThreshold = config_file.getfloat('network','generationHigherThreshold',fallback=0.02)

        if self.linux==True:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.linux_gpu_id

        self.backbone = config_file.get('network','backbone',fallback='resnet34')
        self.max_iterations = config_file.getint('network','max_iterations',fallback=30000)
        self.batch_size = config_file.getint('network','batch_size',fallback=16)
        self.labeled_bs = config_file.getint('network','labeled_bs',fallback= 8)
        self.deterministic = config_file.getint('network','deterministic',fallback= 1)
        self.base_lr = config_file.getfloat('network','base_lr' ,fallback=0.01)
        patch_size = config_file.get('network','patch_size',fallback='[256, 256]')
        self.patch_size = [int(number) for number in patch_size[1:-1].split(',')]

        self.seed = config_file.getint('network','seed' ,fallback=1337)
        self.num_classes = config_file.getint('network','num_classes',fallback= 2)

        # costs
        self.ema_decay = config_file.getfloat('network','ema_decay',fallback=0.99)
        self.consistency_type = config_file.get('network','consistency_type',fallback='mse')
        self.consistency = config_file.getfloat('network','consistency',fallback=0.1)
        self.consistency_rampup = config_file.getfloat('network','consistency_rampup',fallback= 200.0)

