from net_3 import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction, AFF, iAFF
from net_3 import BasicLayer
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models_final/CDDFuse_Final1_106-14-10-39.pth"
for dataset_name in ["TNO","RoadScene"]:
    print("\n"*2+"="*80)
    model_name="CDDFuse    "
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name)
    test_out_folder=os.path.join('test_result_fianl1_1_1',dataset_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    BaseFuseLayer = nn.DataParallel(BasicLayer(dim=64,num_heads=8,depth=1,window_size=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    FeatureB_iAFF = nn.DataParallel(iAFF()).to(device)
    FeatureD_iAFF = nn.DataParallel(iAFF()).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    FeatureB_iAFF.load_state_dict(torch.load(ckpt_path)['FeatureB_iAFF'])
    FeatureD_iAFF.load_state_dict(torch.load(ckpt_path)['FeatureD_iAFF'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    FeatureB_iAFF.eval()
    FeatureD_iAFF.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):

            data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

            data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)

            feature_A_B_1 = FeatureB_iAFF(feature_V_B, feature_I_B)
            feature_F_D_1 = FeatureD_iAFF(feature_I_D, feature_V_D)

            B, C, W, H = feature_A_B_1.shape  # 获取输入的形状
            feature_A_B = feature_A_B_1.view(B, C, -1)  # 将 W 和 H 维度合并成一个维度
            feature_A_B = feature_A_B.permute(0, 2, 1)  # 调整维度顺序

            feature_F_B, W, H = BaseFuseLayer(feature_A_B, W, H)

            B1, WH1, C1 = feature_F_B.shape
            W1 = W
            H1 = H
            feature_F_B = feature_F_B.view(B1, W1, H1, C1)  # 将特征图的宽度和高度恢复
            feature_F_B = feature_F_B.permute(0, 3, 1, 2)  # 调整维度顺序

            feature_F_D = DetailFuseLayer(feature_F_D_1)

            # print(feature_F_B.shape)
            # print(feature_A_B.shape)

            feature_B = FeatureB_iAFF(feature_F_B, feature_A_B_1)
            feature_D = FeatureD_iAFF(feature_F_D, feature_F_D_1)

            data_Fuse, feature_F = Decoder(data_VIS, feature_B, feature_D)
            data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
            # fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            fi = np.squeeze((data_Fuse * 255).cpu().numpy().astype(np.uint8))
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)


    eval_folder=test_out_folder
    ori_img_folder=test_folder

    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder,"ir")):
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
            +str(np.round(metric_result[1], 2))+'\t'
            +str(np.round(metric_result[2], 2))+'\t'
            +str(np.round(metric_result[3], 2))+'\t'
            +str(np.round(metric_result[4], 2))+'\t'
            +str(np.round(metric_result[5], 2))+'\t'
            +str(np.round(metric_result[6], 2))+'\t'
            +str(np.round(metric_result[7], 2))
            )
    print("="*80)