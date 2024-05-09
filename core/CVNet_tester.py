r""" Test code of Correlation Verification Network """
# written by Seongwon Lee (won4113@yonsei.ac.kr)

import torch
import core.checkpoint as checkpoint
from config import cfg
from model.CVNet_Rerank_model import CVNet_Rerank
from test.test_model import test_model, test_cvnet, test_transformer_model
from data.preprocess_feature import extract_superglobal, create_topk_set
import logging

logger = logging.getLogger(__name__)

logger.setLevel(level = logging.INFO)

handler = logging.FileHandler("log.txt")   

handler.setLevel(logging.INFO)

logger.addHandler(handler)

logger.info("Start print log")  
def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    print("=> creating CVNet_Rerank model")
    model = CVNet_Rerank(cfg.MODEL.DEPTH, cfg.MODEL.HEADS.REDUCTION_DIM, cfg.SupG.relup)
    # print(model)
    # cur_device = torch.cuda.current_device()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model

def __main__():
    """Test the model."""
    if cfg.TEST.WEIGHTS == "":
        print("no test weights exist!!")
    else:
        # Construct the model
        model = setup_model()
        # Load checkpoint
        checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
        # test_model(model, cfg.TEST.DATA_DIR, cfg.TEST.DATASET_LIST, cfg.TEST.SCALE_LIST, cfg.SupG.rerank, cfg.SupG.gemp, cfg.SupG.rgem, cfg.SupG.sgem, cfg.SupG.onemeval, cfg.MODEL.DEPTH, logger)
        # test_cvnet(model, cfg.TEST.DATA_DIR, cfg.TEST.DATASET_LIST, cfg.TEST.SCALE_LIST, cfg.TEST.TOPK_LIST)
        
        # extract_superglobal(model)
        # create_topk_set()
        test_transformer_model(model, cfg.TEST.DATA_DIR, cfg.TEST.DATASET_LIST, cfg.TEST.SCALE_LIST, cfg.SupG.gemp, cfg.SupG.rgem, cfg.SupG.sgem)
