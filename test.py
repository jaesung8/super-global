# written by Seongwon Lee (won4113@yonsei.ac.kr)

import config as config
import core.CVNet_tester as CVNet_tester

from config import cfg

from utils.training import train

def main():
    config.load_cfg_fom_args("test a CVNet model.")
    cfg.NUM_GPUS=1
    cfg.freeze()
    CVNet_tester.__main__()
    
    
    # train()



if __name__ == "__main__":
    main()
