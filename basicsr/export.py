import logging
import torch
from os import path as osp

from basicsr import build_network
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def export_model(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    # create model
    model = build_model(opt)

    #export model
    net = model.net_g

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    example_inputs = (torch.randn(1, 3, 64, 64).to(device),)
    model = net.to(device)

    folding = True
    name = opt["name"] + ".onnx"
    try:
        if opt["export"]["custom_name"]:
            name = ["export"]["file_name"]
        folding = opt["export"]["constant_folding"]
    except:
        print("No export settings, using defaults")
    name = "./export/" + name
    print("Starting export of " + opt["name"])
    onnx_program = torch.onnx.export(model,
                                     example_inputs,
                                     f=name,
                                     verbose=False,
                                     export_params=True,
                                     do_constant_folding=folding
                                     )
    print("Export finished!")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    export_model(root_path)
