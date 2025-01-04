import logging
from tqdm import tqdm

import os

import numpy as np
from src.methods import setup_model
from src.utils.utils import get_accuracy, merge_cfg_from_args, get_args
from src.utils.conf import cfg, load_cfg_fom_args
from src.data.data import load_ood_dataset_test

logger = logging.getLogger(__name__)


def validation(cfg):
    model = setup_model(cfg)
    # get the test sequence containing the corruptions or domain names
    dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")


    severities = [cfg.CORRUPTION.SEVERITY[0]]

    accs = []
    aucs = []
    h_scores = []

    # start evaluation
    for i_dom, domain_name in enumerate(tqdm(dom_names_all, desc="Processing Corruptions", ncols=100)):
        if cfg.MODEL.CONTINUAL == 'Fully':
            try:
                model.reset()
                log_message = "Resetting model"
                logger.info(log_message)
                tqdm.write(log_message)
            except Exception as e:
                log_message = f"Model reset failed: {e}"
                logger.warning(log_message)
                tqdm.write(log_message)
        elif cfg.MODEL.CONTINUAL == 'Continual':
            log_message = "Continual learning mode: not resetting model"
            logger.info(log_message)
            tqdm.write(log_message)

        for severity in tqdm(severities, desc=f"Severity Level", leave=False, ncols=80):
            testset, test_loader = load_ood_dataset_test(
                cfg.DATA_DIR, 
                cfg.CORRUPTION.ID_DATASET,
                cfg.CORRUPTION.OOD_DATASET, 
                cfg.CORRUPTION.NUM_OOD_SAMPLES,
                batch_size=cfg.TEST.BATCH_SIZE,
                domain=domain_name, 
                level=severity,
                adaptation=cfg.MODEL.ADAPTATION,
                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                num_aug=cfg.TEST.N_AUGMENTATIONS if cfg.MODEL.ADAPTATION != 'stamp' else cfg.STAMP.NUM_AUG
            )

            for epoch in tqdm(range(cfg.TEST.EPOCH), desc="Epochs", leave=False, ncols=80):
                acc, auc = get_accuracy(
                    model, 
                    data_loader=test_loader, 
                    cfg=cfg,
                    epoch=epoch, 
                    corruption=domain_name, 
                    severity=severity
                )
            h_score = 2 * acc * auc / (acc + auc)
            accs.append(acc)
            aucs.append(auc)
            h_scores.append(h_score)
            log_message = (
                f"{cfg.CORRUPTION.ID_DATASET} with {cfg.CORRUPTION.OOD_DATASET} [#samples={len(testset)}][{domain_name}]"
                f": acc: {acc:.2%}, auc: {auc:.2%}, h-score: {h_score:.2%}"
            )
            logger.info(log_message)
            tqdm.write(log_message)

        # 计算并记录每个污染类型的平均指标
        mean_acc = np.mean(accs)
        mean_auc = np.mean(aucs)
        mean_h_score = np.mean(h_scores)
        mean_message = (
            f"Mean Acc: {mean_acc:.2%}, "
            f"Mean AUC: {mean_auc:.2%}, "
            f"Mean H-score: {mean_h_score:.2%}"
        )
        logger.info(mean_message)
        tqdm.write(mean_message)

    return {
        "accs": accs,
        "aucs": aucs,
        "h_scores": h_scores
    }


if __name__ == "__main__":
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'evaluation_os'
    load_cfg_fom_args(args.cfg, args.output_dir)
    merge_cfg_from_args(cfg, args)
    cfg.CORRUPTION.SOURCE_DOMAIN = cfg.CORRUPTION.SOURCE_DOMAINS[0]
    logger.info(cfg)
    validation(cfg)
