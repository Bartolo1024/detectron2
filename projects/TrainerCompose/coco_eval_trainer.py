import os
import logging
from collections import OrderedDict
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


class COCOEvalTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([COCOEvaluator(dataset_name, cfg, True, output_folder)])

    @classmethod
    def test_with_tta(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
