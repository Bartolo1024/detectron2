from . import apex_trainer, coco_eval_trainer


class TrainerCompose(coco_eval_trainer.COCOEvalTrainer, apex_trainer.ApexTrainer):
    pass
