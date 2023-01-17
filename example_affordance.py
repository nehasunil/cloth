import numpy as np
import torch
from logger import Logger, draw_grasp_point
from perception.infer_affordance.affordance import Affordance
from perception.infer_affordance.affordance_trainer import AffordanceTrainer


def train_affordance():
    logger = Logger(logging_directory="data/affordance/20220602", iteration=0)
    afford = Affordance()

    # fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/affordance_ft_0602"
    # afford.model.load_state_dict(torch.load(fn_afford, map_location=afford.device))

    model = afford.model
    device = afford.device

    affordance_trainer = AffordanceTrainer(logger, model, device, afford)
    # affordance_trainer.best_eval = 0.775
    affordance_trainer.train(learning_rate=1e-4, nb_epoch=200)
    # print(affordance_trainer.evaluate())


def eval_affordance():
    logger = Logger(logging_directory="data/affordance/20220602", iteration=0)
    afford = Affordance()

    # fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/finetune_0610/best_1_0.700.pt"
    # fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/nosim_0609/best_19_0.600.pt"
    # afford.model.load_state_dict(torch.load(fn_afford, map_location=afford.device))

    model = afford.model
    device = afford.device

    affordance_trainer = AffordanceTrainer(logger, model, device, afford)
    for k in range(40, 41):
        precisionk = affordance_trainer.evaluate(k=k)
        print(f"precision@{k}", precisionk)
    # for thresh in np.arange(0.1, 0.6, 0.01):
    #     print(
    #         "Threshold: ",
    #         thresh,
    #         "    Accuracy: ",
    #         affordance_trainer.evaluate_accuracy(threshold=thresh),
    #     )


if __name__ == "__main__":
    # train_affordance()
    eval_affordance()
