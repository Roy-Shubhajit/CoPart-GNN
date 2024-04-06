#To overcome over-fitting, we fine-tuned the hyperparameter settings in the few-shot regime.
#To overcome over-smoothing, we fine-tuned the hyperparameter settings when the coarsening ratio is 0.1.
#example, the coarsening ratio is 0.5
python train.py --dataset cora --experiment fixed --coarsening_ratio 0.5
python train.py --dataset cora --experiment few --epoch1 100 --coarsening_ratio 0.5
python train.py --dataset citeseer --experiment fixed --epoch1 200 --coarsening_ratio 0.5
python train.py --dataset pubmed --experiment fixed --epoch1 200 --coarsening_ratio 0.5
python train.py --dataset pubmed --experiment few --epoch1 60 --coarsening_ratio 0.5
python train.py --dataset dblp --experiment random --epoch1 50 --coarsening_ratio 0.5
python train.py --dataset Physics --experiment random --epoch1 200 --lr 0.001 --weight_decay 0 --coarsening_ratio 0.5
