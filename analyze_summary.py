import pandas as pd

summary_path = "./logs/standard/Sampled_ImageNet/summary_in_20230830.log"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


summary_df = pd.read_json(summary_path, lines=True)

summary_df[['model', 'num_concepts', 'norm_concepts', 'norm_summary',
            'grad_factor', 'loss_sparsity_weight', 'loss_diversity_weight',
            'best_val_acc', 'best_val_acc_major',
            'best_val_acc_minor', 'best_epoch', 'best_checkpoint_path']]
