import datasets
ds = datasets.load_dataset('joyheyueya/0512_ilcr_train_final_pair100k_rl_single', split='train')
ds = ds.train_test_split(0.1, seed=42)

ds['train'].to_parquet('/home/ubuntu/rl/train.parquet')
ds['test'].to_parquet('/home/ubuntu/rl/test.parquet')