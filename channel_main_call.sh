# chmod +x channel_main_call.sh
# sh channel_main_call.sh
#!/usr/bin/env bash

for i in --emb_dim 6 9 12 15 18 21 24 27 30
do
  python main_channel_cat_gaec.py --emb_dim $i
done

for i in --emb_dim 6 9 12 15 18 21 24 27 30
do
  python main_channel_cat_saec.py --emb_dim $i
done

for i in --emb_dim 6 9 12 15 18 21 24 27 30
do
  python main_channel_gae.py --emb_dim $i
done

for i in --emb_dim 6 9 12 15 18 21 24 27 30
do
  python main_channel_sae.py --emb_dim $i
done

for i in --emb_dim 6 9 12 15 18 21 24 27 30
do
  python main_l1min_random.py --emb_dim $i
done

for i in --emb_dim 6 9 12 15 18 21 24 27 30
do
  python main_l1min_random_cat.py --emb_dim $i
done

for i in --emb_dim 9 12 15
do
  python main_channel_cat0_gae.py --emb_dim $i
done

for i in --emb_dim 9 12 15
do
  python main_channel_cat0_sae.py --emb_dim $i
done

for i in --emb_dim 9 12 15
do
  python main_l1min_random_cat0.py --emb_dim $i
done

