# chmod +x call_main.sh
# sh call_main.sh
#!/usr/bin/env bash

for i in --emb_dim 6 9 12 15 18 21 24 27 30
do
  python main_gae.py --emb_dim $i
done

#for i in --emb_dim 6 9 12 15 18 21 24 27 30
#do
#  python main_gaec_cat.py --emb_dim $i
#done
#
#for i in --emb_dim 6 9 12 15 18 21 24 27 30
#do
#  python main_saec_cat.py --emb_dim $i
#done
#
#for i in --emb_dim 6 9 12 15 18 21 24 27 30
#do
#  python main_sae.py --emb_dim $i
#done
#
#for i in --emb_dim 6 9 12 15 18 21 24 27 30
#do
#  python lp_random.py --emb_dim $i
#done
#
#for i in --emb_dim 6 9 12 15 18 21 24 27 30
#do
#  python lp_random_cat.py --emb_dim $i
#done


