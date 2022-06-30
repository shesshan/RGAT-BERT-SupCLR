GPU_ID=2
num_train_epochs=30
alpha_2=0.5
sup_temp=0.1
# RGAT-BERT
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_bert --spc --embedding_type bert --output_dir data/output-gcn --dropout 0.3 --hidden_size 200 --learning_rate 5e-5 --cuda_id $GPU_ID --alpha_2 $alpha_2 --num_train_epochs $num_train_epochs --sup_temp $sup_temp # R-GAT+BERT in restaurant
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_bert --embedding_type bert --dataset_name laptop --output_dir data/output-gcn-laptop --dropout 0.3 --num_heads 7 --hidden_size 200 --learning_rate 5e-5 --cuda_id $GPU_ID --alpha_2 $alpha_2 --num_train_epochs $num_train_epochs --sup_temp $sup_temp # R-GAT+BERT in laptop
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_bert --embedding_type bert --dataset_name twitter --output_dir data/output-gcn-twitter --dropout 0.2  --hidden_size 200 --learning_rate 5e-5 --cuda_id $GPU_ID --alpha_2 $alpha_2 --num_train_epochs $num_train_epochs --sup_temp $sup_temp # R-GAT+BERT in twitter
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_bert --embedding_type bert --dataset_name rest15 --output_dir data/output-gcn-res15 --dropout 0.3  --hidden_size 200 --learning_rate 5e-5 --cuda_id $GPU_ID --alpha_2 $alpha_2 --num_train_epochs $num_train_epochs --sup_temp $sup_temp # R-GAT+BERT in res15
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_bert --embedding_type bert --dataset_name rest16 --output_dir data/output-gcn-res16 --dropout 0.3  --hidden_size 200 --learning_rate 5e-5 --cuda_id $GPU_ID --alpha_2 $alpha_2 --num_train_epochs $num_train_epochs --sup_temp $sup_temp # R-GAT+BERT in res16
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_bert --embedding_type bert --dataset_name mams --output_dir data/output-gcn-mams --dropout 0.3  --hidden_size 200 --learning_rate 5e-5 --cuda_id $GPU_ID --alpha_2 $alpha_2 --num_train_epochs $num_train_epochs --sup_temp $sup_temp # R-GAT+BERT in mams

# RGAT
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_our --highway --num_heads 7 --dropout 0.8 --num_train_epochs $num_train_epochs # R-GAT in restaurant
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_our --dataset_name laptop --output_dir data/output-gcn-laptop --highway --num_heads 9 --per_gpu_train_batch_size 32 --dropout 0.7 --num_layers 3 --hidden_size 400 --final_hidden_size 400 --num_train_epochs $num_train_epochs # R-GAT in laptop
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_our --dataset_name twitter --output_dir data/output-gcn-twitter --highway --num_heads 9 --per_gpu_train_batch_size 8 --dropout 0.6 --num_mlps 1 --final_hidden_size 400 --num_train_epochs $num_train_epochs # R-GAT in twitter
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_our --highway --dataset_name rest15 --output_dir data/output-gcn-res15 --num_heads 7 --dropout 0.8 --num_train_epochs $num_train_epochs # R-GAT in res15
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_our --highway --dataset_name rest16 --output_dir data/output-gcn-res16 --num_heads 7 --dropout 0.8 --num_train_epochs $num_train_epochs # R-GAT in res16
#CUDA_VISIBLE_DEVICES=$GPU_ID python run.py --gat_our --highway --dataset_name mams --output_dir data/output-gcn-mams --num_heads 9 --dropout 0.7 --num_train_epochs $num_train_epochs # R-GAT in mams