python3 test.py --model Condition_FinetuneFromOtherDataset \
    --weights_path weights_ConditionCNN_Finetune_Shopee_Imbalance_19_11_2022/Condition_FinetuneFromOtherDataset_best_weights.h5 \
    --batch 128 \
    --imgsz 224 \
    --data_path ./dataset \
    --csv_test ./data/shopee_fashion_test.csv \
    --master_column MasterCategoryId \
    --sub_column SubCategoryId \
    --article_column ArticalTypeId \
    --filepath_column FilePath 