python3 train.py --model Condition \
    --epochs 20 \
    --batch 128 \
    --imgsz 224 \
    --data_path ./data/fashion-dataset/images \
    --csv_train ./data/shopee_fashion_train.csv \
    --csv_val ./data/shopee_fashion_val.csv \
    --csv_test ./data/shopee_fashion_test.csv \
    --master_column MasterCategoryId \
    --sub_column SubCategoryId \
    --article_column ArticalTypeId \
    --filepath_column FilePath