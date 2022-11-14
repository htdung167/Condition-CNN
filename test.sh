python3 test.py --model Condition \
    --weights_path \
    --batch 128 \
    --imgsz 224 \
    --data_path ./data/fashion-dataset/images \
    --csv_test ./data/fashion_product_test.csv \
    --master_column masterCategory \
    --sub_column subCategory \
    --article_column articleType \
    --filepath_column filepath \