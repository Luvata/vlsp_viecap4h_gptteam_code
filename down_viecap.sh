gdown https://drive.google.com/uc?id=1lbOTlksNA5a97_Ydqh84TE6Dm85Rsy60
gdown https://drive.google.com/uc?id=1lKqdtekrhzlf7duVs34IQPVLROowoLoR
gdown https://drive.google.com/uc?id=1rCDniCZNgaJ7WQUzPpzEwuXW5_WNyave
unzip viecap4h-public-train.zip
unzip vietcap4h-public-test.zip
unzip -qq vietcap4h-private-test.zip
unzip -qq viecap4h-public-train/images_train.zip
unzip -qq vietcap4h-public-test/images_public_test.zip
mkdir viecap
mv images_train viecap
mv images_public_test viecap
mv vietcap4h-private-test/images viecap/images_private_test
mv vietcap4h-private-test/private_sample_sub.json viecap
mv viecap4h-public-train/train_captions.json viecap
mv vietcap4h-public-test/sample_submission.json viecap
rm *.zip
rm -rf viecap4h-public-train/
rm -rf vietcap4h-public-test/
gdown https://drive.google.com/uc?id=1_KME6Zj3NQQk6jb0nRfepsFKEiPTM7Hf
unzip -qq b16.zip
mv b16/*.pt .
rm -rf b16/
