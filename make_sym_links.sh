echo "making sym links for model.onnx"

cd resnet/1/
ln -s v56_9.yaml_model_20250507_135857_9.onnx model.onnx 
cd ../../

cd rine_vit/1/
ln -s rine_vit_large_last.onnx model.onnx
cd ../../
